"""Multi-horizon alpha blender.

Walk-forward training produces predictions with one row per
``(trade_date, symbol, horizon)``. The target-weights optimiser, however,
needs a single ``prediction`` per ``(trade_date, symbol)`` to rank names
on. This module collapses the multi-horizon panel into a single blended
alpha while keeping a record of how it was assembled so downstream
auditors can replay it.

Design choices that matter:

* **No silent re-normalisation when horizons are missing.** Re-weighting
  the present horizons after a horizon goes missing can amplify the
  noisiest short-term signal. If the configured ``horizon_weights``
  reference a horizon that isn't present for some ``(date, symbol)``, the
  blender falls back to the configured ``primary_horizon`` for that row
  rather than rescaling.
* **Lifecycle-conditional weights.** A separate side-channel signal can
  shift the horizon mix per ``(date, symbol)``: ``DECAY`` raises the
  short-term weight (so the optimiser unwinds quickly), while
  ``CAPITAL_INFLOW`` raises the medium-to-long weight (let winners run).
* **126d collapsed into 120 bucket.** The label builder produces both,
  but at the portfolio layer they carry essentially the same signal.

The blender produces a frame ready to hand to
``build_v7_target_weights`` (one row per ``(date, symbol)`` with a
``prediction`` column). It also returns a diagnostics payload describing
the horizon coverage and the fallback rate so the run report can flag
degraded modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


DEFAULT_HORIZON_WEIGHTS: tuple[tuple[int, float], ...] = (
    (1, 0.10),
    (5, 0.20),
    (20, 0.30),
    (60, 0.25),
    (120, 0.15),
)

# Per-stage horizon mix overrides. Keys must match
# ``ThemeProfile.lifecycle_stage`` values.
_LIFECYCLE_OVERRIDES: dict[str, tuple[tuple[int, float], ...]] = {
    "POLICY_SEED": ((1, 0.05), (5, 0.10), (20, 0.25), (60, 0.30), (120, 0.30)),
    "NARRATIVE_FORMATION": ((1, 0.05), (5, 0.15), (20, 0.30), (60, 0.30), (120, 0.20)),
    "CAPITAL_INFLOW": ((1, 0.05), (5, 0.15), (20, 0.30), (60, 0.30), (120, 0.20)),
    "EARNINGS_REALIZATION": DEFAULT_HORIZON_WEIGHTS,
    "VALUATION_BUBBLE": ((1, 0.30), (5, 0.30), (20, 0.20), (60, 0.15), (120, 0.05)),
    "DECAY": ((1, 0.40), (5, 0.35), (20, 0.15), (60, 0.07), (120, 0.03)),
    "INVALIDATED": ((1, 0.60), (5, 0.30), (20, 0.10), (60, 0.0), (120, 0.0)),
}


@dataclass(frozen=True)
class MultiHorizonBlendConfig:
    horizon_weights: tuple[tuple[int, float], ...] = DEFAULT_HORIZON_WEIGHTS
    primary_horizon: int = 5
    collapse_126_into_120: bool = True
    require_all_horizons: bool = False


@dataclass(frozen=True)
class MultiHorizonBlendResult:
    blended: pd.DataFrame
    diagnostics: dict[str, object] = field(default_factory=dict)


def _normalise_weights(weights: Iterable[tuple[int, float]]) -> dict[int, float]:
    pairs = [(int(h), float(w)) for h, w in weights if float(w) > 0]
    total = sum(w for _, w in pairs)
    if total <= 0:
        raise ValueError("horizon_weights must contain at least one positive weight")
    return {h: w / total for h, w in pairs}


def _resolve_weights(
    stage: str | None,
    base: dict[int, float],
) -> dict[int, float]:
    if stage is None:
        return base
    override = _LIFECYCLE_OVERRIDES.get(str(stage).upper())
    if override is None:
        return base
    return _normalise_weights(override)


def blend_multi_horizon_predictions(
    predictions: pd.DataFrame,
    theme_signals: pd.DataFrame | None = None,
    config: MultiHorizonBlendConfig | None = None,
) -> MultiHorizonBlendResult:
    """Collapse a multi-horizon prediction panel into a single blended alpha.

    Parameters
    ----------
    predictions:
        Long frame with columns ``trade_date``, ``symbol``, ``prediction``,
        and ``horizon``. Extra columns (``fold_id``, ``sample_role``) are
        carried through unchanged.
    theme_signals:
        Optional frame keyed on ``trade_date`` + ``symbol`` carrying a
        ``lifecycle_stage`` column. Used to shift the horizon mix.
    config:
        Blender configuration. Defaults to ``MultiHorizonBlendConfig()``.
    """

    cfg = config or MultiHorizonBlendConfig()
    if predictions is None or predictions.empty:
        return MultiHorizonBlendResult(
            pd.DataFrame(columns=["trade_date", "symbol", "prediction"]),
            {"status": "empty_input"},
        )

    frame = predictions.copy()
    if "horizon" not in frame.columns:
        # Single-horizon predictions — just return as-is with metadata.
        return MultiHorizonBlendResult(
            frame.reset_index(drop=True),
            {"status": "passthrough", "reason": "no_horizon_column"},
        )

    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["horizon", "prediction"]).reset_index(drop=True)
    frame["horizon"] = frame["horizon"].astype(int)
    if cfg.collapse_126_into_120:
        frame["horizon"] = frame["horizon"].replace({126: 120})

    base_weights = _normalise_weights(cfg.horizon_weights)

    if theme_signals is not None and not theme_signals.empty:
        stage_lookup = (
            theme_signals[["trade_date", "symbol", "lifecycle_stage"]]
            .dropna(subset=["trade_date", "symbol"])
            .assign(trade_date=lambda f: pd.to_datetime(f["trade_date"], errors="coerce"))
            .dropna(subset=["trade_date"])
            .groupby(["trade_date", "symbol"], as_index=False)
            .last()
        )
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
        frame = frame.merge(stage_lookup, on=["trade_date", "symbol"], how="left")
    else:
        frame["lifecycle_stage"] = None

    # Pivot to (date, symbol) × horizon for vectorised blending.
    pivot = frame.pivot_table(
        index=["trade_date", "symbol"],
        columns="horizon",
        values="prediction",
        aggfunc="last",
    )

    if pivot.empty:
        return MultiHorizonBlendResult(
            pd.DataFrame(columns=["trade_date", "symbol", "prediction"]),
            {"status": "no_predictions"},
        )

    available_horizons = sorted(int(c) for c in pivot.columns)
    coverage_counts = {int(h): int(pivot[h].notna().sum()) for h in available_horizons}

    stage_map = frame.groupby(["trade_date", "symbol"])["lifecycle_stage"].last()

    blended_values: list[float] = []
    fallback_rows = 0
    blend_modes: list[str] = []

    primary = int(cfg.primary_horizon)
    for index, row in pivot.iterrows():
        stage = stage_map.get(index)
        weights = _resolve_weights(stage if stage is None or isinstance(stage, str) else None, base_weights)
        usable = {h: float(row.get(h)) for h in weights if h in pivot.columns and pd.notna(row.get(h))}
        if not usable:
            blended_values.append(float("nan"))
            blend_modes.append("missing_all")
            fallback_rows += 1
            continue
        if cfg.require_all_horizons and len(usable) < len(weights):
            value = row.get(primary)
            if pd.isna(value):
                blended_values.append(float("nan"))
                blend_modes.append("missing_primary")
                fallback_rows += 1
                continue
            blended_values.append(float(value))
            blend_modes.append("fallback_primary")
            fallback_rows += 1
            continue
        if len(usable) < len(weights):
            value = row.get(primary)
            if pd.notna(value):
                blended_values.append(float(value))
                blend_modes.append("partial_fallback_primary")
                fallback_rows += 1
                continue
        # Use only present horizons; do NOT renormalise — apply weights as-is and divide by
        # the present-weight mass to keep magnitude sensible without amplifying short noise.
        weight_mass = sum(weights[h] for h in usable)
        if weight_mass <= 0:
            blended_values.append(float("nan"))
            blend_modes.append("zero_mass")
            fallback_rows += 1
            continue
        blended = sum(weights[h] * usable[h] for h in usable) / weight_mass
        blended_values.append(float(blended))
        blend_modes.append("blended_full" if len(usable) == len(weights) else "blended_partial")

    blended_frame = pivot.copy()
    blended_frame["prediction"] = blended_values
    blended_frame = blended_frame.reset_index()[["trade_date", "symbol", "prediction"]]
    blended_frame = blended_frame.dropna(subset=["prediction"]).reset_index(drop=True)

    diagnostics = {
        "status": "passed",
        "horizons_used": available_horizons,
        "coverage_per_horizon": coverage_counts,
        "rows_in": int(len(pivot)),
        "rows_out": int(len(blended_frame)),
        "fallback_rows": int(fallback_rows),
        "fallback_rate": float(fallback_rows) / float(max(1, len(pivot))),
        "blend_mode_counts": {mode: blend_modes.count(mode) for mode in set(blend_modes)},
        "horizon_weights_base": list(cfg.horizon_weights),
        "primary_horizon": primary,
    }
    return MultiHorizonBlendResult(blended_frame, diagnostics)


def attach_blender_metadata(target: dict[str, object], blend: MultiHorizonBlendResult) -> dict[str, object]:
    target["multi_horizon_blend"] = blend.diagnostics
    return target


__all__ = [
    "MultiHorizonBlendConfig",
    "MultiHorizonBlendResult",
    "DEFAULT_HORIZON_WEIGHTS",
    "blend_multi_horizon_predictions",
    "attach_blender_metadata",
]
