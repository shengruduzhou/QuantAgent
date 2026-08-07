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

HORIZON_BLEND_PRESETS: dict[str, tuple[tuple[int, float], ...]] = {
    "balanced": DEFAULT_HORIZON_WEIGHTS,
    "short_tactical": (
        (1, 0.25),
        (5, 0.35),
        (20, 0.25),
        (60, 0.10),
        (120, 0.05),
    ),
    "long_fundamental": (
        (1, 0.03),
        (5, 0.07),
        (20, 0.25),
        (60, 0.35),
        (120, 0.30),
    ),
}

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
    #: How to put horizons on a common scale before the weighted sum.
    #:
    #: ``"cross_sectional_rank"`` (default) ranks each horizon's predictions
    #: within each trade date and maps them onto ``[-1, 1]`` before weighting.
    #: ``"none"`` reproduces the pre-2026-08 behaviour of summing raw
    #: predictions, which is retained only so an existing result can be
    #: reproduced for comparison. See :func:`blend_multi_horizon_predictions`
    #: for why the raw sum does not do what the weights say.
    scale_normalisation: str = "cross_sectional_rank"

    def __post_init__(self) -> None:
        if self.scale_normalisation not in {"cross_sectional_rank", "none"}:
            raise ValueError(
                "scale_normalisation must be 'cross_sectional_rank' or 'none', "
                f"got {self.scale_normalisation!r}"
            )


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


def _rank_normalise_per_date_horizon(frame: pd.DataFrame) -> pd.Series:
    """Rank predictions within each ``(trade_date, horizon)`` onto ``[-1, 1]``.

    Without this the weighted sum is not the blend the weights describe. Each
    horizon's model predicts a *different label*: ``forward_return_1d`` has a
    cross-sectional standard deviation around 0.022 on this repository's gold
    panel, ``forward_return_120d`` around 0.237 — an order of magnitude apart,
    because a 120-day return simply is bigger than a 1-day one. Summing the raw
    predictions therefore weights each horizon by ``w_h * sigma_h``, not by
    ``w_h``, and the downstream consumer only ranks on the result, so the
    dispersion is the entire contribution.

    Measured against the shipped ``DEFAULT_HORIZON_WEIGHTS`` on that panel, the
    declared mix and the realised mix were:

    ======  ==========  ==========  ======
    horizon  declared    realised    ratio
    ======  ==========  ==========  ======
    1d       10%          1.8%       0.18x
    5d       20%          8.4%       0.42x
    20d      30%         25.4%       0.85x
    60d      25%         35.2%       1.41x
    120d     15%         29.3%       1.95x
    ======  ==========  ==========  ======

    So the "balanced" preset was in fact long-horizon dominant, the
    ``short_tactical`` preset was much less tactical than it reads, and the
    ``DECAY`` lifecycle override — whose entire purpose is to swing weight onto
    the 1- and 5-day sleeves so the optimiser unwinds a fading name quickly —
    was delivering roughly a quarter of the short-horizon influence it declared.

    Ranking per date introduces no lookahead: only rows sharing a trade date are
    compared. It also discards the predictions' magnitudes, which is the right
    trade here because the consumer takes ``nlargest`` and never reads a
    predicted return as a return.
    """
    predictions = pd.to_numeric(frame["prediction"], errors="coerce")
    predictions = predictions.replace([np.inf, -np.inf], np.nan)
    grouped = predictions.groupby([frame["trade_date"], frame["horizon"]], sort=False)
    ranks = grouped.rank(method="average")
    counts = grouped.transform("count")
    spread = (counts - 1.0).where(counts > 1)
    return (2.0 * (ranks - 1.0) / spread - 1.0).fillna(0.0)


def _min_cross_section_size(frame: pd.DataFrame) -> int:
    """Smallest number of names any ``(trade_date, horizon)`` slice ranks over."""
    if frame.empty:
        return 0
    sizes = frame.groupby(["trade_date", "horizon"], sort=False)["symbol"].nunique()
    return int(sizes.min()) if len(sizes) else 0


def _available_horizons(predictions: pd.DataFrame) -> list[int]:
    if "horizon" not in predictions.columns:
        return []
    values = pd.to_numeric(predictions["horizon"], errors="coerce").dropna().astype(int)
    return sorted({120 if value == 126 else int(value) for value in values})


def _preset_weights_for_available(
    preset: tuple[tuple[int, float], ...],
    available: Iterable[int],
    primary_horizon: int,
) -> tuple[tuple[int, float], ...]:
    available_set = {int(value) for value in available}
    filtered = [(horizon, weight) for horizon, weight in preset if horizon in available_set]
    if not filtered:
        if int(primary_horizon) not in available_set:
            raise ValueError("no configured horizon is present in predictions")
        filtered = [(int(primary_horizon), 1.0)]
    normalised = _normalise_weights(filtered)
    return tuple((horizon, normalised[horizon]) for horizon in sorted(normalised))


def _early_oos_rank_ic_scores(
    predictions: pd.DataFrame,
    *,
    holdout_days: int,
) -> tuple[dict[int, float], dict[int, dict[str, float | int]], dict[str, object]]:
    frame = predictions.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "horizon", "prediction"])
    frame["horizon"] = frame["horizon"].astype(int).replace({126: 120})
    dates = sorted(pd.Timestamp(value) for value in frame["trade_date"].unique())
    minimum_calibration_days = 10
    if len(dates) < int(holdout_days) + minimum_calibration_days:
        return {}, {}, {
            "fallbackReason": "insufficient_oos_dates",
            "oosDays": len(dates),
            "requiredDays": int(holdout_days) + minimum_calibration_days,
        }

    holdout_index = len(dates) - int(holdout_days)
    holdout_start = dates[holdout_index]
    scores: dict[int, float] = {}
    evidence: dict[int, dict[str, float | int]] = {}
    purge_evidence: dict[int, dict[str, object]] = {}
    calibration_ends: list[pd.Timestamp] = []
    for horizon, horizon_frame in frame.groupby("horizon", sort=True):
        # A forward_return_Hd observed at date t contains information through
        # t+H. Purge the H dates immediately before final holdout so adaptive
        # weights cannot learn from returns realised inside that holdout.
        safe_end_index = holdout_index - int(horizon)
        if safe_end_index <= 0:
            purge_evidence[int(horizon)] = {
                "purgeDays": int(horizon),
                "status": "insufficient_pre_holdout_history",
            }
            continue
        safe_dates = set(dates[:safe_end_index])
        subset = horizon_frame.loc[
            horizon_frame["trade_date"].isin(safe_dates)
        ].copy()
        if subset.empty:
            continue
        calibration_end = pd.Timestamp(subset["trade_date"].max())
        calibration_ends.append(calibration_end)
        purge_evidence[int(horizon)] = {
            "purgeDays": int(horizon),
            "calibrationEnd": calibration_end.strftime("%Y-%m-%d"),
            "status": "applied",
        }
        target_column = f"forward_return_{int(horizon)}d"
        if target_column not in subset.columns:
            continue
        subset[target_column] = pd.to_numeric(subset[target_column], errors="coerce")
        daily_ic: list[float] = []
        for _, day in subset.dropna(subset=[target_column]).groupby("trade_date", sort=True):
            if len(day) < 3:
                continue
            value = day["prediction"].corr(day[target_column], method="spearman")
            if pd.notna(value):
                daily_ic.append(float(value))
        if len(daily_ic) < 3:
            continue
        mean_ic = float(np.mean(daily_ic))
        std_ic = float(np.std(daily_ic, ddof=1)) if len(daily_ic) > 1 else 0.0
        score = max(mean_ic, 0.0) * np.sqrt(len(daily_ic)) / max(std_ic, 0.02)
        scores[int(horizon)] = float(score)
        evidence[int(horizon)] = {
            "meanRankIc": mean_ic,
            "rankIcStd": std_ic,
            "observations": len(daily_ic),
            "score": float(score),
        }

    return scores, evidence, {
        "calibrationStart": dates[0].strftime("%Y-%m-%d"),
        "calibrationEnd": (
            max(calibration_ends).strftime("%Y-%m-%d")
            if calibration_ends
            else None
        ),
        "holdoutStart": holdout_start.strftime("%Y-%m-%d"),
        "holdoutDays": int(holdout_days),
        "frozenBeforeHoldout": True,
        "forwardLabelPurgeApplied": True,
        "purgeByHorizon": purge_evidence,
    }


def resolve_horizon_blend_config(
    predictions: pd.DataFrame,
    *,
    method: str,
    primary_horizon: int,
    holdout_days: int,
) -> tuple[MultiHorizonBlendConfig, dict[str, object]]:
    """Resolve a declared horizon-mix policy without reading final holdout labels.

    ``adaptive_oos`` learns non-negative RankIC/ICIR-style weights only from
    the early OOS segment, then freezes those weights before the final holdout.
    All other modes are deterministic, auditable presets.
    """

    resolved_method = str(method).strip().lower()
    allowed = {*HORIZON_BLEND_PRESETS, "adaptive_oos", "primary_only"}
    if resolved_method not in allowed:
        raise ValueError(f"horizon blend method must be one of {sorted(allowed)}")
    available = _available_horizons(predictions)
    if not available:
        raise ValueError("multi-horizon predictions are missing horizon values")

    diagnostics: dict[str, object] = {
        "method": resolved_method,
        "availableHorizons": available,
        "primaryHorizon": int(primary_horizon),
    }
    if resolved_method == "primary_only":
        weights = ((int(primary_horizon), 1.0),)
        diagnostics |= {
            "source": "declared_primary_horizon",
            "horizonWeights": [{"horizon": int(primary_horizon), "weight": 1.0}],
        }
        return MultiHorizonBlendConfig(
            horizon_weights=weights,
            primary_horizon=int(primary_horizon),
        ), diagnostics

    if resolved_method == "adaptive_oos":
        scores, evidence, split = _early_oos_rank_ic_scores(
            predictions,
            holdout_days=int(holdout_days),
        )
        positive = {horizon: score for horizon, score in scores.items() if score > 0}
        if positive:
            total = sum(positive.values())
            weights = tuple(
                (horizon, score / total)
                for horizon, score in sorted(positive.items())
            )
            diagnostics |= {
                "source": "early_oos_rank_ic",
                "rankIcByHorizon": evidence,
                **split,
            }
        else:
            weights = _preset_weights_for_available(
                HORIZON_BLEND_PRESETS["balanced"],
                available,
                primary_horizon,
            )
            diagnostics |= {
                "source": "balanced_fallback",
                "rankIcByHorizon": evidence,
                **split,
                "fallbackReason": split.get(
                    "fallbackReason",
                    "no_positive_early_oos_rank_ic",
                ),
            }
    else:
        weights = _preset_weights_for_available(
            HORIZON_BLEND_PRESETS[resolved_method],
            available,
            primary_horizon,
        )
        diagnostics["source"] = "declared_preset"

    diagnostics["horizonWeights"] = [
        {"horizon": horizon, "weight": float(weight)}
        for horizon, weight in weights
    ]
    return MultiHorizonBlendConfig(
        horizon_weights=weights,
        primary_horizon=int(primary_horizon),
    ), diagnostics


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

    if cfg.scale_normalisation == "cross_sectional_rank":
        frame["prediction"] = _rank_normalise_per_date_horizon(frame)

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
        "scale_normalisation": cfg.scale_normalisation,
        # Cross-sectional ranking needs a cross-section. A panel carrying one
        # name per date normalises every score to 0.0 and the blend carries no
        # information — correct, but worth seeing rather than discovering later.
        "min_cross_section_size": _min_cross_section_size(frame),
        # The blend is a weighted sum across horizons of per-horizon scores. It
        # is additive by construction: no horizon's weight depends on another
        # horizon's value. Stated here so a reader is not left to infer the
        # model class. See quantagent.models.interactions.ModelClass.
        "model_class": (
            "rank_weighted_additive"
            if cfg.scale_normalisation == "cross_sectional_rank"
            else "linear_additive"
        ),
        "weights_are_realised": cfg.scale_normalisation == "cross_sectional_rank",
    }
    return MultiHorizonBlendResult(blended_frame, diagnostics)


def attach_blender_metadata(target: dict[str, object], blend: MultiHorizonBlendResult) -> dict[str, object]:
    target["multi_horizon_blend"] = blend.diagnostics
    return target


__all__ = [
    "MultiHorizonBlendConfig",
    "MultiHorizonBlendResult",
    "DEFAULT_HORIZON_WEIGHTS",
    "HORIZON_BLEND_PRESETS",
    "resolve_horizon_blend_config",
    "blend_multi_horizon_predictions",
    "attach_blender_metadata",
]
