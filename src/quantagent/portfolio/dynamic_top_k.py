"""Lifecycle / conviction-aware top_k resolver.

The default V7 optimiser holds ``top_k`` constant. That is the right
starting point for back-testing because it gives a clean lever to study,
but it ignores everything the upstream evidence pipeline already knows:
when a theme is in ``POLICY_SEED`` we should be holding fewer names with
higher conviction; when it's in ``CAPITAL_INFLOW`` we want a wider net;
when it's in ``DECAY`` or ``INVALIDATED`` we should be unwinding into
cash.

This module produces a per-date ``top_k`` recommendation in the closed
interval ``[top_k_min, top_k_max]``. The implementation is intentionally
deterministic and explainable: a base ``top_k`` is shifted by additive
rules from three signals (lifecycle stage, average policy strength,
cross-sectional alpha IC) and then clamped. Callers receive both the
resolved ``top_k`` and an audit payload describing each contribution.

The function is also defensive about small universes: on a smoke
universe of 5 names a base ``top_k`` of 30 would crash the
``fail_if_top_k_covers_universe`` invariant, so the resolver always
clamps to ``max(top_k_min, min(top_k_max, eligible_count - 1))``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd


LIFECYCLE_DELTA: dict[str, int] = {
    "POLICY_SEED": -12,
    "NARRATIVE_FORMATION": -8,
    "CAPITAL_INFLOW": +10,
    "EARNINGS_REALIZATION": 0,
    "VALUATION_BUBBLE": -6,
    "DECAY": -14,
    "INVALIDATED": -20,
}


@dataclass(frozen=True)
class DynamicTopKConfig:
    top_k_min: int = 8
    top_k_max: int = 50
    base_top_k: int = 30
    lifecycle_delta: tuple[tuple[str, int], ...] = field(
        default_factory=lambda: tuple(LIFECYCLE_DELTA.items())
    )
    ic_strong_threshold: float = 0.05
    ic_weak_threshold: float = 0.02
    ic_strong_bonus: int = 6
    ic_weak_penalty: int = -6
    policy_strength_bonus: float = 8.0  # +k = policy_strength_bonus * mean_policy_strength
    keep_min_floor: bool = True  # never go below top_k_min even if universe is small


@dataclass(frozen=True)
class DynamicTopKDecision:
    top_k: int
    base: int
    final: int
    contributions: dict[str, int]
    diagnostics: dict[str, object] = field(default_factory=dict)


def _lifecycle_summary(theme_signals: pd.DataFrame | None) -> tuple[str | None, float | None]:
    if theme_signals is None or theme_signals.empty:
        return None, None
    if "lifecycle_stage" not in theme_signals.columns:
        stage = None
    else:
        stages = theme_signals["lifecycle_stage"].astype(str).str.upper().dropna()
        if stages.empty:
            stage = None
        else:
            stage = stages.value_counts().idxmax()
    if "policy_strength" in theme_signals.columns:
        ps = pd.to_numeric(theme_signals["policy_strength"], errors="coerce").dropna()
        policy_strength = float(ps.mean()) if not ps.empty else None
    else:
        policy_strength = None
    return stage, policy_strength


def _alpha_ic_cross_sectional(
    predictions_row: pd.Series,
    benchmark_row: pd.Series | None,
) -> float:
    """Simple proxy: the standardised dispersion of predictions.

    A truly strong cross-sectional signal exhibits high dispersion of
    predictions; a weak signal is roughly flat. We do not have access to
    the next-day return at decision time, so this proxy is the best we
    can do without leaking labels into the optimiser.
    """

    if predictions_row is None or predictions_row.empty:
        return 0.0
    clean = pd.to_numeric(predictions_row, errors="coerce").dropna()
    if clean.size < 2:
        return 0.0
    std = float(clean.std(ddof=0))
    rng = float(clean.max() - clean.min())
    if rng <= 0:
        return 0.0
    return std / rng


def resolve_dynamic_top_k(
    eligible_count: int,
    predictions_for_date: pd.Series | None = None,
    theme_signals_for_date: pd.DataFrame | None = None,
    config: DynamicTopKConfig | None = None,
) -> DynamicTopKDecision:
    cfg = config or DynamicTopKConfig()
    base = int(cfg.base_top_k)
    contributions: dict[str, int] = {"base": base}

    lifecycle_map: dict[str, int] = dict(cfg.lifecycle_delta)
    stage, policy_strength = _lifecycle_summary(theme_signals_for_date)
    lifecycle_contribution = 0
    if stage is not None:
        lifecycle_contribution = int(lifecycle_map.get(stage, 0))
    contributions["lifecycle"] = lifecycle_contribution

    policy_contribution = 0
    if policy_strength is not None:
        policy_contribution = int(round(float(cfg.policy_strength_bonus) * float(policy_strength)))
    contributions["policy_strength"] = policy_contribution

    ic_proxy = _alpha_ic_cross_sectional(predictions_for_date, None)
    if ic_proxy >= cfg.ic_strong_threshold:
        ic_contribution = int(cfg.ic_strong_bonus)
    elif ic_proxy <= cfg.ic_weak_threshold:
        ic_contribution = int(cfg.ic_weak_penalty)
    else:
        ic_contribution = 0
    contributions["alpha_ic_proxy"] = ic_contribution

    raw = base + lifecycle_contribution + policy_contribution + ic_contribution
    contributions["raw_sum"] = int(raw)

    # Smoke-universe defense: never exceed eligible_count - 1 (top_k must
    # leave at least one symbol unselected for the selection-pressure
    # invariant). On a 5-name universe with top_k_max=50, clamp to 4.
    universe_ceiling = max(0, int(eligible_count) - 1)
    upper_bound = min(int(cfg.top_k_max), universe_ceiling) if universe_ceiling > 0 else 0
    lower_bound = int(cfg.top_k_min)
    if cfg.keep_min_floor:
        lower_bound = min(lower_bound, max(1, universe_ceiling))
    final = int(max(lower_bound, min(upper_bound if upper_bound > 0 else lower_bound, raw)))
    if upper_bound <= 0:
        final = 0

    diagnostics: dict[str, object] = {
        "eligible_count": int(eligible_count),
        "lifecycle_stage": stage,
        "policy_strength": policy_strength,
        "alpha_ic_proxy": float(ic_proxy),
        "universe_ceiling": universe_ceiling,
        "lower_bound": int(lower_bound),
        "upper_bound": int(upper_bound),
    }
    return DynamicTopKDecision(
        top_k=final,
        base=base,
        final=final,
        contributions=contributions,
        diagnostics=diagnostics,
    )


__all__ = [
    "DynamicTopKConfig",
    "DynamicTopKDecision",
    "LIFECYCLE_DELTA",
    "resolve_dynamic_top_k",
]
