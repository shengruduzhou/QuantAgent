"""Lifecycle / conviction-aware top_k resolver.

The default V7 optimiser holds ``top_k`` constant. That is the right starting
point for back-testing because it gives a clean lever to study. The optional
dynamic resolver remains a research feature and uses deterministic state
variables: lifecycle stage, policy strength, and **prediction-score
dispersion**.

Score dispersion is deliberately not called information coefficient. IC is an
ex-post association between a signal and subsequent returns; dispersion is an
ex-ante property of the score cross-section and cannot establish predictive
quality. In particular, a signal and its sign-inverted (wrong-way) version have
the same dispersion.

This module produces a per-date ``top_k`` recommendation in the closed interval
``[top_k_min, top_k_max]`` and returns an audit payload describing each
contribution. Production promotion of an adaptive Top-K rule still requires the
same pre-registration / OOS / search-budget governance as any other strategy
parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    score_dispersion_strong_threshold: float = 0.05
    score_dispersion_weak_threshold: float = 0.02
    score_dispersion_strong_bonus: int = 6
    score_dispersion_weak_penalty: int = -6
    policy_strength_bonus: float = 8.0
    keep_min_floor: bool = True


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
        stage = stages.value_counts().idxmax() if not stages.empty else None
    if "policy_strength" in theme_signals.columns:
        ps = pd.to_numeric(theme_signals["policy_strength"], errors="coerce").dropna()
        policy_strength = float(ps.mean()) if not ps.empty else None
    else:
        policy_strength = None
    return stage, policy_strength


def _score_dispersion_cross_sectional(predictions_row: pd.Series | None) -> float:
    """Scale-normalised dispersion of the available prediction scores.

    This quantity is intentionally *not* IC. It uses no future returns and has
    no sign/direction information about predictive correctness.
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
    lifecycle_contribution = int(lifecycle_map.get(stage, 0)) if stage is not None else 0
    contributions["lifecycle"] = lifecycle_contribution

    policy_contribution = 0
    if policy_strength is not None:
        policy_contribution = int(round(float(cfg.policy_strength_bonus) * float(policy_strength)))
    contributions["policy_strength"] = policy_contribution

    score_dispersion = _score_dispersion_cross_sectional(predictions_for_date)
    if score_dispersion >= cfg.score_dispersion_strong_threshold:
        dispersion_contribution = int(cfg.score_dispersion_strong_bonus)
    elif score_dispersion <= cfg.score_dispersion_weak_threshold:
        dispersion_contribution = int(cfg.score_dispersion_weak_penalty)
    else:
        dispersion_contribution = 0
    contributions["score_dispersion"] = dispersion_contribution

    raw = base + lifecycle_contribution + policy_contribution + dispersion_contribution
    contributions["raw_sum"] = int(raw)

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
        "score_dispersion": float(score_dispersion),
        "score_dispersion_semantics": "prediction_cross_section_only_not_information_coefficient",
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
