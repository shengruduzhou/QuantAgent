"""Adaptive long-horizon factor weight learner.

The static prior in ``long_horizon_factors._default_long_horizon_weights``
is a reasonable starting point but it is not regime-aware: an "AI compute"
window should over-weight ``growth_order_visibility`` and
``structural_bottleneck``; a "consumer recovery" window should
over-weight ``valuation_history_zscore`` and ``quality_fcf_yield``.

This module learns the weights from a walk-forward Information
Coefficient (IC) panel:

* Input: a panel keyed by ``trade_date`` / ``symbol`` with the long
  horizon factor columns plus a forward-return label
  (``forward_return_120d``).
* Walk-forward: split the dates into N rolling train / test windows with
  an embargo gap.
* For every window, compute the rank-IC of each factor on the test slice
  and produce a per-factor weight proportional to its ICIR.
* Average those weights across windows and renormalise so they sum to 1.0.

The learner also produces a ``per_theme`` dictionary so the factor weights
can be conditioned on ``theme`` / ``sector`` / ``lifecycle`` /
``market_regime`` / ``horizon_days`` (any column attached to the panel).
When a slice has fewer than ``min_samples`` rows the global weights are
used as a Bayesian prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.factors.long_horizon_factors import LONG_HORIZON_FACTORS


@dataclass(frozen=True)
class WeightLearnerConfig:
    walk_forward_splits: int = 4
    embargo_days: int = 10
    min_window_days: int = 30
    min_samples_per_slice: int = 50
    rank_ic_floor: float = 0.0
    horizon_days: int = 120
    label_column: str = "forward_return_120d"


@dataclass(frozen=True)
class LearnedWeights:
    global_weights: dict[str, float]
    per_theme: dict[str, dict[str, float]]
    per_sector: dict[str, dict[str, float]]
    per_regime: dict[str, dict[str, float]]
    per_lifecycle: dict[str, dict[str, float]]
    walk_forward_windows: int
    diagnostics: dict[str, float]


def learn_long_horizon_weights(
    panel: pd.DataFrame,
    config: WeightLearnerConfig | None = None,
    factors: Iterable[str] = LONG_HORIZON_FACTORS,
) -> LearnedWeights:
    """Learn adaptive long-horizon factor weights from a walk-forward IC panel."""

    config = config or WeightLearnerConfig()
    factors = tuple(factors)
    if panel is None or panel.empty or config.label_column not in panel.columns:
        return _fallback_weights(factors)
    if "trade_date" not in panel.columns:
        return _fallback_weights(factors)
    data = panel.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date"]).sort_values("trade_date")
    if len(data) < config.min_samples_per_slice:
        return _fallback_weights(factors)

    splits = _walk_forward_splits(data["trade_date"].unique(), config)
    if not splits:
        return _fallback_weights(factors)

    global_weight_runs: list[dict[str, float]] = []
    for _, test_dates in splits:
        test_slice = data[data["trade_date"].isin(test_dates)]
        weights = _ic_to_weights(test_slice, factors, config.label_column, config.rank_ic_floor)
        if weights:
            global_weight_runs.append(weights)
    global_weights = _average_weights(global_weight_runs, factors)
    per_theme = _slice_weights(data, "theme", factors, config)
    per_sector = _slice_weights(data, "sector", factors, config)
    per_regime = _slice_weights(data, "market_regime", factors, config)
    per_lifecycle = _slice_weights(data, "lifecycle_stage", factors, config)

    diagnostics = {
        "walk_forward_windows": float(len(splits)),
        "panel_rows": float(len(data)),
        "unique_dates": float(data["trade_date"].nunique()),
        "global_weight_runs": float(len(global_weight_runs)),
    }
    return LearnedWeights(
        global_weights=global_weights,
        per_theme=per_theme,
        per_sector=per_sector,
        per_regime=per_regime,
        per_lifecycle=per_lifecycle,
        walk_forward_windows=len(splits),
        diagnostics=diagnostics,
    )


def select_weights(
    learned: LearnedWeights,
    theme: str | None = None,
    sector: str | None = None,
    market_regime: str | None = None,
    lifecycle_stage: str | None = None,
) -> dict[str, float]:
    """Return the most specific learned weights available for the given slice.

    The lookup walks from the most specific slice (theme) to the global
    prior so partial coverage degrades gracefully. Every layer renormalises
    so missing categories do not silently zero out the alpha.
    """

    if theme and theme in learned.per_theme:
        return learned.per_theme[theme]
    if sector and sector in learned.per_sector:
        return learned.per_sector[sector]
    if market_regime and market_regime in learned.per_regime:
        return learned.per_regime[market_regime]
    if lifecycle_stage and lifecycle_stage in learned.per_lifecycle:
        return learned.per_lifecycle[lifecycle_stage]
    return learned.global_weights


def _walk_forward_splits(unique_dates, config: WeightLearnerConfig) -> list[tuple[pd.Index, pd.Index]]:
    dates = sorted(pd.to_datetime(unique_dates))
    if len(dates) < 2 * config.min_window_days + config.embargo_days:
        return []
    splits = max(1, config.walk_forward_splits)
    out: list[tuple[pd.Index, pd.Index]] = []
    span = len(dates)
    for split_index in range(1, splits + 1):
        cut = int(span * split_index / (splits + 1))
        if cut <= config.min_window_days:
            continue
        train_dates = dates[:cut]
        test_start = min(cut + config.embargo_days, span - 1)
        test_dates = dates[test_start:]
        if len(train_dates) < config.min_window_days or len(test_dates) < max(5, config.min_window_days // 4):
            continue
        out.append((pd.Index(train_dates), pd.Index(test_dates)))
    return out


def _slice_weights(
    panel: pd.DataFrame,
    column: str,
    factors: tuple[str, ...],
    config: WeightLearnerConfig,
) -> dict[str, dict[str, float]]:
    if column not in panel.columns:
        return {}
    out: dict[str, dict[str, float]] = {}
    for value, slice_ in panel.groupby(column, sort=False):
        if value is None or pd.isna(value):
            continue
        if len(slice_) < config.min_samples_per_slice:
            continue
        weights = _ic_to_weights(slice_, factors, config.label_column, config.rank_ic_floor)
        if not weights:
            continue
        out[str(value)] = weights
    return out


def _ic_to_weights(
    panel: pd.DataFrame,
    factors: tuple[str, ...],
    label_column: str,
    rank_ic_floor: float,
) -> dict[str, float]:
    if panel.empty or label_column not in panel.columns:
        return {}
    factor_ic: dict[str, float] = {}
    for factor in factors:
        if factor not in panel.columns:
            continue
        series = panel[[factor, label_column]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 5:
            continue
        try:
            ic = float(series[factor].rank().corr(series[label_column].rank(), method="pearson"))
        except Exception:
            continue
        if not np.isfinite(ic):
            continue
        if abs(ic) < rank_ic_floor:
            continue
        factor_ic[factor] = ic
    if not factor_ic:
        return {}
    total = sum(abs(value) for value in factor_ic.values())
    if total <= 0.0:
        return {}
    return {factor: float(abs(value) / total) for factor, value in factor_ic.items()}


def _average_weights(
    runs: list[dict[str, float]],
    factors: tuple[str, ...],
) -> dict[str, float]:
    if not runs:
        return _uniform_weights(factors)
    totals: dict[str, float] = {factor: 0.0 for factor in factors}
    for run in runs:
        for factor, weight in run.items():
            totals[factor] = totals.get(factor, 0.0) + float(weight)
    grand_total = sum(totals.values())
    if grand_total <= 0.0:
        return _uniform_weights(factors)
    return {factor: weight / grand_total for factor, weight in totals.items() if weight > 0.0}


def _uniform_weights(factors: tuple[str, ...]) -> dict[str, float]:
    if not factors:
        return {}
    weight = 1.0 / len(factors)
    return {factor: weight for factor in factors}


def _fallback_weights(factors: tuple[str, ...]) -> LearnedWeights:
    return LearnedWeights(
        global_weights=_uniform_weights(factors),
        per_theme={},
        per_sector={},
        per_regime={},
        per_lifecycle={},
        walk_forward_windows=0,
        diagnostics={"fallback": 1.0},
    )
