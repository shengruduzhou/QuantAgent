"""Walk-forward sleeve allocator that learns regime-conditional weights.

The classical V7 sleeve allocator uses a static prior (long_fundamental,
medium_theme, short_event, sector_rotation, hedge, cash_buffer). That works as
a sanity floor but does not adapt to evidence from realised returns.

This module performs a deterministic walk-forward optimisation over a
history of daily sleeve returns. For every walk-forward window it computes
the sleeve combination that would have maximised the
risk-adjusted return (mean / std, capped by a drawdown penalty) under the
configured min/max constraints, then averages those weights across windows
to produce the recommended allocation for the next decision step.

The optimiser is intentionally simple and dependency-free:

* No external solver — we use a constrained grid search.
* No look-ahead — every window only sees its own training slice.
* No backtest leakage — the embargo window between train and test is
  configurable.

The output is a ``SleeveAllocationResult`` compatible with the existing
strategic-tactical allocator, so the rest of V7 does not need to be
re-wired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from quantagent.portfolio.sleeve import (
    DEFAULT_SLEEVE_CONFIGS,
    SleeveAllocationResult,
    SleeveConfig,
    SleeveTarget,
    SleeveType,
)


@dataclass(frozen=True)
class WalkForwardSleeveConfig:
    walk_forward_splits: int = 4
    embargo_days: int = 5
    min_window_days: int = 40
    grid_step: float = 0.05
    drawdown_penalty: float = 0.50
    cash_floor: float = 0.10
    rf_daily: float = 0.0


def allocate_sleeves_walk_forward(
    sleeve_returns: pd.DataFrame,
    regime_history: pd.Series | None = None,
    current_regime: str | None = None,
    sleeves: Sequence[SleeveConfig] = DEFAULT_SLEEVE_CONFIGS,
    config: WalkForwardSleeveConfig | None = None,
) -> SleeveAllocationResult:
    """Learn sleeve weights via walk-forward grid search on past returns.

    ``sleeve_returns`` is a daily return panel indexed by ``trade_date``
    with one column per sleeve type. ``regime_history`` is an optional
    Series aligned to the same index that tags the prevailing regime so
    the allocator can filter to days matching ``current_regime``.
    """

    config = config or WalkForwardSleeveConfig()
    if sleeve_returns is None or sleeve_returns.empty:
        return _fallback_allocation(sleeves, reason="no_sleeve_return_history")
    panel = _normalise_panel(sleeve_returns, sleeves)
    if panel.empty or len(panel) < config.min_window_days:
        return _fallback_allocation(sleeves, reason="insufficient_sleeve_history")
    panel = _filter_regime(panel, regime_history, current_regime)
    if panel.empty or len(panel) < config.min_window_days:
        return _fallback_allocation(sleeves, reason="insufficient_regime_history")

    windows = _walk_forward_splits(panel.index, config)
    if not windows:
        return _fallback_allocation(sleeves, reason="walk_forward_window_unavailable")
    grid = _weight_grid(sleeves, config.grid_step)
    cash_floor = max(config.cash_floor, _config_min_cash(sleeves))
    history: list[tuple[dict[SleeveType, float], float]] = []
    for train_idx, test_idx in windows:
        train = panel.loc[train_idx]
        test = panel.loc[test_idx]
        best = _best_weights_for_window(train, grid, sleeves, config, cash_floor)
        if best is None:
            continue
        score = _portfolio_score(test, best, config)
        history.append((best, score))
    if not history:
        return _fallback_allocation(sleeves, reason="no_valid_walk_forward_windows")
    weights = _average_weights(history, sleeves)
    diagnostics = _diagnostics(history, weights, panel, config)
    return _result(weights, sleeves, diagnostics, reason="walk_forward_learned")


def _walk_forward_splits(
    index: pd.Index,
    config: WalkForwardSleeveConfig,
) -> list[tuple[pd.Index, pd.Index]]:
    dates = sorted(pd.to_datetime(index).unique())
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
        if len(train_dates) < config.min_window_days or len(test_dates) < config.min_window_days // 2:
            continue
        out.append((pd.Index(train_dates), pd.Index(test_dates)))
    return out


def _best_weights_for_window(
    train: pd.DataFrame,
    grid: list[dict[SleeveType, float]],
    sleeves: Sequence[SleeveConfig],
    config: WalkForwardSleeveConfig,
    cash_floor: float,
) -> dict[SleeveType, float] | None:
    best_score = -np.inf
    best_weights: dict[SleeveType, float] | None = None
    for candidate in grid:
        if candidate.get(SleeveType.CASH_BUFFER, 0.0) < cash_floor:
            continue
        score = _portfolio_score(train, candidate, config)
        if score > best_score:
            best_score = score
            best_weights = candidate
    return best_weights


def _portfolio_score(
    panel: pd.DataFrame,
    weights: dict[SleeveType, float],
    config: WalkForwardSleeveConfig,
) -> float:
    if panel.empty:
        return -np.inf
    columns = [s.value for s in weights.keys() if s.value in panel.columns]
    if not columns:
        return -np.inf
    portfolio = pd.Series(0.0, index=panel.index)
    for sleeve_type, weight in weights.items():
        if sleeve_type.value in panel.columns:
            portfolio = portfolio + panel[sleeve_type.value].fillna(0.0) * weight
    mean = float(portfolio.mean())
    std = float(portfolio.std(ddof=0))
    drawdown = _max_drawdown(portfolio)
    if std <= 1e-12:
        return -np.inf
    sharpe = (mean - config.rf_daily) / std * np.sqrt(252)
    return float(sharpe - config.drawdown_penalty * drawdown)


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    rolling_max = nav.cummax()
    drawdown = nav / rolling_max - 1.0
    return float(abs(drawdown.min()))


def _normalise_panel(returns: pd.DataFrame, sleeves: Sequence[SleeveConfig]) -> pd.DataFrame:
    columns = [config.sleeve_type.value for config in sleeves]
    present = [column for column in columns if column in returns.columns]
    if not present:
        return pd.DataFrame()
    panel = returns[present].copy()
    panel.index = pd.to_datetime(panel.index, errors="coerce")
    panel = panel.loc[panel.index.notna()]
    return panel.sort_index()


def _filter_regime(
    panel: pd.DataFrame,
    regime_history: pd.Series | None,
    current_regime: str | None,
) -> pd.DataFrame:
    if regime_history is None or current_regime is None or panel.empty:
        return panel
    series = regime_history.copy()
    series.index = pd.to_datetime(series.index, errors="coerce")
    aligned = series.reindex(panel.index, method="ffill")
    mask = aligned.astype(str) == str(current_regime)
    filtered = panel.loc[mask]
    return filtered if not filtered.empty else panel


def _weight_grid(
    sleeves: Sequence[SleeveConfig],
    step: float,
) -> list[dict[SleeveType, float]]:
    if step <= 0.0 or step > 0.5:
        step = 0.05
    granular_sleeves = [config for config in sleeves]
    n = len(granular_sleeves)
    grid_levels = [round(value, 6) for value in np.arange(0.0, 1.0 + step, step)]
    weights: list[dict[SleeveType, float]] = []
    bounds = [(config.sleeve_type, config.min_weight, config.max_weight) for config in granular_sleeves]

    def backtrack(index: int, remaining: float, current: list[float]) -> None:
        if index == n - 1:
            sleeve, lo, hi = bounds[index]
            if lo <= remaining <= hi + 1e-9:
                final = current + [round(remaining, 6)]
                weights.append({bounds[i][0]: final[i] for i in range(n)})
            return
        sleeve, lo, hi = bounds[index]
        for level in grid_levels:
            if level < lo or level > hi + 1e-9:
                continue
            if level > remaining + 1e-9:
                continue
            backtrack(index + 1, round(remaining - level, 6), current + [level])

    backtrack(0, 1.0, [])
    if not weights:
        weights.append({config.sleeve_type: config.default_weight for config in sleeves})
    return weights


def _config_min_cash(sleeves: Sequence[SleeveConfig]) -> float:
    for sleeve in sleeves:
        if sleeve.sleeve_type == SleeveType.CASH_BUFFER:
            return sleeve.min_weight
    return 0.0


def _average_weights(
    history: list[tuple[dict[SleeveType, float], float]],
    sleeves: Sequence[SleeveConfig],
) -> dict[SleeveType, float]:
    keys = [config.sleeve_type for config in sleeves]
    total = {key: 0.0 for key in keys}
    weight_sum = 0.0
    for weights, score in history:
        weight = max(0.0, float(score))
        if weight == 0.0:
            weight = 1.0
        weight_sum += weight
        for key in keys:
            total[key] += weights.get(key, 0.0) * weight
    if weight_sum == 0.0:
        return {config.sleeve_type: config.default_weight for config in sleeves}
    return {key: total[key] / weight_sum for key in keys}


def _diagnostics(
    history: list[tuple[dict[SleeveType, float], float]],
    weights: dict[SleeveType, float],
    panel: pd.DataFrame,
    config: WalkForwardSleeveConfig,
) -> dict[str, float]:
    realised = _portfolio_score(panel, weights, config)
    return {
        "walk_forward_windows": float(len(history)),
        "expected_sharpe_per_window": float(np.mean([score for _, score in history])) if history else 0.0,
        "realised_score_full_history": float(realised) if np.isfinite(realised) else 0.0,
    }


def _result(
    weights: dict[SleeveType, float],
    sleeves: Sequence[SleeveConfig],
    diagnostics: dict[str, float],
    reason: str,
) -> SleeveAllocationResult:
    targets = tuple(
        SleeveTarget(
            sleeve_type=config.sleeve_type,
            target_weight=float(weights.get(config.sleeve_type, config.default_weight)),
            confidence=min(1.0, max(0.10, diagnostics.get("expected_sharpe_per_window", 0.5))),
            reason=reason,
        )
        for config in sleeves
    )
    cash_weight = float(weights.get(SleeveType.CASH_BUFFER, 0.0))
    return SleeveAllocationResult(
        targets=targets,
        total_nav=1.0,
        cash_weight=cash_weight,
        diagnostics=diagnostics,
    )


def _fallback_allocation(
    sleeves: Sequence[SleeveConfig],
    reason: str,
) -> SleeveAllocationResult:
    targets = tuple(
        SleeveTarget(
            sleeve_type=config.sleeve_type,
            target_weight=config.default_weight,
            confidence=0.20,
            reason=reason,
        )
        for config in sleeves
    )
    cash = next(
        (config.default_weight for config in sleeves if config.sleeve_type == SleeveType.CASH_BUFFER),
        0.30,
    )
    return SleeveAllocationResult(
        targets=targets,
        total_nav=1.0,
        cash_weight=cash,
        diagnostics={"walk_forward_windows": 0.0, "fallback": 1.0},
    )


def synthesise_sleeve_returns(
    market_panel: pd.DataFrame,
    sleeve_membership: dict[str, str],
) -> pd.DataFrame:
    """Build a per-sleeve daily return panel from raw market returns.

    ``sleeve_membership`` maps symbol -> sleeve name. The function takes
    daily returns from ``market_panel['close']``, averages by sleeve and
    returns a DataFrame indexed by ``trade_date``. It is intentionally
    a thin helper — callers can replace it with a richer attribution
    source (e.g. theme contribution) without changing the allocator.
    """

    if market_panel is None or market_panel.empty:
        return pd.DataFrame()
    data = market_panel.copy()
    if "trade_date" not in data.columns or "symbol" not in data.columns or "close" not in data.columns:
        return pd.DataFrame()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.sort_values(["symbol", "trade_date"])
    data["ret"] = data.groupby("symbol")["close"].pct_change()
    rows: list[dict[str, object]] = []
    for symbol, sleeve in sleeve_membership.items():
        slice_ = data[data["symbol"] == symbol]
        if slice_.empty:
            continue
        slice_ = slice_.dropna(subset=["ret"])
        for date, ret in zip(slice_["trade_date"], slice_["ret"]):
            rows.append({"trade_date": date, "sleeve": str(sleeve), "ret": float(ret)})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    pivot = frame.pivot_table(index="trade_date", columns="sleeve", values="ret", aggfunc="mean")
    return pivot.sort_index()
