"""Lightweight full-pipeline PIT backtest scaffold.

This module replays a V7 research callback by ``as_of_date`` and keeps
PIT audit checks honest. It is not the production-grade A-share execution
simulator; use ``ashare_execution_simulator.py`` when target weights must
flow through ``OrderManager`` and ``VirtualBroker`` with retail execution
constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FullPipelineBacktestConfig:
    initial_capital: float = 1_000_000.0
    execution_lag_days: int = 1
    cost_bps: float = 8.0
    max_single_name_weight: float = 0.06
    revalidate_universe_every_n_days: int = 5


@dataclass
class FullPipelineBacktestResult:
    nav: pd.Series
    daily_returns: pd.Series
    target_weight_history: pd.DataFrame
    realized_weight_history: pd.DataFrame
    universe_size_history: pd.Series
    pit_audit: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_return(self) -> float:
        if self.nav.empty:
            return 0.0
        return float(self.nav.iloc[-1] / self.nav.iloc[0] - 1.0)

    @property
    def annualized_return(self) -> float:
        if self.daily_returns.empty:
            return 0.0
        mean = float(self.daily_returns.mean())
        return float(mean * 252.0)

    @property
    def sharpe(self) -> float:
        if self.daily_returns.empty:
            return 0.0
        std = float(self.daily_returns.std())
        if std <= 0:
            return 0.0
        return float(self.daily_returns.mean() / std * np.sqrt(252.0))

    @property
    def max_drawdown(self) -> float:
        if self.nav.empty:
            return 0.0
        running_max = self.nav.cummax()
        drawdown = self.nav / running_max - 1.0
        return float(drawdown.min())


def build_pit_evidence_slice(
    evidence_frame: pd.DataFrame,
    as_of_date: str,
    available_column: str = "available_at",
) -> pd.DataFrame:
    """Return only evidence rows visible at ``as_of_date``."""

    if evidence_frame is None or evidence_frame.empty:
        return evidence_frame
    if available_column not in evidence_frame.columns:
        return evidence_frame
    parsed = pd.to_datetime(evidence_frame[available_column], errors="coerce")
    visible_mask = parsed.notna() & (parsed <= pd.Timestamp(as_of_date))
    return evidence_frame.loc[visible_mask].reset_index(drop=True)


def run_full_pipeline_backtest(
    dates: Iterable[str],
    price_panel: pd.DataFrame,
    daily_step: Callable[[str], dict[str, Any]],
    config: FullPipelineBacktestConfig | None = None,
) -> FullPipelineBacktestResult:
    """Iterate over dates and replay the V7 research callback day by day."""

    config = config or FullPipelineBacktestConfig()
    dates = list(dates)
    if not dates or price_panel is None or price_panel.empty:
        return FullPipelineBacktestResult(
            nav=pd.Series(dtype=float),
            daily_returns=pd.Series(dtype=float),
            target_weight_history=pd.DataFrame(),
            realized_weight_history=pd.DataFrame(),
            universe_size_history=pd.Series(dtype=int),
        )
    panel = price_panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel = panel.dropna(subset=["trade_date"])
    panel = panel.set_index(["trade_date", "symbol"]).sort_index()
    close_wide = panel["close"].unstack("symbol").sort_index()

    target_weights_rows: list[pd.Series] = []
    realized_weights_rows: list[pd.Series] = []
    universe_sizes: dict[pd.Timestamp, int] = {}
    pit_audit: list[dict[str, Any]] = []
    current_weights: dict[str, float] = {}
    nav_values: list[tuple[pd.Timestamp, float]] = []
    nav = config.initial_capital

    sorted_dates = [pd.Timestamp(date) for date in dates]
    pending_target: dict[str, float] = {}
    for index, current_date in enumerate(sorted_dates):
        step = daily_step(current_date.strftime("%Y-%m-%d"))
        target = dict(step.get("target_weights", {}))
        for symbol, weight in list(target.items()):
            if weight > config.max_single_name_weight:
                target[symbol] = config.max_single_name_weight
        target_weights_rows.append(pd.Series(target, name=current_date))
        universe_sizes[current_date] = int(step.get("universe_size", len(target)))
        pit_audit.append(
            {
                "as_of_date": current_date.strftime("%Y-%m-%d"),
                "universe_size": int(universe_sizes[current_date]),
                "target_symbols": len(target),
                "audit": step.get("audit", {}),
            }
        )

        if config.execution_lag_days <= 0:
            current_weights = dict(target)
        else:
            current_weights = dict(pending_target) if index >= config.execution_lag_days else current_weights
            pending_target = dict(target)

        realized_weights_rows.append(pd.Series(current_weights, name=current_date))
        if current_date in close_wide.index and index > 0:
            prev_date = sorted_dates[index - 1]
            if prev_date in close_wide.index:
                ret = (close_wide.loc[current_date] / close_wide.loc[prev_date] - 1.0).fillna(0.0)
                weights_series = pd.Series(current_weights).reindex(close_wide.columns).fillna(0.0)
                day_return = float((weights_series * ret).sum())
                turnover = _turnover(realized_weights_rows, index)
                cost = (config.cost_bps / 10_000.0) * turnover
                nav = nav * (1.0 + day_return - cost)
        nav_values.append((current_date, nav))

    nav_series = pd.Series(dict(nav_values), name="nav").sort_index()
    daily_returns = nav_series.pct_change().dropna()
    target_history = pd.DataFrame(target_weights_rows).fillna(0.0)
    realized_history = pd.DataFrame(realized_weights_rows).fillna(0.0)
    universe_size_history = pd.Series(universe_sizes, name="universe_size").sort_index()
    return FullPipelineBacktestResult(
        nav=nav_series,
        daily_returns=daily_returns,
        target_weight_history=target_history,
        realized_weight_history=realized_history,
        universe_size_history=universe_size_history,
        pit_audit=pit_audit,
    )


def _turnover(realised_rows: list[pd.Series], index: int) -> float:
    if index <= 0:
        return 0.0
    today = realised_rows[index].fillna(0.0)
    yesterday = realised_rows[index - 1].fillna(0.0)
    union_index = today.index.union(yesterday.index)
    today = today.reindex(union_index).fillna(0.0)
    yesterday = yesterday.reindex(union_index).fillna(0.0)
    return float((today - yesterday).abs().sum())
