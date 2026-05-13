from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantagent.backtest.engine import BacktestConfig, BacktestResult, EventDrivenBacktester


@dataclass
class ThemeBacktestResult:
    base_result: BacktestResult
    theme_contribution: dict[str, float] = field(default_factory=dict)
    practical_execution_feasibility: float = 1.0


class EventDrivenThemeBacktester:
    """Theme-aware wrapper around the A-share T+1 event-driven backtester."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.backtester = EventDrivenBacktester(config=config)

    def run(self, target_weights: pd.DataFrame, prices: pd.DataFrame, theme_membership: pd.DataFrame) -> ThemeBacktestResult:
        result = self.backtester.run(target_weights, prices)
        contribution = _theme_contribution(result.holdings, prices, theme_membership)
        attempts = float(len(result.trades) + len(result.rejects))
        feasibility = 1.0 if attempts == 0 else float(len(result.trades) / attempts)
        return ThemeBacktestResult(
            base_result=result,
            theme_contribution=contribution,
            practical_execution_feasibility=feasibility,
        )


def _theme_contribution(holdings: pd.DataFrame, prices: pd.DataFrame, membership: pd.DataFrame) -> dict[str, float]:
    if holdings.empty or prices.empty or membership.empty:
        return {}
    long_prices = prices[["trade_date", "symbol", "close"]].copy()
    long_prices["trade_date"] = pd.to_datetime(long_prices["trade_date"])
    long_prices["return"] = long_prices.sort_values(["symbol", "trade_date"]).groupby("symbol")["close"].pct_change().fillna(0.0)
    weights = holdings.reset_index().melt(id_vars="trade_date", var_name="symbol", value_name="weight")
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    merged = weights.merge(long_prices[["trade_date", "symbol", "return"]], on=["trade_date", "symbol"], how="left")
    merged = merged.merge(membership[["symbol", "theme"]].drop_duplicates(), on="symbol", how="left")
    merged["theme"] = merged["theme"].fillna("unmapped")
    merged["contribution"] = merged["weight"].fillna(0.0) * merged["return"].fillna(0.0)
    return {str(theme): float(value) for theme, value in merged.groupby("theme")["contribution"].sum().items()}
