import numpy as np
import pandas as pd

from quantagent.backtest.engine import EventDrivenBacktester
from quantagent.factors.alpha101 import alpha029
from quantagent.strategy.weight_adapter import apply_lot_liquidity_constraints
from quantagent.domain.schemas import TargetWeight


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    rows = []
    for symbol in ["A", "B", "C"]:
        close = 20 + np.cumsum(rng.normal(0.02, 0.1, len(dates)))
        for i, date in enumerate(dates):
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": close[i] * 0.99,
                    "high": close[i] * 1.01,
                    "low": close[i] * 0.98,
                    "close": close[i],
                    "volume": 1_000_000,
                    "amount": close[i] * 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_v3_factor_weights_run_through_backtester_with_diagnostics():
    prices = _prices()
    factors = alpha029(prices)
    latest = factors.sort_values("trade_date").groupby("symbol").tail(1)
    targets = [
        TargetWeight(row["symbol"], 0.05 if row["factor_value"] >= latest["factor_value"].median() else 0.0, 5, 0.8, "test")
        for _, row in latest.iterrows()
    ]
    last_prices = prices.sort_values("trade_date").groupby("symbol").tail(1).set_index("symbol")["close"]
    adjusted = apply_lot_liquidity_constraints(targets, 1_000_000, last_prices)
    dates = sorted(prices["trade_date"].unique())
    weights = pd.DataFrame(0.0, index=dates, columns=last_prices.index)
    for target in adjusted:
        weights.loc[dates[-5]:, target.symbol] = target.target_weight
    weights.attrs["sleeve_diagnostics"] = {"sleeve_count": 5}
    weights.attrs["stop_loss_diagnostics"] = {"stop_event_count": 1, "blocked_exit_count": 0}
    result = EventDrivenBacktester().run(weights, prices)
    assert result.nav_curve.notna().all()
    assert result.diagnostics["sleeve_count"] == 5.0
    assert result.diagnostics["stop_event_count"] == 1.0

