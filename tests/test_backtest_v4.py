import numpy as np
import pandas as pd

from quantagent.backtest.engine import EventDrivenBacktester


def test_v4_backtester_reports_reject_reasons_and_costs():
    dates = pd.date_range("2026-01-02", periods=12, freq="B")
    rows = []
    for symbol in ["600519.SH", "688981.SH"]:
        close = np.linspace(10, 11, len(dates))
        for i, date in enumerate(dates):
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": close[i],
                    "high": close[i] * 1.01,
                    "low": close[i] * 0.99,
                    "close": close[i],
                    "volume": 0 if i == 5 and symbol == "600519.SH" else 1_000_000,
                    "amount": close[i] * 1_000_000,
                }
            )
    prices = pd.DataFrame(rows)
    weights = pd.DataFrame(0.0, index=dates, columns=["600519.SH", "688981.SH"])
    weights.iloc[0:8] = 0.2
    result = EventDrivenBacktester().run(weights, prices)
    assert "fill_ratio" in result.diagnostics
    assert "cost_attribution" in result.report
    assert isinstance(result.rejects, pd.DataFrame)
