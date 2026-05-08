import numpy as np
import pandas as pd

from quantagent.backtest.engine import BacktestConfig, EventDrivenBacktester


def _build_prices(symbols, dates, drift=0.001):
    rng = np.random.default_rng(0)
    rows = []
    for s in symbols:
        log_close = np.cumsum(rng.normal(drift, 0.01, len(dates)))
        close = 50 * np.exp(log_close)
        for i, d in enumerate(dates):
            rows.append(
                {
                    "trade_date": d,
                    "symbol": s,
                    "open": close[i] * (1 + rng.normal(0, 0.001)),
                    "high": close[i] * (1 + abs(rng.normal(0, 0.002))),
                    "low": close[i] * (1 - abs(rng.normal(0, 0.002))),
                    "close": close[i],
                    "volume": 1_000_000.0,
                    "amount": close[i] * 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_engine_runs_t_plus_one_with_costs():
    dates = pd.date_range("2026-01-02", periods=30, freq="B")
    symbols = ["600519.SH", "300750.SZ", "601318.SH"]
    prices = _build_prices(symbols, dates)
    weights = pd.DataFrame(0.0, index=dates, columns=symbols)
    weights.iloc[0] = 0.3
    engine = EventDrivenBacktester(BacktestConfig(initial_nav=1_000_000.0))
    result = engine.run(weights, prices)
    assert result.nav_curve.notna().all()
    assert "trade_count" in result.diagnostics
    assert result.trades.shape[0] > 0
