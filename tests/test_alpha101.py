import numpy as np
import pandas as pd

from quantagent.factors.alpha101 import alpha001, compute_alpha101
from quantagent.factors.registry import default_registry


def _ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-01-01", periods=80)
    rows = []
    for j, symbol in enumerate(["A", "B", "C", "D", "E"]):
        close = 20 + np.cumsum(rng.normal(0.02 + j * 0.002, 0.2, len(dates)))
        volume = 1_000_000 + rng.normal(0, 10_000, len(dates)).cumsum() + j * 20_000
        for i, date in enumerate(dates):
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": close[i] * 0.99,
                    "high": close[i] * 1.01,
                    "low": close[i] * 0.98,
                    "close": close[i],
                    "volume": max(volume[i], 1000),
                    "amount": max(volume[i], 1000) * close[i],
                }
            )
    return pd.DataFrame(rows)


def test_alpha101_registers_first_thirty_factors():
    names = default_registry.names("alpha101")
    assert "alpha001" in names
    assert "alpha030" in names
    assert "alpha101" in names
    assert len([name for name in names if name.startswith("alpha")]) >= 101


def test_compute_alpha101_outputs_long_form():
    frame = _ohlcv()
    factors = compute_alpha101(frame)
    assert {"trade_date", "symbol", "factor_name", "factor_value"}.issubset(factors.columns)
    assert factors["factor_name"].nunique() == 101


def test_alpha001_has_no_lookahead_from_future_change():
    frame = _ohlcv()
    modified = frame.copy()
    last_date = modified["trade_date"].max()
    modified.loc[modified["trade_date"] == last_date, "close"] *= 10.0
    base = alpha001(frame)
    changed = alpha001(modified)
    before_last = base["trade_date"] < last_date
    pd.testing.assert_series_equal(
        base.loc[before_last, "factor_value"].reset_index(drop=True),
        changed.loc[before_last, "factor_value"].reset_index(drop=True),
        check_names=False,
    )
