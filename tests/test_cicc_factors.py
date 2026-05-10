import numpy as np
import pandas as pd

from quantagent.factors.cicc_high_freq import INTRADAY_ONLY_FACTORS, compute_cicc_high_freq_factors


def _daily() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=30)
    rows = []
    for j, symbol in enumerate(["A", "B", "C"]):
        for i, date in enumerate(dates):
            close = 10 + i * 0.1 + j
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000 + i * 1000,
                    "amount": close * (1_000_000 + i * 1000),
                }
            )
    return pd.DataFrame(rows)


def test_daily_cicc_factors_mark_intraday_unavailable():
    result = compute_cicc_high_freq_factors(_daily())
    assert set(INTRADAY_ONLY_FACTORS).issubset(set(result.unavailable))
    assert "daily_amihud" in set(result.factors["factor_name"])


def test_intraday_cicc_factors_include_intraday_names():
    daily = _daily().head(6)
    rows = []
    for _, row in daily.iterrows():
        for minute in range(10):
            rows.append(
                {
                    **row.to_dict(),
                    "datetime": pd.Timestamp(row["trade_date"]) + pd.Timedelta(minutes=minute),
                    "close": row["open"] * (1 + 0.001 * minute),
                    "amount": row["amount"] / 10,
                    "volume": row["volume"] / 10,
                }
            )
    result = compute_cicc_high_freq_factors(pd.DataFrame(rows), window=3)
    assert result.unavailable == ()
    assert "crowding_fft_ratio" in set(result.factors["factor_name"])

