import numpy as np
import pandas as pd

from quantagent.factors.operators import delay, rank, ts_mean, zscore
from quantagent.factors.preprocessing import neutralize_by_date


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=8)
    rows = []
    for symbol in ["A", "B", "C", "D"]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "close": 10 + i + ord(symbol) % 3,
                    "factor": i + (0 if symbol in {"A", "B"} else 10),
                    "industry": "x" if symbol in {"A", "B"} else "y",
                }
            )
    return pd.DataFrame(rows)


def test_delay_uses_only_past_values():
    frame = _panel()
    values = delay(frame, "close", 1)
    a = frame[frame["symbol"] == "A"].copy()
    assert np.isnan(values.loc[a.index[0]])
    assert values.loc[a.index[3]] == frame.loc[a.index[2], "close"]


def test_time_series_mean_is_trailing():
    frame = _panel()
    values = ts_mean(frame, "close", 3)
    a = frame[frame["symbol"] == "A"].copy()
    expected = frame.loc[a.index[:3], "close"].mean()
    assert values.loc[a.index[2]] == expected


def test_cross_sectional_rank_and_zscore_by_date():
    frame = _panel()
    ranked = rank(frame, "factor")
    scored = zscore(frame, "factor")
    first_date = frame["trade_date"].iloc[0]
    mask = frame["trade_date"] == first_date
    assert ranked.loc[mask].between(0, 1).all()
    assert abs(scored.loc[mask].mean()) < 1e-12


def test_neutralization_removes_industry_mean_exposure():
    frame = _panel()
    result = neutralize_by_date(frame, "factor", industry_column="industry", output_column="factor_neutral")
    means = result.groupby(["trade_date", "industry"])["factor_neutral"].mean().dropna()
    assert means.abs().max() < 1e-10

