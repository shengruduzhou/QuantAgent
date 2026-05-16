"""Excess-return / rank / tradable label variants."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.data.v7_label_builder import build_forward_return_labels


def _panel(days: int = 6) -> pd.DataFrame:
    dates = pd.date_range("2026-05-01", periods=days, freq="B")
    rows = []
    for sidx, symbol in enumerate(("600519.SH", "000858.SZ")):
        for didx, date in enumerate(dates):
            close = 10.0 + sidx + didx * (0.05 + sidx * 0.10)
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000,
                    "amount": close * 1_000_000,
                    "is_suspended": didx == days - 2 and symbol == "000858.SZ",
                    "is_st": False,
                    "is_limit_up": False,
                    "is_limit_down": False,
                }
            )
    return pd.DataFrame(rows)


def test_excess_return_neutralises_cross_section_mean():
    panel = _panel()
    labels = build_forward_return_labels(panel, horizons=(1,))
    excess = labels.frame.groupby("trade_date")["forward_excess_return_1d"].mean()
    assert np.allclose(excess.dropna(), 0.0, atol=1e-9)


def test_rank_label_is_percentile_in_one():
    labels = build_forward_return_labels(_panel(), horizons=(1, 5))
    grouped = labels.frame.groupby("trade_date")["forward_rank_1d"].agg(["min", "max"])
    assert (grouped["max"] - grouped["min"] >= 0).all()
    assert (labels.frame["forward_rank_1d"].dropna() <= 1.0).all()
    assert (labels.frame["forward_rank_1d"].dropna() >= 0.0).all()


def test_tradable_label_masks_untradable_exit():
    labels = build_forward_return_labels(_panel(days=6), horizons=(1, 2))
    # 000858.SZ is suspended at didx=4. Rows whose horizon-1 exit is the
    # suspended day must be NaN on the tradable variant.
    affected = labels.frame[(labels.frame["symbol"] == "000858.SZ") & (labels.frame["trade_date"] == pd.Timestamp("2026-05-06"))]
    if not affected.empty:
        assert np.isnan(affected["forward_tradable_return_1d"].iloc[0])
    raw = labels.frame[(labels.frame["symbol"] == "000858.SZ") & (labels.frame["trade_date"] == pd.Timestamp("2026-05-06"))]
    if not raw.empty:
        assert not np.isnan(raw["forward_return_1d"].iloc[0])


def test_label_schema_lists_all_variants():
    schema = build_forward_return_labels(_panel(), horizons=(1, 5)).label_schema
    assert schema["excess_label_columns"] == ["forward_excess_return_1d", "forward_excess_return_5d"]
    assert schema["rank_label_columns"] == ["forward_rank_1d", "forward_rank_5d"]
    assert "tradable_label_columns" in schema
