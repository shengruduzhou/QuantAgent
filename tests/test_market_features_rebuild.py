"""Stage 2.1 tests: rebuilt build_market_features derives flags from OHLCV.

These tests pin the new behavior so future edits cannot regress:

* is_suspended is True only on days with both volume==0 / NaN AND
  amount==0 / NaN AND the date had >=100 trading symbols (matches
  the universe-filter contract).
* is_limit_up is True iff close/prev_close - 1 >= limit_up_pct
  (default 9.9%). First day per symbol is not flagged.
* is_limit_down mirrors that on the downside.
* is_st is taken from the caller-provided st_flags table; without
  it the column defaults to False (Stage 2.2 will wire akshare).
* available_at is the next observation date per symbol — preserves
  the existing PIT contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.data.v7_dataset_builder import build_market_features


def _panel_with_three_symbols() -> pd.DataFrame:
    """A.SZ trades normally, B.SZ has a suspension day, C.SZ hits +10% limit-up."""
    dates = pd.bdate_range("2024-01-02", periods=8)
    rows = []
    a_close = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7]
    for i, d in enumerate(dates):
        rows.append({"trade_date": d, "symbol": "A.SZ", "open": a_close[i] * 0.99, "high": a_close[i] * 1.01, "low": a_close[i] * 0.98, "close": a_close[i], "volume": 1_000_000, "amount": 1e7})
    b_close = [20.0, 20.3, 20.3, 20.3, 20.5, 20.6, 20.7, 20.8]
    b_volume = [500_000, 500_000, 0, 500_000, 500_000, 500_000, 500_000, 500_000]
    b_amount = [1e7, 1e7, 0, 1e7, 1e7, 1e7, 1e7, 1e7]
    for i, d in enumerate(dates):
        rows.append({"trade_date": d, "symbol": "B.SZ", "open": b_close[i] * 0.99, "high": b_close[i] * 1.01, "low": b_close[i] * 0.98, "close": b_close[i], "volume": b_volume[i], "amount": b_amount[i]})
    c_close = [10.0, 10.0, 10.0, 11.0, 11.0, 11.0, 11.0, 11.0]  # +10% on day 4
    for i, d in enumerate(dates):
        rows.append({"trade_date": d, "symbol": "C.SZ", "open": c_close[i] * 0.99, "high": c_close[i] * 1.01, "low": c_close[i] * 0.98, "close": c_close[i], "volume": 800_000, "amount": 8e6})
    # Pad with 200 quiet symbols so derive_market_flags treats these as active trading days
    for sid in range(200):
        for i, d in enumerate(dates):
            rows.append({"trade_date": d, "symbol": f"X{sid:03d}.SZ", "open": 4.95, "high": 5.05, "low": 4.90, "close": 5.0 + sid * 0.001, "volume": 100_000, "amount": 1e6})
    return pd.DataFrame(rows)


def test_is_suspended_flagged_only_on_suspension_day():
    panel = _panel_with_three_symbols()
    feats = build_market_features(panel)
    b = feats[feats["symbol"] == "B.SZ"].sort_values("trade_date").reset_index(drop=True)
    susp_dates = b.loc[b["is_suspended"], "trade_date"]
    assert len(susp_dates) == 1
    assert susp_dates.iloc[0] == pd.Timestamp("2024-01-04")
    # Other symbols on the same day are NOT suspended
    others_that_day = feats[(feats["trade_date"] == "2024-01-04") & (feats["symbol"] != "B.SZ")]
    assert not others_that_day["is_suspended"].any()


def test_is_limit_up_flagged_only_on_jump_day():
    panel = _panel_with_three_symbols()
    feats = build_market_features(panel, limit_up_pct=0.099)
    c = feats[feats["symbol"] == "C.SZ"].sort_values("trade_date").reset_index(drop=True)
    lu_dates = c.loc[c["is_limit_up"], "trade_date"]
    assert len(lu_dates) == 1
    assert lu_dates.iloc[0] == pd.Timestamp("2024-01-05")  # the +10% day
    # No false positives elsewhere
    a = feats[feats["symbol"] == "A.SZ"]
    assert not a["is_limit_up"].any()


def test_is_limit_down_flagged_on_drop():
    """Verify the down-side path explicitly."""
    dates = pd.bdate_range("2024-01-02", periods=4)
    rows = [
        {"trade_date": dates[0], "symbol": "D.SZ", "open": 9.9, "high": 10.1, "low": 9.8, "close": 10.0, "volume": 1e6, "amount": 1e7},
        {"trade_date": dates[1], "symbol": "D.SZ", "open": 9.0, "high": 9.1, "low": 8.95, "close": 9.0, "volume": 1e6, "amount": 9e6},  # -10%
        {"trade_date": dates[2], "symbol": "D.SZ", "open": 9.0, "high": 9.1, "low": 8.95, "close": 9.0, "volume": 1e6, "amount": 9e6},
        {"trade_date": dates[3], "symbol": "D.SZ", "open": 9.0, "high": 9.1, "low": 8.95, "close": 9.0, "volume": 1e6, "amount": 9e6},
    ]
    for sid in range(200):
        for d in dates:
            rows.append({"trade_date": d, "symbol": f"Y{sid:03d}.SZ", "open": 5.0, "high": 5.1, "low": 4.9, "close": 5.0, "volume": 1e5, "amount": 1e6})
    panel = pd.DataFrame(rows)
    feats = build_market_features(panel, limit_down_pct=-0.099)
    d = feats[feats["symbol"] == "D.SZ"].sort_values("trade_date").reset_index(drop=True)
    ld_dates = d.loc[d["is_limit_down"], "trade_date"]
    assert len(ld_dates) == 1
    assert ld_dates.iloc[0] == dates[1]


def test_is_st_taken_from_caller_table_when_present():
    panel = _panel_with_three_symbols()
    st_flags = pd.DataFrame({
        "trade_date": pd.bdate_range("2024-01-02", periods=8).tolist() * 2,
        "symbol": ["A.SZ"] * 8 + ["B.SZ"] * 8,
        "is_st": [False] * 8 + [True] * 8,
    })
    feats = build_market_features(panel, st_flags=st_flags)
    # B.SZ is ST on every day, A.SZ never
    assert feats[feats["symbol"] == "B.SZ"]["is_st"].all()
    assert not feats[feats["symbol"] == "A.SZ"]["is_st"].any()
    # C.SZ not in st_flags → defaults to False
    assert not feats[feats["symbol"] == "C.SZ"]["is_st"].any()


def test_is_st_defaults_false_without_caller_table():
    panel = _panel_with_three_symbols()
    feats = build_market_features(panel)
    assert not feats["is_st"].any()


def test_amount_mean_20d_uses_only_past_data():
    """Critical PIT check: amount_mean_20d at row i must equal the
    rolling mean over rows [i-19, i] (inclusive of i), NOT use any
    row beyond i.
    """
    dates = pd.bdate_range("2024-01-02", periods=30)
    rows = [{"trade_date": d, "symbol": "X.SZ", "open": 9.9, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 1e6, "amount": float(i + 1) * 1e6}
            for i, d in enumerate(dates)]
    # Pad
    for sid in range(150):
        for d in dates:
            rows.append({"trade_date": d, "symbol": f"P{sid:03d}.SZ", "open": 5.0, "high": 5.05, "low": 4.95, "close": 5.0, "volume": 1e5, "amount": 1e6})
    panel = pd.DataFrame(rows)
    feats = build_market_features(panel)
    x = feats[feats["symbol"] == "X.SZ"].sort_values("trade_date").reset_index(drop=True)
    # At row 19 the trailing-20 mean of amounts [1,2,...,20] * 1e6 = 10.5e6
    expected = float(np.mean([(i + 1) * 1e6 for i in range(20)]))
    assert x.loc[19, "amount_mean_20d"] == pytest.approx(expected, rel=1e-9)
    # At row 24 the trailing-20 should slide: [5..24]
    expected_24 = float(np.mean([(i + 1) * 1e6 for i in range(5, 25)]))
    assert x.loc[24, "amount_mean_20d"] == pytest.approx(expected_24, rel=1e-9)


def test_no_future_function_in_derived_flags():
    """Confirm that altering ROW j > i's data does not change features at row i.

    Build two panels identical except for the LAST row of A.SZ; compute features
    for both; assert all rows EXCEPT the last are identical for A.SZ.
    """
    panel_a = _panel_with_three_symbols()
    panel_b = panel_a.copy()
    # Replace the last A.SZ row close with a huge jump (would create a limit-up
    # IF the function peeked at future data when computing earlier rows)
    last_a_mask = (panel_b["symbol"] == "A.SZ") & (panel_b["trade_date"] == panel_b[panel_b["symbol"] == "A.SZ"]["trade_date"].max())
    panel_b.loc[last_a_mask, "close"] = 999.0

    feats_a = build_market_features(panel_a).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    feats_b = build_market_features(panel_b).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    # All rows for A.SZ except the last must match
    a_a = feats_a[feats_a["symbol"] == "A.SZ"].reset_index(drop=True)
    a_b = feats_b[feats_b["symbol"] == "A.SZ"].reset_index(drop=True)
    flag_cols = ["is_suspended", "is_limit_up", "is_limit_down", "is_st"]
    for col in flag_cols:
        # Match all rows EXCEPT the perturbed last one
        pd.testing.assert_series_equal(a_a[col].iloc[:-1], a_b[col].iloc[:-1], check_names=False)


def test_available_at_preserves_existing_pit_contract():
    """available_at at row i == trade_date at row i+1 (for same symbol),
    falling back to trade_date + 1 calendar day at the last row.
    """
    panel = _panel_with_three_symbols()
    feats = build_market_features(panel)
    a = feats[feats["symbol"] == "A.SZ"].sort_values("trade_date").reset_index(drop=True)
    for i in range(len(a) - 1):
        assert pd.Timestamp(a.loc[i, "available_at"]) == pd.Timestamp(a.loc[i + 1, "trade_date"])
    # Last row falls back to trade_date + 1 day
    last_idx = len(a) - 1
    assert pd.Timestamp(a.loc[last_idx, "available_at"]) == pd.Timestamp(a.loc[last_idx, "trade_date"]) + pd.Timedelta(days=1)
