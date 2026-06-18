from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.execution.intraday_features import (
    CAUSAL_INTRADAY_FEATURE_COLUMNS,
    LEVEL2_FEATURE_COLUMNS,
    build_causal_intraday_feature_frame,
)


def _panel(closes):
    n = len(closes)
    return pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * n,
            "trade_date": ["2026-06-01"] * n,
            "trade_time": pd.date_range("2026-06-01 09:30:00", periods=n, freq="min"),
            "open": closes,
            "high": [c * 1.002 for c in closes],
            "low": [c * 0.998 for c in closes],
            "close": closes,
            "volume": np.linspace(1000, 5000, n),
            "amount": [c * v for c, v in zip(closes, np.linspace(1000, 5000, n))],
            "pre_close": [10.0] * n,
            "index_return": np.linspace(0, 0.001, n),
            "industry_return": np.linspace(0, 0.002, n),
        }
    )


def test_causal_intraday_features_include_required_columns_without_level2_fakes():
    out = build_causal_intraday_feature_frame(_panel(np.linspace(10.0, 10.5, 30)))
    for col in CAUSAL_INTRADAY_FEATURE_COLUMNS:
        assert col in out.columns
    for col in LEVEL2_FEATURE_COLUMNS:
        assert col not in out.columns
    assert out["price_vs_open"].iloc[-1] > 0
    assert out["estimated_spread_bps"].notna().all()


def test_causal_features_do_not_change_when_future_bars_change():
    base = _panel(np.linspace(10.0, 10.5, 30))
    mutated = base.copy()
    mutated.loc[10:, "close"] = np.linspace(20.0, 25.0, len(mutated.loc[10:]))
    f1 = build_causal_intraday_feature_frame(base)
    f2 = build_causal_intraday_feature_frame(mutated)
    cols = ["price_vs_vwap_z", "intraday_percentile_since_open", "rolling_return_5m", "volume_zscore_5m"]
    pd.testing.assert_frame_equal(f1.loc[:9, cols], f2.loc[:9, cols], check_dtype=False)


def test_causal_vwap_falls_back_when_tickflow_amount_volume_units_mismatch():
    base = _panel(np.linspace(10.0, 10.5, 30))
    mismatched = base.copy()
    base["industry_vwap_dev"] = 0.0
    mismatched["industry_vwap_dev"] = 0.0
    mismatched["amount"] = mismatched["close"] * mismatched["volume"] * 100.0

    expected = build_causal_intraday_feature_frame(base)
    out = build_causal_intraday_feature_frame(mismatched)

    pd.testing.assert_series_equal(
        expected["stock_vwap_dev_minus_industry"],
        out["stock_vwap_dev_minus_industry"],
        check_names=False,
    )
    assert out["stock_vwap_dev_minus_industry"].iloc[-1] > -0.10


def test_level2_columns_are_used_only_when_present_and_requested():
    panel = _panel(np.linspace(10.0, 10.5, 10))
    panel["bid_ask_spread"] = 0.01
    out_without = build_causal_intraday_feature_frame(panel, include_level2=False)
    out_with = build_causal_intraday_feature_frame(panel, include_level2=True)
    assert "bid_ask_spread" not in out_without.columns
    assert "bid_ask_spread" in out_with.columns
