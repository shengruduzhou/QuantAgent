from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.execution.intraday_features import (
    FUNDFLOW_FEATURE_COLUMNS,
    merge_fundflow_features,
)


def _panel(n=30):
    t0 = pd.Timestamp("2026-06-17 09:31:00")
    times = [t0 + pd.Timedelta(minutes=i) for i in range(n)]
    px = 10 + np.cumsum(np.full(n, 0.01))
    return pd.DataFrame({
        "symbol": "000605.SZ", "trade_date": pd.Timestamp("2026-06-17"),
        "trade_time": times, "close": px, "volume": 1e5, "amount": px * 1e5,
    })


def test_fundflow_features_compute_and_are_finite():
    p = _panel()
    n = len(p)
    # cumulative net inflow rising (accumulation)
    ff = pd.DataFrame({
        "symbol": "000605.SZ", "trade_time": p["trade_time"],
        "main_net": np.cumsum(np.full(n, 2.0e5)), "super_net": np.cumsum(np.full(n, 1.0e5)),
        "large_net": np.cumsum(np.full(n, 0.5e5)), "mid_net": 0.0, "small_net": 0.0,
    })
    out = merge_fundflow_features(p, ff)
    for c in FUNDFLOW_FEATURE_COLUMNS:
        assert c in out.columns
    assert len(out) == n
    # rising cumulative main flow -> positive main-net intensity by mid-day
    assert out["ff_main_net_intensity"].dropna().iloc[-1] > 0
    assert out["ff_super_large_intensity"].dropna().iloc[-1] > 0


def test_fundflow_missing_yields_nan_columns():
    p = _panel()
    out = merge_fundflow_features(p, pd.DataFrame())
    for c in FUNDFLOW_FEATURE_COLUMNS:
        assert c in out.columns
        assert out[c].isna().all()
    assert len(out) == len(p)
