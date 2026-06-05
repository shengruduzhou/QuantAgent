"""Unit tests for the index-hedge overlay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.portfolio.index_hedge import (
    apply_dynamic_index_hedge,
    apply_index_hedge,
    equal_weight_market_return,
    nav_metrics,
)


def _nav(returns: list[float], start: float = 1_000_000.0) -> pd.Series:
    idx = pd.bdate_range("2024-01-01", periods=len(returns) + 1)
    nav = [start]
    for r in returns:
        nav.append(nav[-1] * (1 + r))
    return pd.Series(nav, index=idx)


def test_hedge_ratio_zero_is_identity():
    nav = _nav([0.01, -0.02, 0.03])
    idx = nav.pct_change().fillna(0.0)
    out = apply_index_hedge(nav, idx, hedge_ratio=0.0)
    pd.testing.assert_series_equal(out, nav.sort_index().dropna().astype(float))


def test_full_hedge_against_self_removes_market_return():
    # if the book IS the index, a ratio-1 hedge leaves only the (negative)
    # cost drag — near-zero, strongly reduced volatility.
    nav = _nav([0.02, -0.03, 0.015, 0.01, -0.02])
    idx = nav.pct_change()
    hedged = apply_index_hedge(nav, idx, hedge_ratio=1.0, annual_cost_bps=0.0)
    hedged_ret = hedged.pct_change().dropna()
    assert float(hedged_ret.abs().max()) < 1e-9  # market fully removed


def test_partial_hedge_reduces_drawdown():
    # a book that falls with the market: hedging cuts the drawdown
    nav = _nav([-0.05, -0.04, -0.03, 0.02])
    idx = nav.pct_change()  # book moves with "index"
    raw_dd = nav_metrics(nav)["max_dd"]
    hedged = apply_index_hedge(nav, idx, hedge_ratio=1.0, annual_cost_bps=0.0)
    hedged_dd = nav_metrics(hedged)["max_dd"]
    assert hedged_dd < raw_dd


def test_dynamic_hedge_only_applies_on_risk_dates():
    nav = _nav([0.02, -0.05, -0.04, 0.02])
    idx = nav.pct_change()
    ratio = pd.Series([0.0, 0.0, 1.0, 1.0, 0.0], index=nav.index)
    hedged = apply_dynamic_index_hedge(nav, idx, ratio, annual_cost_bps=0.0)
    raw_dd = nav_metrics(nav)["max_dd"]
    hedged_dd = nav_metrics(hedged)["max_dd"]
    assert hedged_dd < raw_dd


def test_equal_weight_market_return_shape():
    rows = []
    for d in pd.bdate_range("2024-01-01", periods=5):
        for s, base in (("A.SH", 10.0), ("B.SZ", 20.0)):
            rows.append({"symbol": s, "trade_date": d, "close": base})
    panel = pd.DataFrame(rows)
    r = equal_weight_market_return(panel)
    assert len(r) == 5
    assert r.iloc[0] != r.iloc[0] or pd.isna(r.iloc[0])  # first is NaN (pct_change)
