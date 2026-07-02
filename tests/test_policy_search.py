"""Tests for the portfolio policy-search backtester (after-cost CAGR objective)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.portfolio.policy_search import (
    PolicyConfig,
    annualised_metrics,
    backtest_policy,
    prepare_working_frame,
    universe_benchmark,
)


def _work(n_days=40, seed=0, alt_alpha=False):
    """A=+1%/day (best), B=flat, C=-1%/day (worst); alpha ranks A>B>C unless alt."""
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rows = []
    drift = {"A": 0.01, "B": 0.0, "C": -0.01}
    base = {"A": 10.0, "B": 10.0, "C": 10.0}
    closes = {s: [base[s]] for s in base}
    for i in range(1, n_days):
        for s in base:
            closes[s].append(closes[s][-1] * (1 + drift[s]))
    for i, d in enumerate(dates):
        for s in base:
            if alt_alpha:   # alternate A/B as best each day -> forces turnover
                alpha = (1.0 if (s == "A") == (i % 2 == 0) else (0.5 if s == "B" else 0.0))
            else:
                alpha = {"A": 1.0, "B": 0.5, "C": 0.0}[s]
            rows.append({"symbol": s, "trade_date": d, "alpha_1d": alpha, "alpha_5d": alpha,
                         "alpha_20d": alpha, "close": closes[s][i],
                         "amount": 1e8, "is_st": False, "is_suspended": False, "is_limit_up": False})
    df = pd.DataFrame(rows)
    preds = df[["symbol", "trade_date", "alpha_1d", "alpha_5d", "alpha_20d"]]
    panel = df[["symbol", "trade_date", "close", "amount", "is_st", "is_suspended", "is_limit_up"]]
    return prepare_working_frame(preds, panel, sector=None)


def test_selection_picks_best_alpha_and_is_profitable():
    work = _work()
    cfg = PolicyConfig(horizon=1, top_k=1, rebalance_days=1, side="long_only",
                       cost_bps_per_turnover=0.0)
    res = backtest_policy(work, cfg)
    # Holding A (+1%/day) → strongly positive CAGR, positive total.
    assert res.metrics["total_return"] > 0.10
    assert res.metrics["cagr"] > 0.0
    # Stable selection (A always top) → ~zero turnover after the first buy.
    assert res.metrics["avg_one_way_turnover"] < 0.05


def test_worst_alpha_loses_and_cost_reduces_return():
    work = _work()
    free = backtest_policy(work, PolicyConfig(horizon=1, top_k=1, rebalance_days=1, cost_bps_per_turnover=0.0))
    costed = backtest_policy(work, PolicyConfig(horizon=1, top_k=1, rebalance_days=1, cost_bps_per_turnover=50.0))
    # Cost can only reduce (or equal) the after-cost return.
    assert costed.metrics["total_return"] <= free.metrics["total_return"] + 1e-9


def test_alternating_alpha_drives_turnover_and_cost():
    work = _work(alt_alpha=True)
    res = backtest_policy(work, PolicyConfig(horizon=1, top_k=1, rebalance_days=1, cost_bps_per_turnover=10.0))
    # Each day the top name flips A<->B → near-full one-way turnover each rebalance.
    assert res.metrics["avg_one_way_turnover"] > 0.5
    assert res.metrics["annual_turnover"] > res.metrics["avg_one_way_turnover"]


def test_long_short_spreads_best_minus_worst():
    work = _work()
    res = backtest_policy(work, PolicyConfig(horizon=1, top_k=1, rebalance_days=1,
                                             side="long_short", cost_bps_per_turnover=0.0))
    # long A (+1%) − short C (−1%) ≈ +2%/day gross → strongly positive.
    assert res.metrics["total_return"] > 0.20


def test_universe_benchmark_is_equal_weight_mean():
    work = _work()
    bm = universe_benchmark(work)
    # A+B+C eqw daily mean ≈ (1% + 0% − 1%)/3 ≈ 0 → ~flat basket.
    assert abs(bm["cagr"]) < 0.05
    assert bm["n_days"] > 10


def test_annualised_metrics_basic():
    daily = pd.Series([0.01] * 252)
    m = annualised_metrics(daily)
    assert abs(m["cagr"] - ((1.01 ** 252) - 1)) < 1e-6
    assert m["max_drawdown"] == 0.0
    assert m["win_rate_daily"] == 1.0
