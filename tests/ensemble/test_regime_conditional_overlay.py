"""Tests for the regime-conditional overlay."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.ensemble.regime_conditional_overlay import (
    RegimeOverlayConfig,
    compute_regime_labels,
    conditional_score,
    evidence_composite,
    fit_regime_lambdas,
    portfolio_daily_return,
)


def test_regime_labels_detect_bull_and_bear():
    dates = pd.bdate_range("2024-01-01", periods=120)
    rising = pd.Series(np.linspace(100, 160, 120), index=dates)
    falling = pd.Series(np.linspace(160, 90, 120), index=dates)
    bull = compute_regime_labels(rising)
    bear = compute_regime_labels(falling)
    assert (bull == "bull").any()
    assert (bear == "bear").any()


def _toy_df() -> pd.DataFrame:
    rows = []
    for d in pd.bdate_range("2024-01-01", periods=2):
        for i, sym in enumerate(["A", "B", "C", "D"]):
            rows.append({
                "trade_date": d, "symbol": sym,
                "alpha_score": 4 - i,                 # A>B>C>D
                "fundamental_quality_score": i,        # D>C>B>A (opposes alpha)
                "core_policy_score": 0.0, "core_sentiment_score": 0.0,
                "sector_resonance_score": 0.0, "dip_buy_flow_score": 0.0,
                "old_dealer_risk_score": 0.0,
                "fwd_ret": 0.01 * (4 - i),
            })
    return pd.DataFrame(rows)


def test_lambda_zero_recovers_pure_factor():
    df = _toy_df()
    cfg = RegimeOverlayConfig(top_k=2)
    regimes = pd.Series({d: "bull" for d in df["trade_date"].unique()})
    base = portfolio_daily_return(df, df.groupby("trade_date")["alpha_score"].transform(
        lambda g: (g - g.mean()) / (g.std() or 1.0)), cfg)
    score0 = conditional_score(df, regimes, {"bull": 0.0}, cfg)
    cond = portfolio_daily_return(df, score0, cfg)
    # lambda=0 => identical top-K returns to pure factor
    assert np.allclose(base.values, cond.values)


def test_evidence_composite_is_finite_and_nonzero():
    df = _toy_df()
    comp = evidence_composite(df, RegimeOverlayConfig())
    assert np.isfinite(comp).all()
    assert comp.abs().sum() > 0  # fundamental_quality_score varies => nonzero


def test_fit_returns_lambda_per_bucket():
    df = _toy_df()
    cfg = RegimeOverlayConfig(top_k=2)
    regimes = pd.Series({d: "sideways" for d in df["trade_date"].unique()})
    bench = portfolio_daily_return(df, df["alpha_score"], cfg) * 0.0  # zero benchmark
    lambdas = fit_regime_lambdas(df, regimes, bench, cfg)
    assert set(lambdas) == {"bull", "sideways", "bear"}
    assert all(isinstance(v, float) for v in lambdas.values())
