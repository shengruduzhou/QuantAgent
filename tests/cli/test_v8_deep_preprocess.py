"""Tests for the v8_deep per-date preprocessing helpers (anti-overfit)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.cli.v8_deep import (
    _candidate_feature_names,
    _cross_sectional_normalize,
    _filter_by_regime_dates,
    _normalize_label_per_date,
)


def _toy_frame():
    # 2 dates × 4 symbols; feature "f1" has wildly different scale per symbol
    # (mimics 茅台 1300 vs 微盘 3) so global z-score would be wrong.
    rows = []
    for d in pd.to_datetime(["2024-01-02", "2024-01-03"]):
        for i, sym in enumerate(["A", "B", "C", "D"]):
            rows.append({"trade_date": d, "symbol": sym,
                         "f1": (i + 1) * 100.0, "f2": float(i)})
    return pd.DataFrame(rows)


def test_cross_sectional_rank_is_per_date_and_bounded():
    df = _toy_frame()
    out = _cross_sectional_normalize(df, ["f1", "f2"], method="rank")
    # rank-pct centred → within [-0.5, 0.5]
    assert out["f1"].between(-0.5, 0.5).all()
    # within each date the ordering of f1 (monotone in symbol index) is preserved
    for _, g in out.groupby("trade_date"):
        assert g.sort_values("f1")["symbol"].tolist() == ["A", "B", "C", "D"]
    # highest value per date maps to +0.5, lowest to -0.5 (4 names → ranks .25/.5/.75/1)
    day = out[out["trade_date"] == out["trade_date"].min()].sort_values("symbol")
    assert day["f1"].iloc[-1] == 0.5


def test_cross_sectional_zscore_per_date_mean_zero():
    df = _toy_frame()
    out = _cross_sectional_normalize(df, ["f1"], method="zscore")
    for _, g in out.groupby("trade_date"):
        assert abs(g["f1"].mean()) < 1e-9


def test_cross_sectional_is_leak_free():
    """Normalising date t must not depend on any other date's rows."""
    df = _toy_frame()
    full = _cross_sectional_normalize(df, ["f1"], method="rank")
    # Recompute using only the first date — values must be identical
    d0 = df["trade_date"].min()
    only0 = _cross_sectional_normalize(df[df["trade_date"] == d0].copy(), ["f1"], method="rank")
    merged = full[full["trade_date"] == d0].sort_values("symbol")["f1"].to_numpy()
    isolated = only0.sort_values("symbol")["f1"].to_numpy()
    assert np.allclose(merged, isolated)


def test_label_winsor_zscore_per_date():
    # one date with an extreme micro-cap outlier (+50%) that should be clipped
    d = pd.Timestamp("2024-01-02")
    df = pd.DataFrame({
        "trade_date": [d] * 6,
        "symbol": list("ABCDEF"),
        "forward_return_20d": [0.01, 0.02, -0.01, 0.0, 0.015, 0.50],
    })
    out = _normalize_label_per_date(df, "forward_return_20d", winsor=0.10)
    # post z-score, per-date mean ~ 0
    assert abs(out["forward_return_20d"].mean()) < 1e-9
    # the +0.50 outlier must no longer be the dominating extreme it was:
    # after winsor at 90% its z-score should be far below 50/its-raw-magnitude
    assert out["forward_return_20d"].max() < 3.0


def test_candidate_features_include_cicc_and_agent_selection_scores():
    columns = [
        "symbol", "trade_date", "forward_return_5d", "alpha001",
        "cicc_stock_selection_score", "cicc_sector_selection_score",
        "agent_stock_score", "technical_agent_score",
    ]

    out = _candidate_feature_names(columns, "short_5d")

    assert "cicc_stock_selection_score" in out
    assert "cicc_sector_selection_score" in out
    assert "agent_stock_score" in out
    assert "technical_agent_score" in out


def test_candidate_features_core30_uses_only_core_columns():
    columns = [
        "symbol", "trade_date", "forward_return_5d", "alpha001",
        "core_policy_score", "core_sentiment_score", "old_dealer_risk_score",
        "momentum_5d",
    ]

    out = _candidate_feature_names(columns, "short_5d", feature_policy="core30")

    assert out == ["core_policy_score", "core_sentiment_score", "old_dealer_risk_score", "momentum_5d"]
    assert "alpha001" not in out


def test_filter_by_regime_dates_keeps_only_requested_regime():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    panel = pd.DataFrame({
        "trade_date": dates,
        "symbol": ["A", "A", "A"],
        "value": [1, 2, 3],
    })
    regimes = pd.Series(["bull", "bear", "bull"], index=dates)

    out = _filter_by_regime_dates(panel, regimes, regimes=["bull"], min_rows=1)

    assert out["trade_date"].tolist() == [dates[0], dates[2]]
