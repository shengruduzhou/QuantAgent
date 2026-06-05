from __future__ import annotations

import pandas as pd

from quantagent.factors.core_policy import (
    CORE_FEATURE_COLUMNS,
    aggregate_evidence_scores,
    build_core_factor_frame,
    core_feature_columns,
)


def test_core_factor_frame_caps_features_and_blocks_old_dealer():
    d = pd.Timestamp("2024-01-02")
    dataset = pd.DataFrame({
        "trade_date": [d, d, d, d],
        "symbol": ["A", "B", "C", "D"],
        "return_1d": [0.08, 0.04, -0.03, -0.06],
        "momentum_5d": [0.20, 0.10, -0.10, -0.20],
        "momentum_20d": [0.30, 0.12, -0.15, -0.30],
        "amount_mean_20d": [2e9, 1e9, 3e7, 2e7],
        "volume_concentration": [0.10, 0.20, 0.75, 0.90],
        "spike_minutes": [1, 2, 8, 10],
        "net_buy_pressure": [0.4, 0.2, -0.1, -0.3],
        "vwap_deviation": [0.02, 0.01, -0.01, -0.03],
        "intraday_range_pos": [0.8, 0.6, 0.3, 0.2],
        "cicc_sector_selection_score": [0.3, 0.2, -0.2, -0.4],
        "forward_return_5d": [0.01, 0.02, -0.01, -0.02],
    })
    sector_map = pd.DataFrame({
        "symbol": ["A", "B", "C", "D"],
        "sector_level_1": ["good", "good", "bad", "bad"],
    })

    out, summary = build_core_factor_frame(dataset, sector_map=sector_map)

    assert len(summary.feature_columns) == 30
    assert set(CORE_FEATURE_COLUMNS).issubset(out.columns)
    assert out.set_index("symbol").loc["D", "old_dealer_risk_score"] > out.set_index("symbol").loc["A", "old_dealer_risk_score"]
    assert out.set_index("symbol").loc["D", "old_dealer_block"] == 1
    assert core_feature_columns(out.columns) == list(CORE_FEATURE_COLUMNS)


def test_evidence_scores_do_not_fabricate_missing_sentiment_or_policy():
    ev = pd.DataFrame({
        "available_at": ["2024-01-02"],
        "symbol": ["A"],
        "confidence": [0.9],
    })

    out = aggregate_evidence_scores(ev)

    assert out.loc[0, "evidence_policy_score"] == 0.0
    assert out.loc[0, "evidence_sentiment_score"] == 0.0


def test_evidence_scores_use_explicit_policy_and_sentiment():
    ev = pd.DataFrame({
        "available_at": ["2024-01-02"],
        "symbol": ["A"],
        "confidence": [0.5],
        "policy_direction_score": [0.8],
        "sentiment_score": [-0.4],
    })

    out = aggregate_evidence_scores(ev)

    assert out.loc[0, "evidence_policy_score"] == 0.4
    assert out.loc[0, "evidence_sentiment_score"] == -0.2
