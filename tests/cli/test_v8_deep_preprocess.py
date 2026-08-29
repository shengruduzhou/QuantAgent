"""Tests for the v8_deep per-date preprocessing helpers (anti-overfit)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.cli.v8_deep import (
    _candidate_feature_names,
    _cross_sectional_normalize,
    _filter_by_regime_dates,
    _normalize_label_per_date,
    _select_feature_columns,
    _split_train_validation_test,
    _validate_training_dataset_contract,
)
from quantagent.training.ft_transformer_trainer_impl import _apply_feature_preprocessing


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


def test_candidate_alpha_ranges_are_numeric_not_lexicographic():
    columns = [
        "symbol", "trade_date", "forward_return_5d",
        "alpha001", "alpha060", "alpha061", "alpha119", "alpha120",
        "alpha121", "alpha181", "alpha182", "alpha999",
    ]

    short = _candidate_feature_names(columns, "short_5d")
    mid = _candidate_feature_names(columns, "mid_5d_30d")
    long = _candidate_feature_names(columns, "long_30d_120d")

    assert {"alpha001", "alpha060"}.issubset(short)
    assert "alpha061" not in short
    assert {"alpha060", "alpha061", "alpha119", "alpha120"}.issubset(mid)
    assert "alpha121" not in mid
    assert {"alpha120", "alpha121", "alpha181"}.issubset(long)
    assert "alpha182" not in long
    assert "alpha999" not in long


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


def test_training_contract_and_split_keep_final_test_out_of_validation():
    dates = pd.bdate_range("2024-01-02", periods=40)
    frame = pd.DataFrame({
        "symbol": ["A"] * len(dates),
        "trade_date": dates,
        "available_at": dates,
        "forward_return_5d": np.linspace(-0.02, 0.03, len(dates)),
        "label_end_5d": dates + pd.offsets.BDay(2),
        "point_in_time_valid": [True] * len(dates),
        "alpha001": np.arange(len(dates), dtype=float),
    })
    _validate_training_dataset_contract(
        frame,
        label_col="forward_return_5d",
        label_end_col="label_end_5d",
    )

    train, validation, test, manifest = _split_train_validation_test(
        frame,
        train_end=dates[24],
        embargo_days=2,
        purge_days=3,
        validation_days=5,
        test_end=dates[-1],
        label_col="forward_return_5d",
        label_end_col="label_end_5d",
        session_dates=dates,
    )

    assert train["trade_date"].max() < validation["trade_date"].min()
    assert validation["trade_date"].max() < test["trade_date"].min()
    assert (train["label_end_5d"] < validation["trade_date"].min()).all()
    assert (validation["label_end_5d"] < test["trade_date"].min()).all()
    assert manifest["semantics"] == "train_validation_untouched_test_v1_label_end_purged"
    assert manifest["purge_sessions"] == 3
    assert validation["trade_date"].nunique() == 5

    with np.testing.assert_raises_regex(ValueError, "purge_days must be >= 0"):
        _split_train_validation_test(
            frame, train_end=dates[24], embargo_days=2, purge_days=-1,
            validation_days=5, test_end=dates[-1], label_col="forward_return_5d",
            label_end_col="label_end_5d", session_dates=dates,
        )
    with np.testing.assert_raises_regex(ValueError, "validation_days must be >= 1"):
        _split_train_validation_test(
            frame, train_end=dates[24], embargo_days=2, purge_days=3,
            validation_days=0, test_end=dates[-1], label_col="forward_return_5d",
            label_end_col="label_end_5d", session_dates=dates,
        )


def test_training_contract_rejects_late_availability():
    date = pd.Timestamp("2024-01-02")
    frame = pd.DataFrame({
        "symbol": ["A"],
        "trade_date": [date],
        "available_at": [date + pd.Timedelta(days=1)],
        "point_in_time_valid": [True],
        "forward_return_5d": [0.01],
        "label_end_5d": [date + pd.Timedelta(days=7)],
    })

    with np.testing.assert_raises_regex(ValueError, "available_at after trade_date"):
        _validate_training_dataset_contract(
            frame,
            label_col="forward_return_5d",
            label_end_col="label_end_5d",
        )


def test_training_contract_requires_explicit_pit_verdict():
    date = pd.Timestamp("2024-01-02")
    frame = pd.DataFrame({
        "symbol": ["A"],
        "trade_date": [date],
        "available_at": [date],
        "forward_return_5d": [0.01],
        "label_end_5d": [date + pd.Timedelta(days=7)],
    })

    with np.testing.assert_raises_regex(ValueError, "point_in_time_valid"):
        _validate_training_dataset_contract(
            frame,
            label_col="forward_return_5d",
            label_end_col="label_end_5d",
        )


def test_untouched_test_coverage_cannot_select_a_feature():
    dates = pd.bdate_range("2024-01-02", periods=40)
    frame = pd.DataFrame({
        "symbol": ["A"] * len(dates),
        "trade_date": dates,
        "available_at": dates,
        "point_in_time_valid": [True] * len(dates),
        "forward_return_5d": np.linspace(-0.02, 0.03, len(dates)),
        "label_end_5d": dates + pd.offsets.BDay(2),
        # Only the future partition has enough coverage. A leaky full-frame
        # schema selector would include this feature.
        "alpha001": [np.nan] * 19 + list(np.arange(21, dtype=float)),
        "momentum_5d": np.arange(len(dates), dtype=float),
    })
    train, _validation, _test, _manifest = _split_train_validation_test(
        frame,
        train_end=dates[24],
        embargo_days=2,
        purge_days=3,
        validation_days=5,
        test_end=dates[-1],
        label_col="forward_return_5d",
        label_end_col="label_end_5d",
        session_dates=dates,
    )

    assert "alpha001" in _select_feature_columns(frame, "short_5d")
    assert "alpha001" not in _select_feature_columns(train, "short_5d")


def test_artifact_preprocessing_replays_cross_sectional_rank():
    frame = _toy_frame()
    contract = {
        "method": "rank",
        "group_by": "trade_date",
        "normalized_columns": ["f1", "f2"],
        "passthrough_columns": [],
    }

    replayed = _apply_feature_preprocessing(frame, ("f1", "f2"), contract)
    expected = _cross_sectional_normalize(frame.copy(), ["f1", "f2"], method="rank")

    assert np.allclose(replayed[["f1", "f2"]], expected[["f1", "f2"]])
