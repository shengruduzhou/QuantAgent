"""Horizon-grouped model spec tests."""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.training.horizon_models import (
    DEFAULT_HORIZON_SPECS,
    HorizonClass,
    HorizonEnsembleWeights,
    HorizonModelSpec,
    build_all_horizon_bundles,
    build_horizon_bundle,
    ensemble_horizon_predictions,
    get_horizon_spec,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=20)
    rows = []
    for d in dates:
        for sym in ("600519.SH", "000001.SZ"):
            rows.append({
                "symbol": sym,
                "trade_date": d,
                "available_at": d,
                # short-horizon features
                "rsi_14": 50.0,
                "macd_signal": 0.0,
                "vol_ratio_5d": 1.2,
                "north_flow": 100.0,
                "limit_up_count_5d": 0,
                "auction_strength": 0.5,
                # mid-horizon features
                "sector_strength_20d": 0.6,
                "broker_consensus_score": 0.4,
                "ma20": 100.0,
                "ma60": 95.0,
                "policy_signal": 0.3,
                # long-horizon features
                "pe_ttm": 25.0,
                "pb": 3.0,
                "roe": 0.18,
                "revenue_yoy": 0.20,
                "valuation_percentile": 0.4,
                # labels
                "forward_return_1d": 0.001,
                "forward_return_5d": 0.01,
                "forward_return_20d": 0.03,
                "forward_return_60d": 0.06,
                "forward_return_120d": 0.10,
                # noise column that should NOT enter any class
                "useless_random_field": 1.0,
            })
    return pd.DataFrame(rows)


def test_get_horizon_spec_returns_each_class():
    short = get_horizon_spec(HorizonClass.SHORT)
    mid = get_horizon_spec(HorizonClass.MID)
    lng = get_horizon_spec(HorizonClass.LONG)
    assert short.horizons == (1, 5)
    assert mid.horizons == (5, 20)
    assert lng.horizons == (60, 120)


def test_spec_label_columns_are_forward_return_format():
    spec = get_horizon_spec("short_5d")
    assert spec.label_columns == ("forward_return_1d", "forward_return_5d")


def test_short_bundle_keeps_only_short_features(panel):
    spec = get_horizon_spec(HorizonClass.SHORT)
    bundle = build_horizon_bundle(panel, spec=spec)
    assert "rsi_14" in bundle.feature_columns
    assert "vol_ratio_5d" in bundle.feature_columns
    assert "north_flow" in bundle.feature_columns
    # long-horizon columns must not leak in
    assert "pe_ttm" not in bundle.feature_columns
    assert "roe" not in bundle.feature_columns
    assert "useless_random_field" not in bundle.feature_columns


def test_long_bundle_keeps_only_long_features(panel):
    spec = get_horizon_spec(HorizonClass.LONG)
    bundle = build_horizon_bundle(panel, spec=spec)
    assert "pe_ttm" in bundle.feature_columns
    assert "roe" in bundle.feature_columns
    assert "revenue_yoy" in bundle.feature_columns
    assert "rsi_14" not in bundle.feature_columns
    assert "auction_strength" not in bundle.feature_columns


def test_long_bundle_primary_label_is_longest_horizon(panel):
    spec = get_horizon_spec(HorizonClass.LONG)
    bundle = build_horizon_bundle(panel, spec=spec)
    assert bundle.primary_label == "forward_return_120d"


def test_bundle_drops_label_columns_from_features(panel):
    for cls in HorizonClass:
        spec = get_horizon_spec(cls)
        bundle = build_horizon_bundle(panel, spec=spec)
        for label in spec.label_columns:
            assert label not in bundle.feature_columns


def test_build_all_horizon_bundles_includes_every_class(panel):
    bundles = build_all_horizon_bundles(panel)
    assert set(bundles.keys()) == set(HorizonClass)


def test_bundle_drops_rows_missing_primary_label(panel):
    p = panel.copy()
    # zap forward_return_120d for a chunk
    p.loc[p["symbol"] == "000001.SZ", "forward_return_120d"] = float("nan")
    spec = get_horizon_spec(HorizonClass.LONG)
    bundle = build_horizon_bundle(p, spec=spec, drop_rows_missing_primary_label=True)
    assert (bundle.panel["symbol"] == "600519.SH").all()


def test_bundle_raises_when_no_label_column_present(panel):
    p = panel.drop(columns=["forward_return_60d", "forward_return_120d"])
    spec = get_horizon_spec(HorizonClass.LONG)
    with pytest.raises(ValueError):
        build_horizon_bundle(p, spec=spec)


def test_build_all_skips_classes_without_labels(panel):
    p = panel.drop(columns=["forward_return_60d", "forward_return_120d"])
    bundles = build_all_horizon_bundles(p)
    assert HorizonClass.LONG not in bundles
    assert HorizonClass.SHORT in bundles
    assert HorizonClass.MID in bundles


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def _pred_frame(class_score: float, dates: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_date": d, "symbol": "600519.SH", "alpha_score": class_score}
            for d in dates
        ]
    )


def test_ensemble_blends_three_classes_with_default_weights():
    dates = pd.bdate_range("2024-01-01", periods=3).tolist()
    out = ensemble_horizon_predictions(
        {
            HorizonClass.SHORT: _pred_frame(1.0, dates),
            HorizonClass.MID: _pred_frame(2.0, dates),
            HorizonClass.LONG: _pred_frame(3.0, dates),
        }
    )
    # default weights 0.30 * 1 + 0.45 * 2 + 0.25 * 3 = 1.95
    assert all(abs(v - 1.95) < 1e-9 for v in out["composite_score"])
    assert "short_5d_score" in out.columns
    assert "long_30d_120d_score" in out.columns


def test_ensemble_handles_missing_class_as_abstain():
    dates = pd.bdate_range("2024-01-01", periods=2).tolist()
    out = ensemble_horizon_predictions(
        {
            HorizonClass.SHORT: _pred_frame(1.0, dates),
            HorizonClass.MID: _pred_frame(2.0, dates),
            # long absent — should be treated as 0
        }
    )
    # 0.30*1 + 0.45*2 + 0.25*0 = 1.20
    assert all(abs(v - 1.20) < 1e-9 for v in out["composite_score"])


def test_ensemble_custom_weights():
    dates = pd.bdate_range("2024-01-01", periods=2).tolist()
    w = HorizonEnsembleWeights(short=0.10, mid=0.20, long=0.70)
    out = ensemble_horizon_predictions(
        {
            HorizonClass.SHORT: _pred_frame(1.0, dates),
            HorizonClass.MID: _pred_frame(2.0, dates),
            HorizonClass.LONG: _pred_frame(3.0, dates),
        },
        weights=w,
    )
    # 0.10 + 0.40 + 2.10 = 2.60
    assert all(abs(v - 2.60) < 1e-9 for v in out["composite_score"])


def test_ensemble_returns_empty_with_no_inputs():
    out = ensemble_horizon_predictions({})
    assert len(out) == 0
    assert "composite_score" in out.columns
