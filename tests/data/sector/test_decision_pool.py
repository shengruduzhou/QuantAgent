"""SectorPoolV8 decision-axis pool tests (spec section 3)."""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.sector.decision_pool import (
    SECTOR_POOL_V8_COLUMNS,
    SectorPoolV8Builder,
    SectorPoolV8Config,
    build_sector_pool_v8,
)


@pytest.fixture
def sectors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sector_code": "Semi", "sector_name": "半导体"},
            {"sector_code": "Bank", "sector_name": "银行"},
            {"sector_code": "RealEstate", "sector_name": "地产"},
        ]
    )


def test_empty_input_returns_empty_canonical_frame():
    result = build_sector_pool_v8(date=pd.Timestamp("2024-03-01"), sectors=pd.DataFrame())
    assert list(result.frame.columns) == list(SECTOR_POOL_V8_COLUMNS)
    assert len(result.frame) == 0


def test_pool_has_canonical_schema(sectors):
    result = build_sector_pool_v8(date=pd.Timestamp("2024-03-01"), sectors=sectors)
    assert list(result.frame.columns) == list(SECTOR_POOL_V8_COLUMNS)
    assert len(result.frame) == len(sectors)


def test_pool_uses_capital_flow_thesis(sectors):
    theses = pd.DataFrame([
        {"direction_kind": "sector", "direction_value": "Semi",
         "thesis_sign": 0.8, "confidence": 0.9, "validation_status": "verified"},
        {"direction_kind": "sector", "direction_value": "Bank",
         "thesis_sign": -0.5, "confidence": 0.6, "validation_status": "partially_verified"},
    ])
    result = build_sector_pool_v8(
        date=pd.Timestamp("2024-03-01"), sectors=sectors,
        capital_flow_theses=theses,
    )
    semi = result.frame[result.frame["sector_code"] == "Semi"].iloc[0]
    bank = result.frame[result.frame["sector_code"] == "Bank"].iloc[0]
    re = result.frame[result.frame["sector_code"] == "RealEstate"].iloc[0]
    assert semi["policy_score"] > bank["policy_score"]
    assert pd.isna(re["policy_score"])  # no thesis for RealEstate


def test_pool_uses_capital_flow_panel(sectors):
    dates = pd.bdate_range("2024-02-26", periods=4)
    capital_flow_panel = pd.DataFrame(
        [
            {"trade_date": d, "sector_code": "Semi", "net_flow": 5.0}
            for d in dates
        ] + [
            {"trade_date": d, "sector_code": "Bank", "net_flow": -2.0}
            for d in dates
        ] + [
            {"trade_date": d, "sector_code": "RealEstate", "net_flow": 0.0}
            for d in dates
        ]
    )
    result = build_sector_pool_v8(
        date=pd.Timestamp("2024-03-01"), sectors=sectors,
        capital_flow_panel=capital_flow_panel,
    )
    semi = result.frame[result.frame["sector_code"] == "Semi"].iloc[0]
    bank = result.frame[result.frame["sector_code"] == "Bank"].iloc[0]
    assert semi["capital_flow_score"] > bank["capital_flow_score"]


def test_pool_uses_market_strength(sectors):
    dates = pd.bdate_range("2024-02-15", periods=10)
    rets = []
    for d in dates:
        rets.append({"trade_date": d, "sector_code": "Semi", "ret": 0.02})
        rets.append({"trade_date": d, "sector_code": "Bank", "ret": -0.005})
        rets.append({"trade_date": d, "sector_code": "RealEstate", "ret": 0.001})
    result = build_sector_pool_v8(
        date=pd.Timestamp("2024-03-01"), sectors=sectors,
        sector_returns=pd.DataFrame(rets),
    )
    sorted_by_strength = result.frame.sort_values("market_strength_score", ascending=False)
    assert sorted_by_strength.iloc[0]["sector_code"] == "Semi"
    assert sorted_by_strength.iloc[-1]["sector_code"] == "Bank"


def test_pool_valuation_percentile_lower_is_cheaper(sectors):
    # Semi PE=10 (cheap), Bank PE=20 (mid), RealEstate PE=50 (expensive)
    valuation = pd.DataFrame([
        {"trade_date": pd.Timestamp("2024-03-01"), "sector_code": "Semi", "pe_ttm": 10.0},
        {"trade_date": pd.Timestamp("2024-03-01"), "sector_code": "Bank", "pe_ttm": 20.0},
        {"trade_date": pd.Timestamp("2024-03-01"), "sector_code": "RealEstate", "pe_ttm": 50.0},
    ])
    result = build_sector_pool_v8(
        date=pd.Timestamp("2024-03-01"), sectors=sectors,
        sector_valuation=valuation,
    )
    semi = result.frame[result.frame["sector_code"] == "Semi"].iloc[0]
    re = result.frame[result.frame["sector_code"] == "RealEstate"].iloc[0]
    # Lower percentile = cheaper
    assert semi["valuation_percentile"] < re["valuation_percentile"]


def test_pool_confidence_grows_with_axis_coverage(sectors):
    """One sector with all axes vs another with one axis → higher confidence."""
    dates = pd.bdate_range("2024-02-26", periods=4)
    cf = pd.DataFrame([
        {"trade_date": d, "sector_code": "Semi", "net_flow": 3.0} for d in dates
    ])
    ret = pd.DataFrame([
        {"trade_date": d, "sector_code": "Semi", "ret": 0.01} for d in dates
    ])
    liq = pd.DataFrame([
        {"trade_date": d, "sector_code": "Semi", "amount": 100.0} for d in dates
    ])
    val = pd.DataFrame([
        {"trade_date": pd.Timestamp("2024-03-01"), "sector_code": "Semi", "pe_ttm": 15.0},
        {"trade_date": pd.Timestamp("2024-03-01"), "sector_code": "Bank", "pe_ttm": 15.0},
    ])
    result = build_sector_pool_v8(
        date=pd.Timestamp("2024-03-01"), sectors=sectors,
        capital_flow_panel=cf, sector_returns=ret,
        sector_liquidity=liq, sector_valuation=val,
    )
    semi_conf = result.frame[result.frame["sector_code"] == "Semi"]["confidence"].iloc[0]
    re_conf = result.frame[result.frame["sector_code"] == "RealEstate"]["confidence"].iloc[0]
    assert semi_conf is not None and re_conf is not None
    assert semi_conf > re_conf


def test_pool_final_sector_rank_dense_ranking(sectors):
    """final_sector_rank should produce dense rank starting at 1."""
    dates = pd.bdate_range("2024-02-15", periods=10)
    ret = []
    for d in dates:
        ret.append({"trade_date": d, "sector_code": "Semi", "ret": 0.02})
        ret.append({"trade_date": d, "sector_code": "Bank", "ret": 0.005})
        ret.append({"trade_date": d, "sector_code": "RealEstate", "ret": 0.001})
    result = build_sector_pool_v8(
        date=pd.Timestamp("2024-03-01"), sectors=sectors,
        sector_returns=pd.DataFrame(ret),
    )
    ranks = result.frame["final_sector_rank"].dropna().tolist()
    assert min(ranks) == 1
    assert max(ranks) <= len(sectors)


def test_pool_builder_writes_files(tmp_path, sectors):
    builder = SectorPoolV8Builder(SectorPoolV8Config(output_root=tmp_path, source_version="t1"))
    result = builder.build(date=pd.Timestamp("2024-03-01"), sectors=sectors)
    builder.write(result)
    assert (tmp_path / "silver" / "sector_pool_v8" / "sector_pool_v8.parquet").exists()
    assert (tmp_path / "manifests" / "sector_pool_v8.json").exists()


def test_pool_does_not_emit_target_weights(sectors):
    """Spec: sector pool is a filter, not a signal — must not emit target_weight."""
    result = build_sector_pool_v8(date=pd.Timestamp("2024-03-01"), sectors=sectors)
    forbidden = {"target_weight", "weight", "order_intent"}
    assert not (forbidden & set(result.frame.columns))
