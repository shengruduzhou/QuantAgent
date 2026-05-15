import pandas as pd
import pytest

from quantagent.backtest import build_pit_evidence_slice
from quantagent.data.ingestion import (
    EVIDENCE_COLUMNS,
    EvidenceStore,
    EvidenceStoreConfig,
)
from quantagent.data.ingestion.evidence_store import build_evidence_quality_report
from quantagent.themes import (
    StockPoolGateConfig,
    apply_stock_pool_gate,
    gate_summary,
)
from quantagent.v7.schemas import (
    ChainRelationType,
    InvestmentHorizonBucket,
    StockPoolSelectionReport,
    ThematicUniverseMember,
    ThemeLifecycleStage,
    UniverseBucket,
)


def _member(symbol: str, theme: str, bucket: UniverseBucket, source_confidence: float = 0.8, exposure_type: ChainRelationType = ChainRelationType.DIRECT_EXPOSURE) -> ThematicUniverseMember:
    return ThematicUniverseMember(
        symbol=symbol,
        company_name=symbol,
        theme=theme,
        sub_theme="server",
        chain_node="server",
        exposure_type=exposure_type,
        exposure_score=80.0,
        revenue_exposure_estimate=0.4,
        profit_exposure_estimate=0.3,
        evidence_count=3,
        source_confidence=source_confidence,
        fundamental_score=75.0,
        valuation_score=55.0,
        quality_score=70.0,
        fraud_risk_score=25.0,
        liquidity_score=70.0,
        market_attention_score=60.0,
        theme_lifecycle_stage=ThemeLifecycleStage.FUNDAMENTAL_VALIDATION,
        entry_date="2026-05-01",
        expiry_date="2026-09-01",
        last_validated_at="2026-05-14",
        watchlist_status=bucket,
    )


def _report(theme: str, factor_names: tuple[str, ...] = ("theme_momentum",)) -> StockPoolSelectionReport:
    return StockPoolSelectionReport(
        theme_name=theme,
        horizon_bucket=InvestmentHorizonBucket.MEDIUM_TERM,
        expected_horizon_days=40,
        lifecycle_stage=ThemeLifecycleStage.FUNDAMENTAL_VALIDATION,
        core_symbols=(),
        strong_symbols=(),
        satellite_symbols=(),
        watchlist_symbols=(),
        exclusion_symbols=(),
        direct_relation_symbols=(),
        strong_relation_symbols=(),
        false_association_symbols=(),
        applicable_factor_names=factor_names,
        revalidation_interval_days=5,
        selection_rationale="test",
    )


def test_stock_pool_gate_drops_weak_and_false_association():
    members = [
        _member("600001.SH", "ai_compute", UniverseBucket.CORE_BENEFICIARY),
        _member("002371.SZ", "ai_compute", UniverseBucket.STRONG_CORRELATION),
        _member("000858.SZ", "ai_compute", UniverseBucket.WATCHLIST),
        _member("300750.SZ", "ai_compute", UniverseBucket.EXCLUSION),
        _member(
            "688981.SH",
            "ai_compute",
            UniverseBucket.OPTIONAL_SATELLITE,
            exposure_type=ChainRelationType.FALSE_ASSOCIATION,
        ),
    ]
    kept, drop_log = apply_stock_pool_gate(members, [_report("ai_compute")])
    kept_symbols = [member.symbol for member in kept]
    assert "600001.SH" in kept_symbols
    assert "002371.SZ" in kept_symbols
    assert "000858.SZ" not in kept_symbols  # watchlist
    assert "300750.SZ" not in kept_symbols  # exclusion
    assert "688981.SH" not in kept_symbols  # false association
    summary = gate_summary(drop_log)
    assert summary.get("blocked_relation", 0) >= 1
    assert summary.get("excluded_bucket", 0) >= 1


def test_stock_pool_gate_allows_high_confidence_satellite():
    member = _member(
        "688981.SH",
        "ai_compute",
        UniverseBucket.OPTIONAL_SATELLITE,
        source_confidence=0.85,
    )
    kept, _ = apply_stock_pool_gate(
        [member],
        [_report("ai_compute")],
        StockPoolGateConfig(allow_satellite_if_confidence_above=0.75),
    )
    assert len(kept) == 1
    assert kept[0].symbol == "688981.SH"


def test_stock_pool_gate_blocks_when_no_factor_coverage():
    member = _member("600001.SH", "ai_compute", UniverseBucket.CORE_BENEFICIARY)
    kept, drop_log = apply_stock_pool_gate(
        [member],
        [_report("ai_compute", factor_names=())],
        StockPoolGateConfig(require_factor_coverage=True),
    )
    assert kept == []
    assert "no_factor_coverage_for_theme" in drop_log["600001.SH"]


def test_evidence_store_round_trip(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "evidence_id": "e1",
                "source_type": "policy",
                "source_name": "gov.cn",
                "source_authority": 0.95,
                "source_reliability": 0.92,
                "is_primary_source": True,
                "is_official": True,
                "url": "https://www.gov.cn/p1",
                "title": "AI compute policy",
                "body": "Support GPU and server",
                "published_at": "2026-05-12",
                "available_at": "2026-05-12",
                "ingested_at": "2026-05-14",
                "symbol": "",
                "company_name": "",
                "theme_candidates": "ai_compute",
                "chain_node_candidates": "gpu,server",
                "event_type": "policy_support",
                "confidence": 0.85,
                "cross_validation_count": 0,
                "contradiction_count": 0,
                "horizon_days": 120,
                "decay_half_life": 90.0,
                "rumor_risk_flag": False,
                "affected_symbols": "",
                "raw_hash": "abcd",
                "point_in_time_valid": True,
            },
            {
                "evidence_id": "e2",
                "source_type": "news",
                "source_name": "eastmoney",
                "source_authority": 0.55,
                "source_reliability": 0.60,
                "is_primary_source": False,
                "is_official": False,
                "url": "https://eastmoney.com/n1",
                "title": "Server hot",
                "body": "Server market gains",
                "published_at": "2026-05-18",
                "available_at": "2026-05-18",  # leaks past as_of
                "ingested_at": "2026-05-14",
                "symbol": "600001.SH",
                "company_name": "Server Co",
                "theme_candidates": "ai_compute",
                "chain_node_candidates": "server",
                "event_type": "sentiment_positive",
                "confidence": 0.55,
                "cross_validation_count": 0,
                "contradiction_count": 0,
                "horizon_days": 5,
                "decay_half_life": 3.0,
                "rumor_risk_flag": False,
                "affected_symbols": "600001.SH",
                "raw_hash": "efgh",
                "point_in_time_valid": False,
            },
        ]
    )
    for column in EVIDENCE_COLUMNS:
        assert column in frame.columns
    store = EvidenceStore(EvidenceStoreConfig(root=str(tmp_path / "store"), file_format="csv"))
    store.write(frame)
    # Only the policy row is visible at 2026-05-14
    visible = store.read_visible("2026-05-14")
    assert len(visible) == 1
    assert visible.iloc[0]["evidence_id"] == "e1"
    quality = store.quality_report("2026-05-14")
    assert quality["row_count"] == 1
    assert quality["pit_violation_count"] == 0


def test_build_pit_evidence_slice_drops_future_rows():
    frame = pd.DataFrame(
        [
            {"evidence_id": "a", "available_at": "2026-05-10"},
            {"evidence_id": "b", "available_at": "2026-05-20"},
        ]
    )
    sliced = build_pit_evidence_slice(frame, "2026-05-14")
    assert list(sliced["evidence_id"]) == ["a"]


def test_evidence_quality_report_counts_duplicates_missing_columns_and_pit():
    frame = pd.DataFrame(
        [
            {"source_reliability": 0.9, "available_at": "2026-05-10", "raw_hash": "h1"},
            {"source_reliability": 0.3, "available_at": "2026-05-20", "raw_hash": "h1"},
        ]
    )
    report = build_evidence_quality_report(
        frame,
        as_of_date="2026-05-14",
        required_columns=("available_at", "raw_hash", "source_reliability", "source"),
    )

    assert report["row_count"] == 2
    assert report["duplicate_rate"] == 0.5
    assert report["pit_violation_count"] == 1
    assert report["source_reliability_mean"] == 0.6
    assert report["missing_columns"] == ["source"]
