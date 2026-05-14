import pandas as pd

from quantagent.themes.company_exposure_mapper import (
    ExposureMapperConfig,
    map_company_exposures,
)
from quantagent.v7.schemas import (
    ChainNode,
    ChainRelationType,
    EventType,
    EvidenceRecord,
    SourceType,
    ThemeLifecycleStage,
    UniverseBucket,
)
from quantagent.v7.scoring import classify_universe_bucket


def _evidence(
    evidence_id: str,
    symbol: str,
    event_type: EventType,
    theme: str = "ai_compute",
    chain_node: str = "server",
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        source="sse",
        source_type=SourceType.EXCHANGE_DISCLOSURE,
        source_authority_level=0.90,
        timestamp="2026-05-01",
        published_at="2026-05-01",
        symbol=symbol,
        theme=theme,
        chain_node=chain_node,
        event_type=event_type,
        confidence=0.80,
    )


def test_exposure_mapper_promotes_company_with_real_revenue_and_order_evidence():
    nodes = [ChainNode(node_id="server", node_name="AI server")]
    profiles = pd.DataFrame(
        [
            {
                "symbol": "600001.SH",
                "company_name": "Server Co",
                "business_scope": "AI server and data center",
                "server_revenue_exposure": 0.45,
            }
        ]
    )
    evidence = [
        _evidence("e1", "600001.SH", EventType.ORDER_CONFIRMED),
        _evidence("e2", "600001.SH", EventType.EARNINGS_GROWTH),
    ]
    frame = map_company_exposures(
        profiles,
        "ai_compute",
        nodes,
        evidence=evidence,
        node_keywords={"server": ("server", "ai server", "data center")},
    )
    row = frame.iloc[0]
    assert row["exposure_type"] == ChainRelationType.DIRECT_EXPOSURE.value
    assert row["revenue_exposure_estimate"] == 0.45
    assert row["order_evidence_count"] >= 1
    assert row["source_confidence"] >= 0.65


def test_exposure_mapper_marks_disclaimer_as_false_association():
    nodes = [ChainNode(node_id="server", node_name="AI server")]
    profiles = pd.DataFrame(
        [
            {
                "symbol": "000858.SZ",
                "company_name": "Liquor Co",
                "business_scope": "high-end liquor 蒸馏酒",
                "announcement_text": "公司核心业务为白酒，AI 服务器业务占比较小，未对公司业绩产生重大影响",
            }
        ]
    )
    frame = map_company_exposures(
        profiles,
        "ai_compute",
        nodes,
        node_keywords={"server": ("ai 服务器", "server")},
        config=ExposureMapperConfig(direct_revenue_threshold=0.15),
    )
    if not frame.empty:
        row = frame.iloc[0]
        assert row["exposure_type"] == ChainRelationType.FALSE_ASSOCIATION.value
        assert bool(row["company_disclaimer_detected"]) is True


def test_exposure_mapper_marks_news_only_as_weak_association():
    nodes = [ChainNode(node_id="cloud", node_name="cloud application")]
    profiles = pd.DataFrame(
        [
            {
                "symbol": "002000.SZ",
                "company_name": "Cloud Co",
                "business_scope": "cloud platform integration services",
            }
        ]
    )
    evidence = [
        _evidence("n1", "002000.SZ", EventType.SENTIMENT_POSITIVE, chain_node="cloud"),
    ]
    frame = map_company_exposures(
        profiles,
        "ai_compute",
        nodes,
        evidence=evidence,
        node_keywords={"cloud": ("cloud", "platform")},
    )
    row = frame.iloc[0]
    assert row["exposure_type"] == ChainRelationType.WEAK_ASSOCIATION.value
    assert row["source_confidence"] <= 0.45


def test_universe_bucket_blocks_core_without_revenue_proof():
    bucket = classify_universe_bucket(
        exposure_score=85.0,
        fundamental_score=75.0,
        fraud_risk_score=30.0,
        liquidity_score=70.0,
        source_confidence=0.80,
        evidence_count=4,
        valuation_score=55.0,
        revenue_exposure_estimate=0.05,  # below 15% threshold
        exposure_type="direct_exposure",
        order_evidence_count=2,
    )
    assert bucket == UniverseBucket.STRONG_CORRELATION


def test_universe_bucket_blocks_core_for_non_direct_relation_when_provided():
    bucket = classify_universe_bucket(
        exposure_score=85.0,
        fundamental_score=78.0,
        fraud_risk_score=25.0,
        liquidity_score=70.0,
        source_confidence=0.85,
        evidence_count=4,
        revenue_exposure_estimate=0.30,
        exposure_type="weak_association",
        order_evidence_count=2,
    )
    assert bucket == UniverseBucket.STRONG_CORRELATION


def test_universe_bucket_excludes_explicit_disclaimer():
    bucket = classify_universe_bucket(
        exposure_score=70.0,
        fundamental_score=80.0,
        fraud_risk_score=30.0,
        liquidity_score=70.0,
        source_confidence=0.80,
        evidence_count=3,
        company_disclaimer_detected=True,
        revenue_exposure_estimate=0.02,
    )
    assert bucket == UniverseBucket.EXCLUSION


def test_universe_bucket_keeps_core_when_all_strict_signals_present():
    bucket = classify_universe_bucket(
        exposure_score=90.0,
        fundamental_score=80.0,
        fraud_risk_score=25.0,
        liquidity_score=70.0,
        source_confidence=0.80,
        evidence_count=4,
        revenue_exposure_estimate=0.30,
        exposure_type="direct_exposure",
        order_evidence_count=2,
    )
    assert bucket == UniverseBucket.CORE_BENEFICIARY
