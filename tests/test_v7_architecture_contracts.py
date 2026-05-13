from quantagent.v7.agent_contracts import V7_AGENT_SPECS, get_agent_spec
from quantagent.v7.dag import V7_DAILY_DAG, validate_dag
from quantagent.v7.schemas import EvidenceRecord, EventType, SourceType, ThemeLifecycleStage, UniverseBucket
from quantagent.v7.scoring import (
    classify_theme_lifecycle,
    classify_universe_bucket,
    execution_feasibility_score,
    fraud_confidence_multiplier,
    news_confidence_score,
)


def test_v7_evidence_record_hash_is_stable_and_point_in_time():
    record = EvidenceRecord(
        evidence_id="policy-20260514-001",
        source="state_council",
        source_type=SourceType.OFFICIAL_POLICY,
        source_authority_level=0.95,
        timestamp="2026-05-14T15:00:00+08:00",
        published_at="2026-05-14T10:00:00+08:00",
        theme="ai_compute",
        event_type=EventType.POLICY_SUPPORT,
        direction=1.0,
        magnitude=0.8,
        confidence=0.9,
        evidence_quality=0.9,
        source_reliability=0.95,
        cross_validation_count=2,
        horizon_days=120,
        rationale="Central policy supports compute infrastructure.",
    )

    hashed = record.with_hash()

    assert hashed.hash
    assert hashed.hash == record.with_hash().hash
    assert hashed.point_in_time_valid is True


def test_v7_agent_specs_preserve_order_boundary():
    assert len(V7_AGENT_SPECS) >= 20
    assert get_agent_spec("portfolio_construction_agent").outputs == ("PortfolioPlan", "Constraint")
    assert all(spec.can_emit_orders is False for spec in V7_AGENT_SPECS)
    assert all("OrderIntent" not in output for spec in V7_AGENT_SPECS for output in spec.outputs)


def test_v7_daily_dag_dependencies_are_ordered():
    assert not validate_dag(V7_DAILY_DAG)
    assert V7_DAILY_DAG[-1].task_id == "write_audit_log"


def test_v7_scoring_enforces_theme_fraud_news_and_execution_rules():
    assert classify_theme_lifecycle(0.9, 0.2, 0.1, 0.1, 0.1, 0.1) == ThemeLifecycleStage.POLICY_SEED
    assert classify_theme_lifecycle(0.8, 0.8, 0.2, 0.8, 0.9, 0.8) == ThemeLifecycleStage.VALUATION_BUBBLE
    assert fraud_confidence_multiplier(85.0, "fundamental") == 0.20
    assert fraud_confidence_multiplier(85.0, "news") == 0.30
    assert (
        classify_universe_bucket(
            exposure_score=88.0,
            fundamental_score=78.0,
            fraud_risk_score=25.0,
            liquidity_score=70.0,
            source_confidence=0.8,
            evidence_count=4,
        )
        == UniverseBucket.CORE_BENEFICIARY
    )
    assert (
        classify_universe_bucket(
            exposure_score=88.0,
            fundamental_score=78.0,
            fraud_risk_score=90.0,
            liquidity_score=70.0,
            source_confidence=0.8,
            evidence_count=4,
        )
        == UniverseBucket.EXCLUSION
    )
    official_news = news_confidence_score(0.9, True, True, 3, 0, 0.0)
    rumor_news = news_confidence_score(0.35, False, False, 0, 2, 0.9)
    assert official_news > rumor_news
    assert execution_feasibility_score(True, False, False, 80.0, 0.05) == 0.0
    assert execution_feasibility_score(False, False, True, 80.0, 0.05) < 0.8
