from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DagTask:
    task_id: str
    owner_agent: str
    depends_on: tuple[str, ...]
    outputs: tuple[str, ...]
    point_in_time_cutoff: str = "as_of_close"
    retry_policy: str = "skip_with_risk_flag"


V7_DAILY_DAG: tuple[DagTask, ...] = (
    DagTask("ingest_market_and_reference_data", "data_ingestion", (), ("MarketPanel", "BaseUniverse", "PositionState")),
    DagTask("ingest_policy_announcement_news_financials", "data_ingestion", (), ("PolicyDocument", "Announcement", "NewsItem", "FinancialStatement")),
    DagTask("parse_policy_documents", "policy_agent", ("ingest_policy_announcement_news_financials",), ("EvidenceRecord", "ThemePolicyScore")),
    DagTask("score_news_credibility", "news_credibility_agent", ("ingest_policy_announcement_news_financials",), ("NewsCredibilityScore", "EvidenceRecord")),
    DagTask("discover_themes", "theme_discovery_agent", ("parse_policy_documents", "score_news_credibility", "ingest_market_and_reference_data"), ("ThemeProfile",)),
    DagTask("build_industry_chain_graph", "industry_chain_graph_agent", ("discover_themes",), ("ChainNode", "ChainEdge")),
    DagTask("score_fundamentals", "fundamental_due_diligence_agent", ("ingest_policy_announcement_news_financials", "build_industry_chain_graph"), ("FundamentalScore",)),
    DagTask("score_fraud_risk", "financial_fraud_risk_agent", ("ingest_policy_announcement_news_financials",), ("FraudRiskScore", "RiskFlag")),
    DagTask("score_valuation", "valuation_agent", ("score_fundamentals",), ("ValuationScore",)),
    DagTask("build_thematic_universe", "thematic_universe_builder", ("build_industry_chain_graph", "score_fundamentals", "score_fraud_risk", "score_valuation", "ingest_market_and_reference_data"), ("ThematicUniverseMember",)),
    DagTask("classify_market_regime", "market_regime_agent", ("ingest_market_and_reference_data",), ("MarketRegimeSnapshot",)),
    DagTask("score_sector_rotation", "sector_rotation_agent", ("classify_market_regime", "discover_themes", "ingest_market_and_reference_data"), ("SectorRotationScore",)),
    DagTask("validate_factor_applicability", "factor_applicability_agent", ("build_thematic_universe", "classify_market_regime"), ("FactorApplicability",)),
    DagTask("predict_multi_horizon_alpha", "multi_horizon_alpha_agent", ("build_thematic_universe", "validate_factor_applicability", "classify_market_regime", "score_news_credibility"), ("MultiHorizonAlpha",)),
    DagTask("score_technical_timing", "technical_timing_agent", ("predict_multi_horizon_alpha", "ingest_market_and_reference_data", "score_sector_rotation"), ("TechnicalTimingPlan",)),
    DagTask("construct_portfolio", "portfolio_construction_agent", ("predict_multi_horizon_alpha", "score_technical_timing", "classify_market_regime", "build_thematic_universe"), ("PortfolioPlan",)),
    DagTask("decide_hedge", "hedge_decision_agent", ("construct_portfolio", "classify_market_regime"), ("HedgeDecision",)),
    DagTask("simulate_ashare_execution", "ashare_execution_agent", ("construct_portfolio", "decide_hedge", "ingest_market_and_reference_data"), ("ExecutionConstraintReport",)),
    DagTask("apply_risk_gate", "risk_gate_agent", ("construct_portfolio", "simulate_ashare_execution", "score_fraud_risk", "classify_market_regime"), ("RiskGateReport",)),
    DagTask("run_backtest_attribution", "backtest_attribution_agent", ("apply_risk_gate", "simulate_ashare_execution"), ("BacktestAttributionReport",)),
    DagTask("write_audit_log", "audit_agent", ("apply_risk_gate", "run_backtest_attribution"), ("AuditLogRecord",)),
)


def validate_dag(tasks: tuple[DagTask, ...] = V7_DAILY_DAG) -> list[str]:
    seen: set[str] = set()
    errors: list[str] = []
    for task in tasks:
        missing = [dep for dep in task.depends_on if dep not in seen]
        if missing:
            errors.append(f"{task.task_id} missing dependencies: {','.join(missing)}")
        seen.add(task.task_id)
    return errors


def dag_edges(tasks: tuple[DagTask, ...] = V7_DAILY_DAG) -> list[tuple[str, str]]:
    return [(dependency, task.task_id) for task in tasks for dependency in task.depends_on]
