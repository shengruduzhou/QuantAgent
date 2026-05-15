from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any


ORDER_FORBIDDEN_KEYS: tuple[str, ...] = (
    "order",
    "orders",
    "order_intent",
    "order_intents",
    "broker_order",
    "broker_orders",
    "submit_order",
)

EVIDENCE_REQUIRED_FIELDS: tuple[str, ...] = (
    "source",
    "available_at",
    "raw_hash",
    "confidence",
)

DOWNSTREAM_DECISION_AGENTS: tuple[str, ...] = (
    "portfolio_construction_agent",
    "hedge_decision_agent",
    "ashare_execution_agent",
    "risk_gate_agent",
    "backtest_attribution_agent",
)

AUDIT_TRAIL_KEYS: tuple[str, ...] = (
    "audit_trail",
    "audit_log",
    "audit",
    "decision_id",
    "final_decision_reason",
    "rationale",
)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    responsibility: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    existing_extension_points: tuple[str, ...] = ()
    new_modules: tuple[str, ...] = ()
    can_emit_orders: bool = False
    point_in_time_required: bool = True


@dataclass(frozen=True)
class AgentValidationResult:
    agent_name: str
    passed: bool
    errors: tuple[str, ...]


class AgentContractViolation(ValueError):
    """Raised when a runtime agent output violates the V7 safety contract."""


V7_AGENT_SPECS: tuple[AgentSpec, ...] = (
    AgentSpec(
        name="policy_agent",
        responsibility="Parse policy documents, authority level, affected industries, effective dates, and policy horizon.",
        inputs=("PolicyDocument", "PolicyTaxonomy", "as_of_date"),
        outputs=("EvidenceRecord", "ThemePolicyScore"),
        existing_extension_points=("src/quantagent/agents/policy_agent.py", "src/quantagent/data/event_store.py"),
        new_modules=(
            "src/quantagent/data/providers/official_policy_provider.py",
            "src/quantagent/themes/policy_crawler.py",
            "src/quantagent/themes/policy_parser.py",
        ),
    ),
    AgentSpec(
        name="theme_discovery_agent",
        responsibility="Detect theme strength, lifecycle stage, crowding, bubble risk, and invalidation evidence.",
        inputs=("EvidenceRecord", "MarketBreadth", "SectorFlow", "IndustryFundamentalPanel"),
        outputs=("ThemeProfile", "EvidenceRecord", "RiskFlag"),
        new_modules=(
            "src/quantagent/themes/theme_extractor.py",
            "src/quantagent/themes/theme_lifecycle.py",
        ),
    ),
    AgentSpec(
        name="industry_chain_graph_agent",
        responsibility="Reason an evidence-driven industry chain from policy/disclosure/news data — no static templates.",
        inputs=("ThemeProfile", "EvidenceRecord"),
        outputs=("ChainNode", "ChainEdge", "EvidenceRecord"),
        new_modules=(
            "src/quantagent/themes/industry_chain_reasoner.py",
            "src/quantagent/themes/company_exposure_mapper.py",
        ),
    ),
    AgentSpec(
        name="thematic_universe_builder",
        responsibility="Build dynamic A-share theme pools with tradability, exposure, fraud, valuation, and liquidity filters.",
        inputs=("BaseUniverse", "ChainNode", "ChainEdge", "EvidenceRecord", "FundamentalScore", "MarketState"),
        outputs=("ThematicUniverseMember", "StockPoolSelectionReport", "Constraint", "RiskFlag"),
        existing_extension_points=("src/quantagent/data/universe.py",),
        new_modules=("src/quantagent/themes/theme_universe_builder.py", "src/quantagent/themes/stock_pool_selector.py"),
    ),
    AgentSpec(
        name="fundamental_due_diligence_agent",
        responsibility="Score business model fit, revenue exposure, profitability, orders, capacity, financial quality, and governance.",
        inputs=("FinancialStatement", "Announcement", "IndustryData", "ThematicUniverseMember"),
        outputs=("FundamentalScore", "FundamentalDueDiligenceReport", "EvidenceRecord", "RiskFlag"),
        existing_extension_points=("src/quantagent/fundamental/scores.py", "src/quantagent/fundamental/quality.py"),
        new_modules=("src/quantagent/fundamental/financial_statement_agent.py", "src/quantagent/fundamental/due_diligence.py"),
    ),
    AgentSpec(
        name="valuation_agent",
        responsibility="Score PE/PB/PS/EVEBITDA/PEG/FCF yield, relative percentile, DCF, reverse DCF, and valuation overextension.",
        inputs=("FundamentalPanel", "IndustryValuationPanel", "ThemeProfile"),
        outputs=("ValuationScore", "EvidenceRecord", "RiskFlag"),
        existing_extension_points=("src/quantagent/fundamental/valuation.py",),
        new_modules=("src/quantagent/fundamental/valuation_agent.py",),
    ),
    AgentSpec(
        name="financial_fraud_risk_agent",
        responsibility="Detect accounting anomaly, regulatory penalty, audit opinion, cashflow mismatch, receivable and inventory risk.",
        inputs=("FinancialStatement", "RegulatoryDisclosure", "AuditOpinion", "PriceActionAroundDisclosure"),
        outputs=("FraudRiskScore", "EvidenceRecord", "RiskFlag"),
        existing_extension_points=("src/quantagent/fundamental/forensic_accounting.py",),
        new_modules=(
            "src/quantagent/fundamental/fraud_risk_agent.py",
            "src/quantagent/fundamental/confidence_adjuster.py",
        ),
    ),
    AgentSpec(
        name="news_credibility_agent",
        responsibility="Separate primary-source events from headlines, rumors, duplicated reposts, and low-credibility sentiment.",
        inputs=("NewsItem", "Announcement", "PolicyDocument", "IndustryData"),
        outputs=("NewsCredibilityScore", "EvidenceRecord", "RiskFlag"),
        existing_extension_points=("src/quantagent/agents/news_agent.py",),
        new_modules=("src/quantagent/credibility/news_credibility_agent.py",),
    ),
    AgentSpec(
        name="sentiment_agent",
        responsibility="Estimate short-cycle sentiment after source reliability and rumor discounts.",
        inputs=("NewsCredibilityScore", "SocialMediaPanel", "MarketAttentionPanel"),
        outputs=("SentimentScore", "EvidenceRecord"),
        existing_extension_points=("src/quantagent/agents/sentiment_agent.py",),
    ),
    AgentSpec(
        name="market_regime_agent",
        responsibility="Classify market regime, breadth, liquidity, volatility, drawdown risk, and gross exposure cap.",
        inputs=("MarketPanel", "IndexPanel", "MacroPanel", "FundFlowPanel"),
        outputs=("MarketRegimeSnapshot", "EvidenceRecord", "Constraint"),
        existing_extension_points=("src/quantagent/quant_math/regime.py",),
    ),
    AgentSpec(
        name="sector_rotation_agent",
        responsibility="Score sector trend, relative strength, capital flow, crowding, leadership, and fade signals.",
        inputs=("SectorPanel", "MarketRegimeSnapshot", "ThemeProfile"),
        outputs=("SectorRotationScore", "EvidenceRecord", "RiskFlag"),
        existing_extension_points=("src/quantagent/agents/sector_rotation_agent.py", "src/quantagent/factors/sector_rotation.py"),
    ),
    AgentSpec(
        name="factor_applicability_agent",
        responsibility="Bind factors to valid universes, sectors, themes, regimes, horizons, capacity, decay, and lifecycle stage.",
        inputs=("FactorLifecycleReport", "ThematicUniverseMember", "MarketRegimeSnapshot"),
        outputs=("FactorApplicability", "Constraint", "RiskFlag"),
        existing_extension_points=("src/quantagent/factors/lifecycle.py", "src/quantagent/factors/governance.py"),
        new_modules=("src/quantagent/factors/factor_applicability_agent.py",),
    ),
    AgentSpec(
        name="multi_horizon_alpha_agent",
        responsibility="Produce 1D/5D/20D/60D/120D/126D alpha with conformal confidence, intervals, contributions, and penalties.",
        inputs=("FeatureStore", "EvidenceRecord", "FactorApplicability", "MarketRegimeSnapshot"),
        outputs=("MultiHorizonAlpha", "EvidenceRecord"),
        existing_extension_points=("src/quantagent/models/v6_model_system.py", "src/quantagent/models/v6_outputs.py"),
        new_modules=("src/quantagent/models/v7_multi_horizon.py",),
    ),
    AgentSpec(
        name="technical_timing_agent",
        responsibility="Produce timing zones and invalidation levels without deciding whether the theme or fundamentals are true.",
        inputs=("MarketPanel", "MultiHorizonAlpha", "ThemeProfile", "ThematicUniverseMember"),
        outputs=("TechnicalTimingPlan", "EvidenceRecord", "Constraint"),
        existing_extension_points=("src/quantagent/quant_math/technical_indicators.py",),
    ),
    AgentSpec(
        name="portfolio_construction_agent",
        responsibility="Allocate sleeve weights and target weights from approved evidence, alpha, risk, liquidity, and regime inputs.",
        inputs=("ThematicUniverseMember", "MultiHorizonAlpha", "TechnicalTimingPlan", "MarketRegimeSnapshot", "RiskFlag"),
        outputs=("PortfolioPlan", "Constraint"),
        existing_extension_points=("src/quantagent/portfolio/allocator.py", "src/quantagent/portfolio/v6_portfolio_service.py"),
        new_modules=("src/quantagent/portfolio/strategic_tactical_allocator.py",),
    ),
    AgentSpec(
        name="hedge_decision_agent",
        responsibility="Choose cash, exposure reduction, concentration reduction, defensive replacement, or legal hedge tools.",
        inputs=("PortfolioPlan", "MarketRegimeSnapshot", "RiskGateReport", "BacktestAttributionReport"),
        outputs=("HedgeDecision", "Constraint", "RiskFlag"),
        new_modules=(
            "src/quantagent/portfolio/hedge_decision_engine.py",
            "src/quantagent/portfolio/sector_etf_allocator.py",
        ),
    ),
    AgentSpec(
        name="ashare_execution_agent",
        responsibility="Simulate A-share retail execution constraints before OrderManager receives target weights.",
        inputs=("PortfolioPlan", "MarketState", "PositionState", "CostModel"),
        outputs=("ExecutionConstraintReport", "Constraint", "RiskFlag"),
        existing_extension_points=("src/quantagent/backtest/engine.py", "src/quantagent/execution/virtual_broker.py"),
        new_modules=("src/quantagent/backtest/tplus1_engine.py",),
    ),
    AgentSpec(
        name="risk_gate_agent",
        responsibility="Reject, reduce, block, or require cash buffers before any execution-preparation path.",
        inputs=("PortfolioPlan", "ExecutionConstraintReport", "MarketRegimeSnapshot", "FraudRiskScore"),
        outputs=("RiskGateReport", "Constraint", "RiskFlag"),
        existing_extension_points=("src/quantagent/risk/risk_gate.py", "src/quantagent/risk/kill_switch.py"),
    ),
    AgentSpec(
        name="backtest_attribution_agent",
        responsibility="Run PIT walk-forward backtests with A-share constraints and theme, factor, agent, risk, and hedge attribution.",
        inputs=("PortfolioPlan", "EvidenceRecord", "MarketPanel", "ExecutionConstraintReport"),
        outputs=("BacktestAttributionReport", "EvidenceRecord"),
        existing_extension_points=("src/quantagent/backtest/engine.py", "src/quantagent/quant_math/factor_attribution.py"),
        new_modules=("src/quantagent/backtest/event_driven_theme_backtester.py",),
    ),
    AgentSpec(
        name="audit_agent",
        responsibility="Persist data versions, model versions, feature versions, evidence hashes, risk results, and final rationale.",
        inputs=("EvidenceRecord", "PortfolioPlan", "RiskGateReport", "BacktestAttributionReport"),
        outputs=("AuditLogRecord",),
        existing_extension_points=("src/quantagent/execution/audit.py", "src/quantagent/execution/audit_replay.py"),
    ),
)


def get_agent_spec(name: str) -> AgentSpec:
    for spec in V7_AGENT_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def validate_agent_specs(specs: tuple[AgentSpec, ...] = V7_AGENT_SPECS) -> AgentValidationResult:
    errors: list[str] = []
    for spec in specs:
        if spec.can_emit_orders:
            errors.append(f"{spec.name}:can_emit_orders_true")
        for output in spec.outputs:
            if "OrderIntent" in output or output.lower() in ORDER_FORBIDDEN_KEYS:
                errors.append(f"{spec.name}:forbidden_output:{output}")
    return AgentValidationResult("V7_AGENT_SPECS", not errors, tuple(errors))


def validate_agent_output(agent_name: str, payload: Any) -> AgentValidationResult:
    spec = get_agent_spec(agent_name)
    errors: list[str] = []
    if spec.can_emit_orders:
        errors.append(f"{agent_name}:can_emit_orders_true")
    materialized = _materialize(payload)
    _scan_for_order_payload(materialized, agent_name, errors)
    _scan_for_evidence_payload(materialized, agent_name, errors)
    if agent_name in DOWNSTREAM_DECISION_AGENTS and not _has_audit_trail(materialized):
        errors.append(f"{agent_name}:missing_audit_trail")
    return AgentValidationResult(agent_name, not errors, tuple(errors))


def assert_agent_output_valid(agent_name: str, payload: Any) -> None:
    result = validate_agent_output(agent_name, payload)
    if not result.passed:
        raise AgentContractViolation(";".join(result.errors))


def _materialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _materialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_materialize(item) for item in value]
    return value


def _scan_for_order_payload(value: Any, agent_name: str, errors: list[str], path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in ORDER_FORBIDDEN_KEYS or "orderintent" in key_text:
                errors.append(f"{agent_name}:forbidden_order_field:{path}.{key}")
            _scan_for_order_payload(item, agent_name, errors, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_for_order_payload(item, agent_name, errors, f"{path}[{index}]")


def _scan_for_evidence_payload(value: Any, agent_name: str, errors: list[str], path: str = "$") -> None:
    if isinstance(value, dict):
        if _looks_like_evidence(value):
            missing = [
                field
                for field in EVIDENCE_REQUIRED_FIELDS
                if field not in value or value.get(field) in (None, "")
            ]
            if missing:
                errors.append(f"{agent_name}:evidence_missing:{path}:{','.join(missing)}")
        for key, item in value.items():
            _scan_for_evidence_payload(item, agent_name, errors, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_for_evidence_payload(item, agent_name, errors, f"{path}[{index}]")


def _looks_like_evidence(value: dict[str, Any]) -> bool:
    keys = set(value)
    if "evidence_id" in keys:
        return True
    return {"source", "confidence"} <= keys and ("raw_hash" in keys or "hash" in keys or "published_at" in keys)


def _has_audit_trail(value: Any) -> bool:
    if isinstance(value, dict):
        if any(key in value and value.get(key) not in (None, "", (), []) for key in AUDIT_TRAIL_KEYS):
            return True
        return any(_has_audit_trail(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_audit_trail(item) for item in value)
    return False
