from __future__ import annotations

from dataclasses import dataclass


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


V7_AGENT_SPECS: tuple[AgentSpec, ...] = (
    AgentSpec(
        name="policy_agent",
        responsibility="Parse policy documents, authority level, affected industries, effective dates, and policy horizon.",
        inputs=("PolicyDocument", "PolicyTaxonomy", "as_of_date"),
        outputs=("EvidenceRecord", "ThemePolicyScore"),
        existing_extension_points=("src/quantagent/agents/policy_agent.py", "src/quantagent/data/event_store.py"),
        new_modules=(
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
        responsibility="Expand a theme into chain nodes, relation types, bottlenecks, substitution paths, and listed-company mappings.",
        inputs=("ThemeProfile", "IndustryChainSeed", "EvidenceRecord"),
        outputs=("ChainNode", "ChainEdge", "EvidenceRecord"),
        new_modules=("src/quantagent/themes/industry_chain_graph.py",),
    ),
    AgentSpec(
        name="thematic_universe_builder",
        responsibility="Build dynamic A-share theme pools with tradability, exposure, fraud, valuation, and liquidity filters.",
        inputs=("BaseUniverse", "ChainNode", "ChainEdge", "EvidenceRecord", "FundamentalScore", "MarketState"),
        outputs=("ThematicUniverseMember", "Constraint", "RiskFlag"),
        existing_extension_points=("src/quantagent/data/universe.py",),
        new_modules=("src/quantagent/themes/theme_universe_builder.py",),
    ),
    AgentSpec(
        name="fundamental_due_diligence_agent",
        responsibility="Score business model fit, revenue exposure, profitability, orders, capacity, financial quality, and governance.",
        inputs=("FinancialStatement", "Announcement", "IndustryData", "ThematicUniverseMember"),
        outputs=("FundamentalScore", "EvidenceRecord", "RiskFlag"),
        existing_extension_points=("src/quantagent/fundamental/scores.py", "src/quantagent/fundamental/quality.py"),
        new_modules=("src/quantagent/fundamental/financial_statement_agent.py",),
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
