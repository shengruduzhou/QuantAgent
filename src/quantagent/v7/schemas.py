from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from hashlib import sha256
import json
from typing import Any


class SourceType(str, Enum):
    OFFICIAL_POLICY = "official_policy"
    COMPANY_ANNOUNCEMENT = "company_announcement"
    EXCHANGE_DISCLOSURE = "exchange_disclosure"
    FINANCIAL_STATEMENT = "financial_statement"
    INDUSTRY_DATA = "industry_data"
    NEWS = "news"
    RESEARCH_REPORT = "research_report"
    SOCIAL_MEDIA = "social_media"
    MARKET_DATA = "market_data"
    ANALYST_ESTIMATE = "analyst_estimate"
    ALTERNATIVE_DATA = "alternative_data"


class EventType(str, Enum):
    POLICY_SUPPORT = "policy_support"
    SUBSIDY = "subsidy"
    INDUSTRIAL_PLAN = "industrial_plan"
    DEMAND_GROWTH = "demand_growth"
    SUPPLY_SHORTAGE = "supply_shortage"
    ORDER_CONFIRMED = "order_confirmed"
    EARNINGS_GROWTH = "earnings_growth"
    MARGIN_EXPANSION = "margin_expansion"
    VALUATION_REPAIR = "valuation_repair"
    CAPITAL_INFLOW = "capital_inflow"
    SENTIMENT_POSITIVE = "sentiment_positive"
    SENTIMENT_NEGATIVE = "sentiment_negative"
    REGULATORY_PENALTY = "regulatory_penalty"
    FRAUD_RISK = "fraud_risk"
    ACCOUNTING_ANOMALY = "accounting_anomaly"
    LIQUIDITY_RISK = "liquidity_risk"
    THEME_ROTATION = "theme_rotation"
    BUBBLE_WARNING = "bubble_warning"
    NO_TRADE = "no_trade"
    HEDGE_SIGNAL = "hedge_signal"


class ThemeLifecycleStage(str, Enum):
    POLICY_SEED = "policy_seed_stage"
    NARRATIVE_FORMATION = "narrative_formation_stage"
    CAPITAL_INFLOW = "capital_inflow_stage"
    FUNDAMENTAL_VALIDATION = "fundamental_validation_stage"
    EARNINGS_REALIZATION = "earnings_realization_stage"
    VALUATION_BUBBLE = "valuation_bubble_stage"
    DIVERGENCE = "divergence_stage"
    DECAY = "decay_stage"
    INVALIDATED = "invalidated_stage"


class ChainRelationType(str, Enum):
    DIRECT_EXPOSURE = "direct_exposure"
    CRITICAL_BOTTLENECK = "critical_bottleneck"
    UPSTREAM_SUPPLIER = "upstream_supplier"
    DOWNSTREAM_APPLICATION = "downstream_application"
    INFRASTRUCTURE_DEPENDENCY = "infrastructure_dependency"
    COST_BENEFICIARY = "cost_beneficiary"
    DOMESTIC_SUBSTITUTION = "domestic_substitution"
    CAPACITY_EXPANSION = "capacity_expansion"
    CUSTOMER_SUPPLIER_LINK = "customer_supplier_link"
    TECHNOLOGY_ENABLER = "technology_enabler"
    POLICY_BENEFICIARY = "policy_beneficiary"
    WEAK_ASSOCIATION = "weak_association"
    FALSE_ASSOCIATION = "false_association"


class UniverseBucket(str, Enum):
    CORE_BENEFICIARY = "core_beneficiary_pool"
    STRONG_CORRELATION = "strong_correlation_pool"
    OPTIONAL_SATELLITE = "optional_satellite_pool"
    WATCHLIST = "watchlist_pool"
    EXCLUSION = "exclusion_pool"


class SleeveType(str, Enum):
    LONG_FUNDAMENTAL = "long_fundamental"
    MEDIUM_THEME = "medium_theme"
    SHORT_EVENT = "short_event"
    SECTOR_ROTATION = "sector_rotation"
    HEDGE = "hedge"
    CASH_BUFFER = "cash_buffer"


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LIQUIDITY_CRUNCH = "liquidity_crunch"
    POLICY_DRIVEN = "policy_driven"
    THEME_SPECULATION = "theme_speculation"
    RISK_OFF = "risk_off"
    RISK_ON = "risk_on"


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source: str
    source_type: SourceType
    source_authority_level: float
    timestamp: str
    published_at: str
    effective_start_date: str | None = None
    effective_end_date: str | None = None
    symbol: str | None = None
    sector: str | None = None
    industry: str | None = None
    theme: str | None = None
    sub_theme: str | None = None
    chain_node: str | None = None
    event_type: EventType = EventType.NO_TRADE
    direction: float = 0.0
    magnitude: float = 0.0
    confidence: float = 0.5
    evidence_quality: float = 0.5
    source_reliability: float = 0.5
    cross_validation_count: int = 0
    decay_half_life: float = 5.0
    horizon_days: int = 5
    rationale: str = ""
    raw_reference: dict[str, Any] = field(default_factory=dict)
    hash: str = ""
    point_in_time_valid: bool = True
    risk_flags: tuple[str, ...] = ()

    def with_hash(self) -> "EvidenceRecord":
        return replace(self, hash=stable_record_hash(self))


@dataclass(frozen=True)
class ThemeProfile:
    theme_name: str
    theme_category: str
    theme_strength: float
    policy_strength: float
    market_strength: float
    industry_fundamental_strength: float
    capital_flow_strength: float
    news_sentiment_strength: float
    lifecycle_stage: ThemeLifecycleStage
    expected_horizon_days: int
    theme_confidence: float
    bubble_risk: float
    crowding_score: float
    expiry_date: str
    update_frequency: str
    key_evidence: tuple[str, ...] = ()
    opposing_evidence: tuple[str, ...] = ()
    required_follow_up_data: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChainNode:
    node_id: str
    node_name: str
    upstream_nodes: tuple[str, ...] = ()
    downstream_nodes: tuple[str, ...] = ()
    dependency_strength: float = 0.0
    bottleneck_score: float = 0.0
    domestic_substitution_score: float = 0.0
    supply_shortage_score: float = 0.0
    price_elasticity: float = 0.0
    profit_elasticity: float = 0.0
    demand_visibility: float = 0.0
    policy_support_score: float = 0.0
    technology_barrier: float = 0.0
    competition_intensity: float = 0.0
    listed_company_count: int = 0
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChainEdge:
    source_node_id: str
    target_node_id: str
    relation_type: ChainRelationType
    relation_strength: float
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ThematicUniverseMember:
    symbol: str
    company_name: str
    theme: str
    sub_theme: str
    chain_node: str
    exposure_type: ChainRelationType
    exposure_score: float
    revenue_exposure_estimate: float | None
    profit_exposure_estimate: float | None
    evidence_count: int
    source_confidence: float
    fundamental_score: float
    valuation_score: float
    quality_score: float
    fraud_risk_score: float
    liquidity_score: float
    market_attention_score: float
    theme_lifecycle_stage: ThemeLifecycleStage
    entry_date: str
    expiry_date: str
    last_validated_at: str
    watchlist_status: UniverseBucket
    removal_reason: str | None = None
    sector: str | None = None
    industry: str | None = None
    membership_ttl_days: int = 20
    validation_status: str = "active"
    data_quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class FundamentalScore:
    symbol: str
    fundamental_score: float
    quality_score: float
    growth_score: float
    valuation_score: float
    earnings_visibility_score: float
    fraud_risk_score: float
    management_risk_score: float
    margin_of_safety: float
    investment_horizon: int
    confidence: float
    rationale: str
    key_risks: tuple[str, ...] = ()
    required_follow_up: tuple[str, ...] = ()


@dataclass(frozen=True)
class FraudRiskScore:
    symbol: str
    beneish_m_score: float | None
    piotroski_f_score: float | None
    altman_z_score: float | None
    accruals_quality_score: float
    cashflow_quality_score: float
    receivables_risk_score: float
    inventory_risk_score: float
    related_party_risk_score: float
    regulatory_penalty_score: float
    audit_opinion_score: float
    overall_fraud_risk_score: float
    risk_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class NewsCredibilityScore:
    news_id: str
    source: str
    source_reliability: float
    is_primary_source: bool
    is_official: bool
    cross_validation_count: int
    event_type: EventType
    affected_symbols: tuple[str, ...]
    affected_theme: str | None
    sentiment_score: float
    fundamental_impact_score: float
    short_term_impact_score: float
    medium_term_impact_score: float
    confidence: float
    decay_half_life: float
    horizon_days: int
    contradiction_flags: tuple[str, ...] = ()
    rumor_risk: float = 0.0
    rationale: str = ""


@dataclass(frozen=True)
class MultiHorizonAlpha:
    symbol: str
    alpha_1d: float
    alpha_5d: float
    alpha_20d: float
    alpha_60d: float
    alpha_120d: float
    alpha_126d: float
    expected_return: float
    expected_excess_return: float
    volatility_forecast: float
    downside_risk: float
    confidence: float
    conformal_confidence: float
    prediction_interval_low: float
    prediction_interval_high: float
    rank_score: float
    regime_adjusted_score: float
    factor_contribution: dict[str, float] = field(default_factory=dict)
    evidence_contribution: dict[str, float] = field(default_factory=dict)
    risk_penalty: float = 0.0
    final_alpha_score: float = 0.0


@dataclass(frozen=True)
class FactorApplicability:
    factor_name: str
    factor_category: str
    applicable_universe: tuple[str, ...]
    applicable_sector: tuple[str, ...]
    applicable_theme: tuple[str, ...]
    applicable_market_regime: tuple[MarketRegime, ...]
    horizon_days: int
    decay_half_life: float
    rank_ic: float
    rank_icir: float
    hit_rate: float
    turnover: float
    capacity: float
    crowding_score: float
    factor_lifecycle_stage: str
    last_validated_at: str
    invalidation_condition: str


@dataclass(frozen=True)
class MarketRegimeSnapshot:
    market_regime: MarketRegime
    sector_regime: dict[str, str]
    risk_on_score: float
    risk_off_score: float
    liquidity_score: float
    breadth_score: float
    volatility_score: float
    drawdown_risk: float
    sector_rotation_score: dict[str, float]
    recommended_gross_exposure: float
    recommended_cash_weight: float
    hedge_need_score: float


@dataclass(frozen=True)
class TechnicalTimingPlan:
    symbol: str
    timing_score: float
    entry_zone: tuple[float, float] | None
    add_position_zone: tuple[float, float] | None
    reduce_zone: tuple[float, float] | None
    stop_loss_level: float | None
    take_profit_level: float | None
    invalidation_level: float | None
    max_chase_risk: float
    current_position_action: str
    rationale: str


@dataclass(frozen=True)
class PortfolioPlan:
    sleeve_weights: dict[SleeveType, float]
    target_weights: dict[str, float]
    max_single_name_weight: float
    max_sector_weight: float
    max_theme_weight: float
    cash_weight: float
    hedge_weight: float
    turnover_limit: float
    position_reason: dict[str, str] = field(default_factory=dict)
    sector_weights: dict[str, float] = field(default_factory=dict)
    theme_weights: dict[str, float] = field(default_factory=dict)
    constraint_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class HedgeDecision:
    hedge_need_score: float
    hedge_type: str
    hedge_weight: float
    reduce_exposure_amount: float
    cash_buffer_target: float
    affected_positions: tuple[str, ...]
    rationale: str
    reactivation_condition: str


@dataclass(frozen=True)
class ExecutionConstraintReport:
    symbol: str
    can_buy: bool
    can_sell: bool
    t_plus_one_blocked: bool
    limit_up_no_buy: bool
    limit_down_no_sell: bool
    suspended_no_trade: bool
    st_blocked: bool
    min_lot_size: int
    volume_participation_cap: float
    slippage_bps: float
    impact_bps: float
    feasibility_score: float
    rejection_reason: str | None = None


@dataclass(frozen=True)
class RiskGateReport:
    risk_passed: bool
    rejected_symbols: dict[str, str]
    reduced_symbols: dict[str, float]
    blocked_symbols: dict[str, str]
    risk_warnings: tuple[str, ...]
    max_allowed_position: dict[str, float]
    required_cash_buffer: float
    kill_switch_triggered: bool
    rationale: str


@dataclass(frozen=True)
class BacktestAttributionReport:
    annual_return: float
    cumulative_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    volatility: float
    hit_rate: float
    win_loss_ratio: float
    turnover: float
    transaction_cost: float
    alpha: float
    beta: float
    information_ratio: float
    rank_ic: float
    rank_icir: float
    factor_decay: dict[str, float]
    capacity: float
    tail_risk: float
    drawdown_recovery_days: int
    theme_contribution: dict[str, float] = field(default_factory=dict)
    factor_contribution: dict[str, float] = field(default_factory=dict)
    agent_contribution: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditLogRecord:
    decision_id: str
    timestamp: str
    input_data_versions: dict[str, str]
    model_version: str
    feature_version: str
    evidence_hashes: tuple[str, ...]
    risk_gate_result: str
    final_decision_reason: str


def stable_record_hash(record: EvidenceRecord) -> str:
    payload = asdict(record)
    payload["hash"] = ""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return sha256(raw.encode("utf-8")).hexdigest()
