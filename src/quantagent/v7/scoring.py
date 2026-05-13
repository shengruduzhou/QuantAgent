from __future__ import annotations

from quantagent.v7.schemas import ThemeLifecycleStage, UniverseBucket


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def policy_authority_score(source_level: str) -> float:
    table = {
        "central": 0.95,
        "state_council": 0.95,
        "ministry": 0.88,
        "provincial": 0.72,
        "municipal": 0.65,
        "industry_association": 0.55,
        "industrial_park": 0.45,
        "media_interpretation": 0.35,
    }
    return table.get(source_level, 0.30)


def theme_strength_score(
    policy_strength: float,
    market_strength: float,
    industry_fundamental_strength: float,
    capital_flow_strength: float,
    news_sentiment_strength: float,
    opposing_evidence_penalty: float = 0.0,
    bubble_risk: float = 0.0,
) -> float:
    score = (
        0.28 * clamp01(policy_strength)
        + 0.20 * clamp01(market_strength)
        + 0.24 * clamp01(industry_fundamental_strength)
        + 0.18 * clamp01(capital_flow_strength)
        + 0.10 * clamp01(news_sentiment_strength)
    )
    score -= 0.20 * clamp01(opposing_evidence_penalty)
    score -= 0.10 * clamp01(bubble_risk)
    return clamp01(score)


def classify_theme_lifecycle(
    policy_strength: float,
    market_strength: float,
    fundamental_strength: float,
    capital_flow_strength: float,
    bubble_risk: float,
    crowding_score: float,
    invalidation_score: float = 0.0,
    trend_decay_score: float = 0.0,
) -> ThemeLifecycleStage:
    policy = clamp01(policy_strength)
    market = clamp01(market_strength)
    fundamental = clamp01(fundamental_strength)
    flow = clamp01(capital_flow_strength)
    bubble = clamp01(bubble_risk)
    crowding = clamp01(crowding_score)
    invalidation = clamp01(invalidation_score)
    decay = clamp01(trend_decay_score)
    if invalidation >= 0.80:
        return ThemeLifecycleStage.INVALIDATED
    if decay >= 0.70 and market < 0.45:
        return ThemeLifecycleStage.DECAY
    if bubble >= 0.75 and crowding >= 0.65 and market >= 0.60:
        return ThemeLifecycleStage.VALUATION_BUBBLE
    if crowding >= 0.70 and fundamental >= 0.50 and market < 0.65:
        return ThemeLifecycleStage.DIVERGENCE
    if fundamental >= 0.75 and policy >= 0.45:
        return ThemeLifecycleStage.EARNINGS_REALIZATION
    if fundamental >= 0.55:
        return ThemeLifecycleStage.FUNDAMENTAL_VALIDATION
    if flow >= 0.60 and market >= 0.55:
        return ThemeLifecycleStage.CAPITAL_INFLOW
    if policy >= 0.60 and market >= 0.35:
        return ThemeLifecycleStage.NARRATIVE_FORMATION
    return ThemeLifecycleStage.POLICY_SEED


def fraud_confidence_multiplier(fraud_risk_score: float, signal_kind: str = "fundamental") -> float:
    score = clamp01(fraud_risk_score / 100.0) * 100.0
    if score > 80.0:
        if signal_kind == "news":
            return 0.30
        if signal_kind == "fundamental":
            return 0.20
        return 0.25
    if score >= 60.0:
        return 0.50
    return max(0.70, 1.0 - score / 300.0)


def classify_universe_bucket(
    exposure_score: float,
    fundamental_score: float,
    fraud_risk_score: float,
    liquidity_score: float,
    source_confidence: float,
    evidence_count: int,
    valuation_score: float = 50.0,
) -> UniverseBucket:
    if fraud_risk_score > 80.0 or liquidity_score < 25.0 or source_confidence < 0.25:
        return UniverseBucket.EXCLUSION
    if exposure_score >= 80.0 and fundamental_score >= 70.0 and evidence_count >= 3 and valuation_score >= 35.0:
        return UniverseBucket.CORE_BENEFICIARY
    if exposure_score >= 65.0 and evidence_count >= 2 and fundamental_score >= 50.0:
        return UniverseBucket.STRONG_CORRELATION
    if exposure_score >= 45.0 and liquidity_score >= 45.0:
        return UniverseBucket.OPTIONAL_SATELLITE
    return UniverseBucket.WATCHLIST


def news_confidence_score(
    source_reliability: float,
    is_primary_source: bool,
    is_official: bool,
    cross_validation_count: int,
    contradiction_count: int,
    rumor_risk: float,
) -> float:
    score = 0.55 * clamp01(source_reliability)
    score += 0.15 if is_primary_source else 0.0
    score += 0.15 if is_official else 0.0
    score += min(0.15, 0.04 * max(cross_validation_count, 0))
    score -= min(0.25, 0.08 * max(contradiction_count, 0))
    score -= 0.25 * clamp01(rumor_risk)
    return clamp01(score)


def portfolio_exposure_cap(
    risk_off_score: float,
    liquidity_score: float,
    drawdown_risk: float,
    hedge_need_score: float,
    model_uncertainty: float,
) -> tuple[float, float]:
    gross_cap = 1.0
    gross_cap -= 0.35 * clamp01(risk_off_score)
    gross_cap -= 0.20 * (1.0 - clamp01(liquidity_score))
    gross_cap -= 0.20 * clamp01(drawdown_risk)
    gross_cap -= 0.15 * clamp01(hedge_need_score)
    gross_cap -= 0.10 * clamp01(model_uncertainty)
    gross_cap = clamp01(gross_cap)
    cash_buffer = clamp01(1.0 - gross_cap + 0.10 * clamp01(hedge_need_score))
    return gross_cap, cash_buffer


def execution_feasibility_score(
    is_suspended: bool,
    is_limit_up: bool,
    is_limit_down: bool,
    liquidity_score: float,
    participation_rate: float,
    lot_size_ok: bool = True,
) -> float:
    if is_suspended:
        return 0.0
    score = clamp01(liquidity_score / 100.0)
    if is_limit_up:
        score -= 0.40
    if is_limit_down:
        score -= 0.50
    if not lot_size_ok:
        score -= 0.20
    if participation_rate > 0.10:
        score -= min(0.30, (participation_rate - 0.10) * 2.0)
    return clamp01(score)
