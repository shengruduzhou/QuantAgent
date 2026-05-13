from __future__ import annotations

from quantagent.v7.schemas import HedgeDecision, MarketRegime, MarketRegimeSnapshot, PortfolioPlan


def decide_v7_hedge(
    market: MarketRegimeSnapshot,
    portfolio: PortfolioPlan,
    portfolio_drawdown: float = 0.0,
    theme_crowding_score: float = 0.0,
    model_uncertainty: float = 0.0,
) -> HedgeDecision:
    score = _hedge_score(market, portfolio_drawdown, theme_crowding_score, model_uncertainty)
    if score >= 0.75:
        hedge_type = "cash_and_exposure_reduction"
    elif score >= 0.55:
        hedge_type = "cash_buffer_and_concentration_reduction"
    elif score >= 0.35:
        hedge_type = "watch_and_reduce_short_event"
    else:
        hedge_type = "none"
    reduce_amount = max(0.0, score - 0.35) * 0.50
    cash_target = min(0.80, max(portfolio.cash_weight, 0.15 + score * 0.45))
    affected = tuple(symbol for symbol, weight in portfolio.target_weights.items() if weight >= portfolio.max_single_name_weight * 0.8)
    return HedgeDecision(
        hedge_need_score=score,
        hedge_type=hedge_type,
        hedge_weight=min(0.30, portfolio.hedge_weight + max(0.0, score - 0.50) * 0.30),
        reduce_exposure_amount=reduce_amount,
        cash_buffer_target=cash_target,
        affected_positions=affected,
        rationale=(
            f"regime={market.market_regime.value}, risk_off={market.risk_off_score:.2f}, "
            f"breadth={market.breadth_score:.2f}, drawdown={portfolio_drawdown:.2f}, crowding={theme_crowding_score:.2f}"
        ),
        reactivation_condition="risk_off_score<0.45 and breadth_score>0.50 and volatility_score<0.55",
    )


def _hedge_score(market: MarketRegimeSnapshot, drawdown: float, crowding: float, uncertainty: float) -> float:
    score = 0.30 * market.risk_off_score
    score += 0.18 * market.volatility_score
    score += 0.15 * market.drawdown_risk
    score += 0.12 * (1.0 - market.breadth_score)
    score += 0.10 * (1.0 - market.liquidity_score)
    score += 0.08 * max(0.0, drawdown)
    score += 0.04 * crowding
    score += 0.03 * uncertainty
    if market.market_regime in {MarketRegime.RISK_OFF, MarketRegime.BEAR, MarketRegime.LIQUIDITY_CRUNCH}:
        score += 0.12
    return max(0.0, min(1.0, score))
