from __future__ import annotations

from quantagent.domain.schemas import ModelScores, RiskLimits, TradeAction
from quantagent.portfolio.position_state import StopDecision


def risk_gate(scores: ModelScores, limits: RiskLimits | None = None) -> tuple[TradeAction, str]:
    limits = limits or RiskLimits()

    if scores.long_score < limits.block_long_score_below:
        return TradeAction.EXIT, "long score is below the long-term holding threshold"

    if scores.risk_score >= limits.force_reduce_risk_score:
        return TradeAction.REDUCE, "risk score breached force-reduction threshold"

    if (
        scores.long_score >= limits.allow_buy_min_long_score
        and scores.short_score >= limits.allow_buy_min_short_score
        and scores.risk_score <= limits.max_risk_score_for_buy
    ):
        return TradeAction.BUY, "long, short, and risk gates allow new exposure"

    return TradeAction.HOLD, "signal is not strong enough for new exposure"


def stop_decision_to_action(decision: StopDecision) -> tuple[TradeAction, str]:
    if decision.blocked_exit:
        return TradeAction.BLOCK, decision.reason
    if decision.should_exit:
        return TradeAction.EXIT, decision.reason
    return TradeAction.HOLD, decision.reason
