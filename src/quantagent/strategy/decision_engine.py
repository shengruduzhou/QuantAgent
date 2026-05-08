from __future__ import annotations

from quantagent.domain.schemas import ModelScores, RiskLimits, SignalDecision
from quantagent.strategy.position_sizing import PositionSizingConfig, target_weight
from quantagent.strategy.risk_gate import risk_gate
from quantagent.strategy.score_fusion import FusionWeights, fuse_scores


def decide_trade(
    scores: ModelScores,
    fusion_weights: FusionWeights | None = None,
    risk_limits: RiskLimits | None = None,
    sizing_config: PositionSizingConfig | None = None,
) -> SignalDecision:
    final_score = fuse_scores(scores, fusion_weights)
    action, reason = risk_gate(scores, risk_limits)
    weight = target_weight(action, final_score, scores, sizing_config)
    return SignalDecision(
        ticker=scores.ticker,
        action=action,
        final_score=final_score,
        target_weight=weight,
        reason=reason,
    )
