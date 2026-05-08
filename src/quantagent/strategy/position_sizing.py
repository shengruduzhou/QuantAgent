from __future__ import annotations

from dataclasses import dataclass

from quantagent.domain.schemas import ModelScores, TradeAction


@dataclass(frozen=True)
class PositionSizingConfig:
    base_weight: float = 0.03
    high_conviction_weight: float = 0.06
    max_single_name_weight: float = 0.08
    min_confidence: float = 0.35


def target_weight(
    action: TradeAction,
    final_score: float,
    scores: ModelScores,
    config: PositionSizingConfig | None = None,
) -> float:
    config = config or PositionSizingConfig()

    if action in {TradeAction.EXIT, TradeAction.BLOCK}:
        return 0.0
    if action == TradeAction.REDUCE:
        return min(config.base_weight, config.max_single_name_weight) * 0.5
    if action == TradeAction.HOLD:
        return 0.0
    if scores.confidence < config.min_confidence:
        return 0.0

    conviction = config.high_conviction_weight if final_score >= 75.0 else config.base_weight
    risk_multiplier = max(0.2, 1.0 - scores.risk_score / 100.0)
    confidence_multiplier = max(0.0, min(1.0, scores.confidence))
    return min(conviction * risk_multiplier * confidence_multiplier, config.max_single_name_weight)
