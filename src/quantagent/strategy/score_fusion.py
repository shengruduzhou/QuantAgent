from __future__ import annotations

from dataclasses import dataclass

from quantagent.domain.schemas import ModelScores


@dataclass(frozen=True)
class FusionWeights:
    short_weight: float = 0.35
    long_weight: float = 0.30
    news_weight: float = 0.15
    llm_weight: float = 0.10
    risk_weight: float = -0.20


def clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def fuse_scores(scores: ModelScores, weights: FusionWeights | None = None) -> float:
    weights = weights or FusionWeights()
    final_score = (
        weights.short_weight * clamp_score(scores.short_score)
        + weights.long_weight * clamp_score(scores.long_score)
        + weights.news_weight * clamp_score(scores.news_score)
        + weights.llm_weight * clamp_score(scores.llm_score)
        + weights.risk_weight * clamp_score(scores.risk_score)
    )
    return clamp_score(final_score)


def confidence_adjusted_score(raw_score: float, confidence: float, risk_score: float = 0.0) -> float:
    adjusted = clamp_score(raw_score) * max(0.0, min(1.0, confidence)) - 0.2 * clamp_score(risk_score)
    return clamp_score(adjusted)
