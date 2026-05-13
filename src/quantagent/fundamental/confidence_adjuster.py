from __future__ import annotations

from quantagent.v7.scoring import clamp01, fraud_confidence_multiplier


def adjust_confidence(
    base_confidence: float,
    fraud_risk_score: float = 50.0,
    news_confidence: float = 0.5,
    data_quality: float = 1.0,
    model_uncertainty: float = 0.0,
    signal_kind: str = "fundamental",
) -> float:
    """Apply V7 confidence discounts for fraud, weak news, data quality, and uncertainty."""
    confidence = clamp01(base_confidence)
    confidence *= fraud_confidence_multiplier(fraud_risk_score, signal_kind=signal_kind)
    confidence *= 0.5 + 0.5 * clamp01(news_confidence)
    confidence *= clamp01(data_quality)
    confidence *= 1.0 - 0.5 * clamp01(model_uncertainty)
    return clamp01(confidence)
