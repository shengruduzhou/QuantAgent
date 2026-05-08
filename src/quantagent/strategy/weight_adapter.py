from __future__ import annotations

from quantagent.domain.schemas import TargetWeight


def horizon_alpha(short_weight: float, horizon_days: int) -> float:
    if horizon_days <= 10:
        return short_weight
    if horizon_days <= 60:
        return 0.5
    return 1.0 - short_weight


def short_signal_to_weight(
    short_signal: float,
    volatility: float,
    vol_target: float = 0.012,
    max_abs_weight: float = 0.04,
) -> float:
    if volatility <= 0:
        return 0.0
    raw = short_signal / volatility * vol_target
    return max(-max_abs_weight, min(max_abs_weight, raw))


def long_score_to_weight(
    long_score: float,
    margin_of_safety: float,
    quality_gate: float,
    max_weight: float = 0.08,
) -> float:
    normalized = max(0.0, min(1.0, long_score / 100.0))
    raw = normalized * max(0.0, margin_of_safety) * max(0.0, min(1.0, quality_gate))
    return max(0.0, min(max_weight, raw))


def combine_short_long_weights(
    symbol: str,
    short_weight: float,
    long_weight: float,
    horizon_days: int,
    confidence: float,
    max_abs_weight: float = 0.10,
) -> TargetWeight:
    alpha = horizon_alpha(0.8, horizon_days)
    combined = alpha * short_weight + (1.0 - alpha) * long_weight
    combined = max(-max_abs_weight, min(max_abs_weight, combined)) * max(0.0, min(1.0, confidence))
    return TargetWeight(
        symbol=symbol,
        target_weight=combined,
        horizon_days=horizon_days,
        confidence=confidence,
        source="weight_adapter",
        reason="combined short-horizon and long-horizon raw weights",
    )
