from __future__ import annotations

import numpy as np


def fractional_kelly(expected_return: float, variance: float, fraction: float = 0.25) -> float:
    if variance <= 0:
        return 0.0
    return max(0.0, fraction * expected_return / variance)


def volatility_target_weight(
    forecast_volatility: float,
    target_volatility: float,
    max_weight: float,
) -> float:
    if forecast_volatility <= 0:
        return 0.0
    return float(np.clip(target_volatility / forecast_volatility, 0.0, max_weight))


def confidence_adjusted_alpha(alpha: float, confidence: float) -> float:
    return alpha * max(0.0, min(1.0, confidence))
