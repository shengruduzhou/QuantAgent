from __future__ import annotations

import numpy as np


def quality_score(
    roic: float,
    wacc: float,
    gross_margin_stability: float,
    fcf_conversion: float,
    net_debt_to_ebitda: float,
    interest_coverage: float,
) -> float:
    """Return 0-100 quality score from normalized financial quality inputs."""
    roic_spread = _sigmoid((roic - wacc) * 20.0)
    margin = max(0.0, min(1.0, gross_margin_stability))
    fcf = max(0.0, min(1.0, fcf_conversion))
    leverage = 1.0 - _sigmoid(net_debt_to_ebitda - 3.0)
    coverage = _sigmoid((interest_coverage - 3.0) / 3.0)
    score = 0.30 * roic_spread + 0.20 * margin + 0.20 * fcf + 0.15 * leverage + 0.15 * coverage
    return float(100.0 * max(0.0, min(1.0, score)))


def fraud_risk_score(
    beneish_m_score_percentile: float,
    accruals_percentile: float,
    revenue_ar_gap_percentile: float,
    inventory_growth_gap_percentile: float,
) -> float:
    score = (
        0.35 * beneish_m_score_percentile
        + 0.25 * accruals_percentile
        + 0.20 * revenue_ar_gap_percentile
        + 0.20 * inventory_growth_gap_percentile
    )
    return float(100.0 * max(0.0, min(1.0, score)))


def long_horizon_score(
    valuation: float,
    quality: float,
    growth: float,
    policy: float,
    ownership: float,
    medium_term_technical: float,
    risk: float,
) -> float:
    score = (
        0.25 * valuation
        + 0.20 * quality
        + 0.15 * growth
        + 0.15 * policy
        + 0.10 * ownership
        + 0.10 * medium_term_technical
        - 0.20 * risk
    )
    return float(max(0.0, min(100.0, score)))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))
