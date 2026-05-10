from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantagent.fundamental.valuation import DCFInputs, dcf_intrinsic_value_per_share, margin_of_safety, reverse_dcf_implied_growth


@dataclass(frozen=True)
class TargetPriceEstimate:
    symbol: str
    bear_price: float
    base_price: float
    bull_price: float
    current_price: float
    expected_upside: float
    implied_growth: float
    margin_of_safety: float
    confidence: float
    risk_flags: tuple[str, ...]


def dcf_target_price_range(
    base_inputs: DCFInputs,
    bear_growth_delta: float = -0.03,
    bull_growth_delta: float = 0.03,
    wacc_spread: float = 0.01,
) -> tuple[float, float, float]:
    bear = dcf_intrinsic_value_per_share(
        DCFInputs(
            fcff=base_inputs.fcff,
            growth_rate=base_inputs.growth_rate + bear_growth_delta,
            terminal_growth_rate=base_inputs.terminal_growth_rate,
            wacc=base_inputs.wacc + wacc_spread,
            years=base_inputs.years,
            net_debt=base_inputs.net_debt,
            shares_outstanding=base_inputs.shares_outstanding,
        )
    )
    base = dcf_intrinsic_value_per_share(base_inputs)
    bull = dcf_intrinsic_value_per_share(
        DCFInputs(
            fcff=base_inputs.fcff,
            growth_rate=base_inputs.growth_rate + bull_growth_delta,
            terminal_growth_rate=base_inputs.terminal_growth_rate,
            wacc=max(base_inputs.wacc - wacc_spread, base_inputs.terminal_growth_rate + 0.005),
            years=base_inputs.years,
            net_debt=base_inputs.net_debt,
            shares_outstanding=base_inputs.shares_outstanding,
        )
    )
    return float(bear), float(base), float(bull)


def relative_valuation_target_price(
    peer_multiple: float,
    company_metric_per_share: float,
    discount: float = 0.0,
) -> float:
    return float(peer_multiple * company_metric_per_share * (1.0 - discount))


def sum_of_the_parts_placeholder(segment_values: dict[str, float], shares_outstanding: float) -> float:
    if shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be positive")
    return float(sum(segment_values.values()) / shares_outstanding)


def scenario_analysis(
    base_inputs: DCFInputs,
    current_price: float,
    relative_price: float | None = None,
) -> dict[str, float]:
    bear, base, bull = dcf_target_price_range(base_inputs)
    if relative_price is not None and np.isfinite(relative_price):
        base = 0.7 * base + 0.3 * relative_price
        bear = min(bear, relative_price * 0.85)
        bull = max(bull, relative_price * 1.15)
    return {
        "bear": float(bear),
        "base": float(base),
        "bull": float(bull),
        "current": float(current_price),
    }


def final_target_price_band(
    symbol: str,
    current_price: float,
    dcf_inputs: DCFInputs,
    relative_price: float | None = None,
    fraud_risk: float = 0.0,
    quality_score: float = 50.0,
) -> TargetPriceEstimate:
    scenarios = scenario_analysis(dcf_inputs, current_price, relative_price)
    base_price = scenarios["base"]
    implied_growth = reverse_dcf_implied_growth(
        market_cap=current_price * dcf_inputs.shares_outstanding,
        fcff=dcf_inputs.fcff,
        wacc=dcf_inputs.wacc,
        terminal_growth_rate=dcf_inputs.terminal_growth_rate,
        years=dcf_inputs.years,
        net_debt=dcf_inputs.net_debt,
    )
    mos = margin_of_safety(base_price, current_price)
    risk_flags: list[str] = []
    if fraud_risk >= 0.7:
        risk_flags.append("high_fraud_risk")
    if quality_score < 40:
        risk_flags.append("low_quality")
    if dcf_inputs.wacc <= dcf_inputs.terminal_growth_rate + 0.01:
        risk_flags.append("thin_wacc_spread")
    confidence = float(np.clip(0.6 + quality_score / 250.0 - fraud_risk * 0.4 - len(risk_flags) * 0.08, 0.0, 1.0))
    return TargetPriceEstimate(
        symbol=symbol,
        bear_price=float(scenarios["bear"]),
        base_price=float(base_price),
        bull_price=float(scenarios["bull"]),
        current_price=float(current_price),
        expected_upside=float(base_price / current_price - 1.0) if current_price > 0 else np.nan,
        implied_growth=float(implied_growth),
        margin_of_safety=float(mos),
        confidence=confidence,
        risk_flags=tuple(risk_flags),
    )

