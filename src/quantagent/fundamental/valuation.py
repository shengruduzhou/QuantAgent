from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DCFInputs:
    fcff: float
    growth_rate: float
    terminal_growth_rate: float
    wacc: float
    years: int
    net_debt: float
    shares_outstanding: float


def dcf_equity_value(inputs: DCFInputs) -> float:
    if inputs.wacc <= inputs.terminal_growth_rate:
        raise ValueError("WACC must exceed terminal growth rate")
    cash_flows = [
        inputs.fcff * (1.0 + inputs.growth_rate) ** year
        for year in range(1, inputs.years + 1)
    ]
    discounted = [
        cash_flow / (1.0 + inputs.wacc) ** year
        for year, cash_flow in enumerate(cash_flows, start=1)
    ]
    terminal_fcff = cash_flows[-1] * (1.0 + inputs.terminal_growth_rate)
    terminal_value = terminal_fcff / (inputs.wacc - inputs.terminal_growth_rate)
    enterprise_value = sum(discounted) + terminal_value / (1.0 + inputs.wacc) ** inputs.years
    return enterprise_value - inputs.net_debt


def dcf_intrinsic_value_per_share(inputs: DCFInputs) -> float:
    if inputs.shares_outstanding <= 0:
        raise ValueError("Shares outstanding must be positive")
    return dcf_equity_value(inputs) / inputs.shares_outstanding


def margin_of_safety(intrinsic_value: float, market_price: float) -> float:
    if market_price <= 0:
        raise ValueError("Market price must be positive")
    return intrinsic_value / market_price - 1.0


def reverse_dcf_implied_growth(
    market_cap: float,
    fcff: float,
    wacc: float,
    terminal_growth_rate: float,
    years: int = 5,
    net_debt: float = 0.0,
    lower: float = -0.5,
    upper: float = 0.8,
    tolerance: float = 1e-5,
    max_iter: int = 100,
) -> float:
    """Infer constant FCFF growth implied by current enterprise value."""
    target_equity = market_cap
    for _ in range(max_iter):
        mid = (lower + upper) / 2.0
        value = dcf_equity_value(
            DCFInputs(
                fcff=fcff,
                growth_rate=mid,
                terminal_growth_rate=terminal_growth_rate,
                wacc=wacc,
                years=years,
                net_debt=net_debt,
                shares_outstanding=1.0,
            )
        )
        if abs(value - target_equity) <= tolerance * max(abs(target_equity), 1.0):
            return mid
        if value < target_equity:
            lower = mid
        else:
            upper = mid
    return (lower + upper) / 2.0


def relative_valuation_zscore(
    ev_ebitda_z: float,
    ps_z: float,
    pe_z: float,
    fcf_yield_z: float,
) -> float:
    return float(np.mean([-ev_ebitda_z, -ps_z, -pe_z, fcf_yield_z]))
