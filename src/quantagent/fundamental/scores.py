from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PiotroskiInputs:
    net_income: float
    operating_cash_flow: float
    roa: float
    roa_prev: float
    leverage: float
    leverage_prev: float
    current_ratio: float
    current_ratio_prev: float
    shares_outstanding: float
    shares_outstanding_prev: float
    gross_margin: float
    gross_margin_prev: float
    asset_turnover: float
    asset_turnover_prev: float


def piotroski_f_score(inputs: PiotroskiInputs) -> int:
    """9-point Piotroski 2000 fundamental quality score."""
    points = 0
    points += int(inputs.net_income > 0)
    points += int(inputs.operating_cash_flow > 0)
    points += int(inputs.roa > inputs.roa_prev)
    points += int(inputs.operating_cash_flow > inputs.net_income)
    points += int(inputs.leverage < inputs.leverage_prev)
    points += int(inputs.current_ratio > inputs.current_ratio_prev)
    points += int(inputs.shares_outstanding <= inputs.shares_outstanding_prev)
    points += int(inputs.gross_margin > inputs.gross_margin_prev)
    points += int(inputs.asset_turnover > inputs.asset_turnover_prev)
    return int(points)


@dataclass(frozen=True)
class AltmanInputs:
    working_capital: float
    retained_earnings: float
    ebit: float
    market_cap: float
    total_liabilities: float
    sales: float
    total_assets: float


def altman_z_score(inputs: AltmanInputs) -> float:
    """Altman 1968 original Z (manufacturing). Below 1.81 = distress, above 2.99 = safe."""
    if inputs.total_assets <= 0 or inputs.total_liabilities <= 0:
        return float("nan")
    a = inputs.working_capital / inputs.total_assets
    b = inputs.retained_earnings / inputs.total_assets
    c = inputs.ebit / inputs.total_assets
    d = inputs.market_cap / inputs.total_liabilities
    e = inputs.sales / inputs.total_assets
    return float(1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e)


@dataclass(frozen=True)
class BeneishInputs:
    receivables_curr: float
    receivables_prev: float
    sales_curr: float
    sales_prev: float
    cogs_curr: float
    cogs_prev: float
    current_assets_curr: float
    current_assets_prev: float
    ppe_curr: float
    ppe_prev: float
    total_assets_curr: float
    total_assets_prev: float
    depreciation_curr: float
    depreciation_prev: float
    sga_curr: float
    sga_prev: float
    leverage_curr: float
    leverage_prev: float
    net_income_curr: float
    operating_cash_flow_curr: float


def beneish_m_score(inputs: BeneishInputs) -> float:
    """Beneish 1999 8-variable M-score. M > -1.78 flags potential earnings manipulation."""
    s_curr, s_prev = inputs.sales_curr, inputs.sales_prev
    if s_curr <= 0 or s_prev <= 0 or inputs.total_assets_curr <= 0 or inputs.total_assets_prev <= 0:
        return float("nan")
    dsri = (inputs.receivables_curr / s_curr) / (inputs.receivables_prev / s_prev)
    gm_curr = (s_curr - inputs.cogs_curr) / s_curr
    gm_prev = (s_prev - inputs.cogs_prev) / s_prev
    gmi = gm_prev / gm_curr if gm_curr != 0 else float("nan")
    aqi_curr = 1.0 - (inputs.current_assets_curr + inputs.ppe_curr) / inputs.total_assets_curr
    aqi_prev = 1.0 - (inputs.current_assets_prev + inputs.ppe_prev) / inputs.total_assets_prev
    aqi = aqi_curr / aqi_prev if aqi_prev != 0 else float("nan")
    sgi = s_curr / s_prev
    depi_curr = inputs.depreciation_curr / (inputs.depreciation_curr + inputs.ppe_curr)
    depi_prev = inputs.depreciation_prev / (inputs.depreciation_prev + inputs.ppe_prev)
    depi = depi_prev / depi_curr if depi_curr != 0 else float("nan")
    sgai = (inputs.sga_curr / s_curr) / (inputs.sga_prev / s_prev)
    lvgi = inputs.leverage_curr / inputs.leverage_prev if inputs.leverage_prev != 0 else float("nan")
    tata = (inputs.net_income_curr - inputs.operating_cash_flow_curr) / inputs.total_assets_curr
    return float(
        -4.84
        + 0.92 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )


def piotroski_to_score_100(f_score: int) -> float:
    """Map 0-9 F-score to 0-100 conformity with quality_score domain."""
    return float(100.0 * max(0, min(9, f_score)) / 9.0)


def altman_to_distress_percentile(z: float) -> float:
    """Map Altman Z to 0-100 distress percentile (higher = more distressed)."""
    if z != z:
        return 50.0
    if z >= 2.99:
        return 5.0
    if z <= 1.81:
        return 95.0
    return float(95.0 - (z - 1.81) / (2.99 - 1.81) * 90.0)


def beneish_to_fraud_percentile(m: float) -> float:
    """Map Beneish M to 0-100 fraud-risk percentile."""
    if m != m:
        return 50.0
    if m >= -1.78:
        return float(min(99.0, 70.0 + (m + 1.78) * 20.0))
    return float(max(1.0, 50.0 + (m + 1.78) * 20.0))
