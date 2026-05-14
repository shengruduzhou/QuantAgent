"""Economic-principles analyzer for industries and themes.

Applies textbook economics (business-cycle stage, supply/demand, price elasticity,
capital intensity, Cobb-Douglas-style production efficiency, credit cycle, monetary
stance, FX and commodity cost pressure) to a fundamentals + theme panel. Output
feeds the long-horizon factor library and the multi-horizon alpha model so
slow-moving macro tailwinds get explicit weight in the 60-126 day horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EconomicAnalyzerConfig:
    capacity_utilization_target: float = 0.80
    capex_growth_warning: float = 0.25
    inventory_days_warning: float = 90.0
    margin_growth_window_quarters: int = 4
    credit_impulse_window_quarters: int = 6
    monetary_tailwind_lpr_anchor: float = 3.45
    monetary_tailwind_rrr_anchor: float = 7.50
    fx_neutral_zone: float = 0.02
    commodity_neutral_zone: float = 0.05
    inflation_floor: float = 0.005
    inflation_ceiling: float = 0.035


@dataclass(frozen=True)
class IndustryEconomicSnapshot:
    industry: str
    industry_cycle_stage: str
    supply_demand_balance: float
    pricing_power: float
    capacity_utilization: float
    inventory_days_zscore: float
    capex_intensity_trend: float
    credit_impulse_alignment: float
    monetary_tailwind: float
    fx_pressure: float
    commodity_cost_pressure: float
    policy_support_strength: float
    expected_industry_revenue_growth_yoy: float
    expected_horizon_days: int
    economic_thesis: str
    rationale: str
    cobb_douglas_efficiency: float = 0.0


@dataclass(frozen=True)
class CompanyEconomicSnapshot:
    symbol: str
    industry: str
    sector: str
    industry_cycle_stage: str
    supply_demand_balance: float
    pricing_power: float
    capacity_utilization: float
    inventory_days_zscore: float
    capex_intensity_trend: float
    credit_impulse_alignment: float
    monetary_tailwind: float
    fx_pressure: float
    commodity_cost_pressure: float
    policy_support_strength: float
    expected_industry_revenue_growth_yoy: float
    expected_horizon_days: int
    cobb_douglas_efficiency: float
    company_pricing_power: float
    company_capital_efficiency: float
    company_demand_visibility: float
    rationale: str


@dataclass(frozen=True)
class MacroSnapshot:
    as_of_date: str
    business_cycle_stage: str
    monetary_stance: str
    fiscal_stance: str
    credit_impulse: float
    cny_strength: float
    commodity_index_zscore: float
    inflation_pressure: float
    risk_appetite: float
    rationale: str
    fields: dict[str, float] = field(default_factory=dict)


def analyze_macro(macro_indicators: pd.DataFrame | None, as_of_date: str, config: EconomicAnalyzerConfig | None = None) -> MacroSnapshot:
    config = config or EconomicAnalyzerConfig()
    if macro_indicators is None or macro_indicators.empty:
        return _neutral_macro(as_of_date)
    latest = (
        macro_indicators.assign(as_of_date=pd.to_datetime(macro_indicators.get("as_of_date"), errors="coerce"))
        .sort_values("as_of_date")
        .iloc[-1]
        .to_dict()
    )
    lpr = _safe_float(latest.get("lpr"), default=config.monetary_tailwind_lpr_anchor)
    rrr = _safe_float(latest.get("rrr"), default=config.monetary_tailwind_rrr_anchor)
    cny = _safe_float(latest.get("cny_usd"), default=7.10)
    commodity_z = _safe_float(latest.get("commodity_index_zscore"))
    inflation = _safe_float(latest.get("cpi_yoy"), default=0.015)
    credit_impulse = _safe_float(latest.get("credit_impulse"))
    pmi = _safe_float(latest.get("pmi"), default=50.0)

    monetary_stance = _monetary_stance(lpr, rrr, config)
    fiscal_stance = _fiscal_stance(latest)
    business_cycle = _business_cycle(pmi, credit_impulse, inflation, config)
    risk_appetite = float(np.clip(0.50 + 0.5 * np.tanh(credit_impulse * 8.0) - max(0.0, inflation - config.inflation_ceiling) * 8.0, 0.0, 1.0))

    rationale = (
        f"as_of={as_of_date}; cycle={business_cycle}; monetary={monetary_stance}; "
        f"fiscal={fiscal_stance}; cny={cny:.2f}; commodity_z={commodity_z:.2f}; "
        f"inflation_yoy={inflation:.2%}; credit_impulse={credit_impulse:.3f}; pmi={pmi:.1f}"
    )
    return MacroSnapshot(
        as_of_date=as_of_date,
        business_cycle_stage=business_cycle,
        monetary_stance=monetary_stance,
        fiscal_stance=fiscal_stance,
        credit_impulse=credit_impulse,
        cny_strength=float(7.10 - cny),
        commodity_index_zscore=commodity_z,
        inflation_pressure=inflation,
        risk_appetite=risk_appetite,
        rationale=rationale,
        fields={
            "lpr": lpr,
            "rrr": rrr,
            "cny_usd": cny,
            "pmi": pmi,
        },
    )


def analyze_industries(
    fundamentals: pd.DataFrame,
    theme_profiles: list,
    macro_snapshot: MacroSnapshot,
    config: EconomicAnalyzerConfig | None = None,
) -> list[IndustryEconomicSnapshot]:
    config = config or EconomicAnalyzerConfig()
    if fundamentals is None or fundamentals.empty or "industry" not in fundamentals.columns:
        return []
    snapshots: list[IndustryEconomicSnapshot] = []
    theme_strength_by_industry = _theme_strength_by_industry(theme_profiles)
    grouped = fundamentals.copy()
    grouped["report_date"] = pd.to_datetime(grouped.get("report_date"), errors="coerce")
    grouped = grouped.sort_values(["industry", "symbol", "report_date"])
    for industry, group in grouped.groupby("industry"):
        latest = group.groupby("symbol", sort=False).tail(1)
        capacity_utilization = _aggregate_capacity_utilization(latest)
        inventory_days_z = _aggregate_inventory_days_z(latest)
        capex_trend = _capex_intensity_trend(group)
        pricing_power = _industry_pricing_power(group, config)
        cobb_efficiency = _cobb_douglas_efficiency(latest)
        supply_demand = float(np.clip(0.10 + 0.5 * (capacity_utilization - config.capacity_utilization_target) - 0.4 * inventory_days_z, -1.0, 1.0))
        credit_alignment = float(np.clip(macro_snapshot.credit_impulse * 4.0, -1.0, 1.0))
        monetary_tailwind = _industry_monetary_sensitivity(industry, macro_snapshot)
        fx_pressure = _industry_fx_pressure(industry, macro_snapshot)
        commodity_pressure = _industry_commodity_pressure(industry, macro_snapshot)
        policy_strength = float(theme_strength_by_industry.get(industry, 0.0))
        expected_growth = _expected_industry_growth(latest, pricing_power, supply_demand)
        cycle_stage = _industry_cycle_stage(supply_demand, capex_trend, inventory_days_z, macro_snapshot)
        thesis = (
            f"{industry}: stage={cycle_stage}, sd_balance={supply_demand:+.2f}, "
            f"capacity={capacity_utilization:.2f}, capex_trend={capex_trend:+.2f}, "
            f"pricing_power={pricing_power:.2f}, policy={policy_strength:.2f}, "
            f"monetary={monetary_tailwind:+.2f}"
        )
        snapshots.append(
            IndustryEconomicSnapshot(
                industry=str(industry),
                industry_cycle_stage=cycle_stage,
                supply_demand_balance=supply_demand,
                pricing_power=pricing_power,
                capacity_utilization=capacity_utilization,
                inventory_days_zscore=inventory_days_z,
                capex_intensity_trend=capex_trend,
                credit_impulse_alignment=credit_alignment,
                monetary_tailwind=monetary_tailwind,
                fx_pressure=fx_pressure,
                commodity_cost_pressure=commodity_pressure,
                policy_support_strength=policy_strength,
                expected_industry_revenue_growth_yoy=expected_growth,
                expected_horizon_days=120,
                cobb_douglas_efficiency=cobb_efficiency,
                economic_thesis=thesis,
                rationale=thesis,
            )
        )
    return snapshots


def industry_snapshots_to_company_frame(
    fundamentals: pd.DataFrame,
    snapshots: list[IndustryEconomicSnapshot],
    as_of_date: str,
) -> pd.DataFrame:
    """Return a per-symbol frame keyed by `symbol` carrying the industry economics."""
    if fundamentals is None or fundamentals.empty or "industry" not in fundamentals.columns:
        return pd.DataFrame()
    snapshot_by_industry = {snapshot.industry: snapshot for snapshot in snapshots}
    latest = (
        fundamentals.assign(report_date=pd.to_datetime(fundamentals.get("report_date"), errors="coerce"))
        .sort_values(["symbol", "report_date"])
        .groupby("symbol", sort=False)
        .tail(1)
    )
    rows = []
    for _, row in latest.iterrows():
        symbol = str(row["symbol"])
        industry = str(row.get("industry") or row.get("sector") or "")
        snapshot = snapshot_by_industry.get(industry)
        if snapshot is None:
            continue
        rows.append(
            {
                "symbol": symbol,
                "as_of_date": as_of_date,
                "industry": industry,
                "industry_cycle_stage": snapshot.industry_cycle_stage,
                "supply_demand_balance": snapshot.supply_demand_balance,
                "pricing_power": snapshot.pricing_power,
                "capacity_utilization": snapshot.capacity_utilization,
                "inventory_days_zscore": snapshot.inventory_days_zscore,
                "capex_intensity_trend": snapshot.capex_intensity_trend,
                "credit_impulse_alignment": snapshot.credit_impulse_alignment,
                "monetary_tailwind": snapshot.monetary_tailwind,
                "fx_pressure": snapshot.fx_pressure,
                "commodity_cost_pressure": snapshot.commodity_cost_pressure,
                "policy_support_strength": snapshot.policy_support_strength,
                "expected_industry_revenue_growth_yoy": snapshot.expected_industry_revenue_growth_yoy,
                "expected_horizon_days": snapshot.expected_horizon_days,
                "cobb_douglas_efficiency": snapshot.cobb_douglas_efficiency,
                "company_pricing_power": _company_pricing_power(row),
                "company_capital_efficiency": _company_capital_efficiency(row),
                "company_demand_visibility": _company_demand_visibility(row),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _aggregate_capacity_utilization(latest: pd.DataFrame) -> float:
    if "capacity_utilization" in latest.columns:
        series = pd.to_numeric(latest["capacity_utilization"], errors="coerce").dropna()
        if not series.empty:
            return float(np.clip(series.mean(), 0.0, 1.0))
    if "capacity_release_score" in latest.columns:
        return float(np.clip(pd.to_numeric(latest["capacity_release_score"], errors="coerce").dropna().mean() / 100.0, 0.0, 1.0))
    return 0.65


def _aggregate_inventory_days_z(latest: pd.DataFrame) -> float:
    if "inventory_days" in latest.columns:
        series = pd.to_numeric(latest["inventory_days"], errors="coerce").dropna()
        if len(series) >= 3:
            mean = series.mean()
            std = series.std(ddof=0)
            if std > 0:
                return float(np.clip((series.tail(1).iloc[0] - mean) / std, -3.0, 3.0))
    if {"inventory", "cogs"}.issubset(latest.columns):
        inventory = pd.to_numeric(latest["inventory"], errors="coerce").dropna()
        cogs = pd.to_numeric(latest["cogs"], errors="coerce").dropna()
        if not inventory.empty and not cogs.empty and (cogs > 0).any():
            days = (inventory / cogs * 90.0).dropna()
            if len(days) >= 3:
                mean = days.mean()
                std = days.std(ddof=0)
                if std > 0:
                    return float(np.clip((days.iloc[-1] - mean) / std, -3.0, 3.0))
    return 0.0


def _capex_intensity_trend(group: pd.DataFrame) -> float:
    if "capex" not in group.columns or "revenue" not in group.columns:
        return 0.0
    series = (pd.to_numeric(group["capex"].abs(), errors="coerce") / pd.to_numeric(group["revenue"], errors="coerce")).dropna()
    if len(series) < 4:
        return 0.0
    recent = series.tail(2).mean()
    prior = series.head(max(1, len(series) - 2)).mean()
    if prior <= 0:
        return 0.0
    return float(np.clip((recent - prior) / max(abs(prior), 0.01), -1.0, 1.0))


def _industry_pricing_power(group: pd.DataFrame, config: EconomicAnalyzerConfig) -> float:
    if "gross_margin" not in group.columns or "revenue_growth" not in group.columns:
        return 0.50
    margin = pd.to_numeric(group["gross_margin"], errors="coerce").dropna()
    growth = pd.to_numeric(group["revenue_growth"], errors="coerce").dropna()
    if margin.empty:
        return 0.50
    margin_mean = float(margin.tail(config.margin_growth_window_quarters).mean())
    margin_trend = float(margin.tail(config.margin_growth_window_quarters).diff().mean()) if len(margin) >= 2 else 0.0
    growth_mean = float(growth.tail(config.margin_growth_window_quarters).mean()) if not growth.empty else 0.05
    raw = 0.40 * margin_mean + 0.30 * (margin_trend * 5.0) + 0.30 * np.clip(growth_mean, -0.2, 0.4)
    return float(np.clip(raw, 0.0, 1.0))


def _cobb_douglas_efficiency(latest: pd.DataFrame) -> float:
    if not {"total_assets", "revenue"}.issubset(latest.columns):
        return 0.50
    capital = pd.to_numeric(latest["total_assets"], errors="coerce")
    revenue = pd.to_numeric(latest["revenue"], errors="coerce")
    if (capital <= 0).all() or (revenue <= 0).all():
        return 0.50
    log_capital = np.log(capital.where(capital > 0).dropna())
    log_revenue = np.log(revenue.where(revenue > 0).dropna()).reindex(log_capital.index).dropna()
    log_capital = log_capital.reindex(log_revenue.index).dropna()
    if len(log_capital) < 3:
        return 0.50
    slope = np.polyfit(log_capital, log_revenue, 1)[0]
    return float(np.clip(0.50 + (slope - 1.0) * 0.5, 0.0, 1.0))


def _industry_monetary_sensitivity(industry: str, macro: MacroSnapshot) -> float:
    rate_sensitive = {"real_estate", "auto", "consumer_durable", "infrastructure", "construction", "non_bank_financial"}
    insensitive = {"banking", "utilities", "consumer_staples"}
    base = 0.0
    if macro.monetary_stance == "easing":
        base = 0.5
    elif macro.monetary_stance == "tightening":
        base = -0.5
    if industry in rate_sensitive:
        base *= 1.5
    elif industry in insensitive:
        base *= 0.5
    return float(np.clip(base, -1.0, 1.0))


def _industry_fx_pressure(industry: str, macro: MacroSnapshot) -> float:
    exporter = {"electronics", "shipping", "machinery", "textile", "auto_parts"}
    importer = {"airline", "energy", "agriculture", "paper"}
    cny_strength = macro.cny_strength
    if industry in exporter:
        return float(np.clip(-cny_strength, -1.0, 1.0))
    if industry in importer:
        return float(np.clip(cny_strength, -1.0, 1.0))
    return 0.0


def _industry_commodity_pressure(industry: str, macro: MacroSnapshot) -> float:
    cost_takers = {"steel", "chemicals", "cement", "shipping", "auto"}
    cost_makers = {"oil_gas", "coal", "non_ferrous_metals", "rare_earth"}
    z = macro.commodity_index_zscore
    if industry in cost_takers:
        return float(np.clip(-z, -1.0, 1.0))
    if industry in cost_makers:
        return float(np.clip(z, -1.0, 1.0))
    return 0.0


def _expected_industry_growth(latest: pd.DataFrame, pricing_power: float, supply_demand: float) -> float:
    if "revenue_growth" not in latest.columns:
        base = 0.08
    else:
        base = float(pd.to_numeric(latest["revenue_growth"], errors="coerce").dropna().median() or 0.08)
    return float(np.clip(base + 0.30 * (pricing_power - 0.50) + 0.30 * supply_demand, -0.20, 0.40))


def _industry_cycle_stage(supply_demand: float, capex_trend: float, inventory_z: float, macro: MacroSnapshot) -> str:
    if inventory_z > 1.0 and supply_demand < 0:
        return "downturn"
    if supply_demand > 0.30 and capex_trend < 0.0:
        return "late_cycle"
    if macro.business_cycle_stage in {"trough", "early_expansion"} and supply_demand >= 0:
        return "early_cycle"
    if macro.business_cycle_stage == "expansion":
        return "mid_cycle"
    if supply_demand < -0.30 and capex_trend > 0.20:
        return "trough"
    return "recovery" if supply_demand >= 0 else "downturn"


def _monetary_stance(lpr: float, rrr: float, config: EconomicAnalyzerConfig) -> str:
    if lpr < config.monetary_tailwind_lpr_anchor - 0.10 or rrr < config.monetary_tailwind_rrr_anchor - 0.30:
        return "easing"
    if lpr > config.monetary_tailwind_lpr_anchor + 0.10 or rrr > config.monetary_tailwind_rrr_anchor + 0.30:
        return "tightening"
    return "neutral"


def _fiscal_stance(latest: dict) -> str:
    deficit = _safe_float(latest.get("budget_deficit_ratio"), default=0.03)
    if deficit > 0.038:
        return "expansionary"
    if deficit < 0.025:
        return "contractionary"
    return "neutral"


def _business_cycle(pmi: float, credit_impulse: float, inflation: float, config: EconomicAnalyzerConfig) -> str:
    if pmi >= 52.0 and credit_impulse > 0:
        return "expansion"
    if pmi >= 50.0 and credit_impulse > 0:
        return "early_expansion"
    if pmi < 48.5 and inflation > config.inflation_ceiling:
        return "stagflation"
    if pmi < 48.5:
        return "contraction"
    if 48.5 <= pmi < 50.0:
        return "trough"
    return "peak"


def _theme_strength_by_industry(theme_profiles: Iterable) -> dict[str, float]:
    strengths: dict[str, float] = {}
    for profile in theme_profiles:
        category = getattr(profile, "theme_category", "")
        if not category:
            continue
        strengths[category] = max(strengths.get(category, 0.0), float(getattr(profile, "policy_strength", 0.0)))
    return strengths


def _company_pricing_power(row: pd.Series) -> float:
    margin = _safe_float(row.get("gross_margin"))
    margin_trend = _safe_float(row.get("gross_margin_trend"))
    return float(np.clip(0.40 * margin + 0.30 * (margin_trend * 5.0) + 0.30 * _safe_float(row.get("price_index_change")), 0.0, 1.0))


def _company_capital_efficiency(row: pd.Series) -> float:
    roa = _safe_float(row.get("roa"))
    asset_turnover = _safe_float(row.get("asset_turnover"))
    if asset_turnover == 0 and _safe_float(row.get("total_assets")) > 0:
        asset_turnover = _safe_float(row.get("revenue")) / _safe_float(row.get("total_assets"))
    return float(np.clip(0.5 * roa + 0.5 * asset_turnover, -1.0, 1.0))


def _company_demand_visibility(row: pd.Series) -> float:
    return float(np.clip(_safe_float(row.get("order_visibility_score")) / 100.0, 0.0, 1.0))


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(numeric) or np.isinf(numeric):
        return default
    return numeric


def _neutral_macro(as_of_date: str) -> MacroSnapshot:
    return MacroSnapshot(
        as_of_date=as_of_date,
        business_cycle_stage="mid_cycle",
        monetary_stance="neutral",
        fiscal_stance="neutral",
        credit_impulse=0.0,
        cny_strength=0.0,
        commodity_index_zscore=0.0,
        inflation_pressure=0.018,
        risk_appetite=0.50,
        rationale="macro_indicators_unavailable",
    )
