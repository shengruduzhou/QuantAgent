from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.fundamental.market_valuation import enrich_market_valuation
from quantagent.v7.schemas import FundamentalScore


def score_financial_statements(frame: pd.DataFrame, horizon_days: int = 120) -> list[FundamentalScore]:
    """Score latest point-in-time financial rows into V7 fundamental scores."""
    if frame.empty:
        return []
    data = enrich_market_valuation(frame)
    if "report_date" in data.columns:
        data["report_date"] = pd.to_datetime(data["report_date"])
        data = data.sort_values(["symbol", "report_date"]).groupby("symbol", sort=False).tail(1)
    rows: list[FundamentalScore] = []
    for _, row in data.iterrows():
        symbol = str(row["symbol"])
        quality = _quality_score(row)
        growth = _growth_score(row)
        valuation = _valuation_score(row)
        bubble = _safe_float(row.get("valuation_bubble_score", 0.0))
        earnings_visibility = _visibility_score(row)
        fraud = float(row.get("fraud_risk_score", 50.0))
        management_risk = float(row.get("management_risk_score", row.get("governance_risk_score", 35.0)))
        margin = float(row.get("margin_of_safety", _safe_float(row.get("margin_of_safety_score", 0.0)) / 100.0))
        theme_exposure = _safe_float(row.get("theme_revenue_exposure", row.get("revenue_exposure", 0.0)))
        fundamental = (
            0.18 * theme_exposure
            + 0.14 * growth
            + 0.14 * quality
            + 0.14 * earnings_visibility
            + 0.12 * valuation
            + 0.10 * max(0.0, 100.0 - fraud)
            + 0.08 * max(0.0, 100.0 - management_risk)
            + 0.10 * _margin_to_score(margin)
            + 0.10 * _safe_float(row.get("order_visibility_score", 50.0))
            - 0.08 * bubble
        )
        rows.append(
            FundamentalScore(
                symbol=symbol,
                fundamental_score=float(np.clip(fundamental, 0.0, 100.0)),
                quality_score=quality,
                growth_score=growth,
                valuation_score=valuation,
                earnings_visibility_score=earnings_visibility,
                fraud_risk_score=fraud,
                management_risk_score=management_risk,
                margin_of_safety=margin,
                investment_horizon=horizon_days,
                confidence=float(np.clip(0.35 + quality / 300.0 + earnings_visibility / 300.0 - fraud / 400.0 - bubble / 600.0, 0.05, 0.95)),
                rationale=_rationale(row, quality, growth, valuation, fraud, bubble),
                key_risks=_key_risks(row, fraud, management_risk),
                required_follow_up=_required_follow_up(row),
                market_cap=_optional_float(row.get("market_cap")),
                free_float_market_cap=_optional_float(row.get("free_float_market_cap")),
                pe_ttm=_optional_float(row.get("pe_ttm")),
                pb=_optional_float(row.get("pb")),
                ps_ttm=_optional_float(row.get("ps_ttm", row.get("ps"))),
                ev_ebitda=_optional_float(row.get("ev_ebitda")),
                peg=_optional_float(row.get("peg")),
                industry_valuation_percentile=_optional_float(row.get("industry_valuation_percentile")),
                history_valuation_percentile=_optional_float(row.get("history_valuation_percentile")),
                valuation_bubble_score=bubble,
            )
        )
    return rows


def _quality_score(row: pd.Series) -> float:
    roe = _safe_float(row.get("roe", 0.0)) * 100.0
    roa = _safe_float(row.get("roa", 0.0)) * 100.0
    gross_margin = _safe_float(row.get("gross_margin", 0.0)) * 100.0
    cfo = _safe_float(row.get("operating_cash_flow", 0.0))
    net_income = _safe_float(row.get("net_income", 0.0))
    cash_quality = 50.0 if net_income == 0 else np.clip((cfo / max(abs(net_income), 1.0)) * 50.0, 0.0, 100.0)
    debt_penalty = _safe_float(row.get("debt_to_asset", 0.5)) * 45.0
    return float(np.clip(25.0 + 1.3 * roe + 0.8 * roa + 0.35 * gross_margin + 0.20 * cash_quality - debt_penalty, 0.0, 100.0))


def _growth_score(row: pd.Series) -> float:
    revenue_growth = _safe_float(row.get("revenue_growth", 0.0)) * 100.0
    profit_growth = _safe_float(row.get("profit_growth", row.get("net_income_growth", 0.0))) * 100.0
    return float(np.clip(50.0 + 0.45 * revenue_growth + 0.35 * profit_growth, 0.0, 100.0))


def _valuation_score(row: pd.Series) -> float:
    percentile = row.get("industry_valuation_percentile", row.get("valuation_percentile"))
    if percentile is not None and not pd.isna(percentile):
        return float(np.clip(100.0 - float(percentile), 0.0, 100.0))
    pe = _safe_float(row.get("pe_ttm", 25.0))
    pb = _safe_float(row.get("pb", 3.0))
    return float(np.clip(100.0 - 1.2 * pe - 5.0 * pb, 0.0, 100.0))


def _visibility_score(row: pd.Series) -> float:
    order = _safe_float(row.get("order_visibility_score", 50.0))
    capacity = _safe_float(row.get("capacity_release_score", 50.0))
    customer = _safe_float(row.get("customer_validation_score", 50.0))
    return float(np.clip(0.4 * order + 0.3 * capacity + 0.3 * customer, 0.0, 100.0))


def _margin_to_score(margin: float) -> float:
    return float(np.clip(50.0 + margin * 100.0, 0.0, 100.0))


def _safe_float(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _rationale(row: pd.Series, quality: float, growth: float, valuation: float, fraud: float, bubble: float) -> str:
    return (
        f"quality={quality:.1f}, growth={growth:.1f}, valuation={valuation:.1f}, "
        f"fraud_risk={fraud:.1f}, valuation_bubble={bubble:.1f}, "
        f"market_cap={_safe_float(row.get('market_cap', 0.0)):.1f}, "
        f"theme_exposure={_safe_float(row.get('theme_revenue_exposure', 0.0)):.1f}"
    )


def _key_risks(row: pd.Series, fraud: float, management_risk: float) -> tuple[str, ...]:
    risks = []
    if fraud > 60:
        risks.append("fraud_risk_requires_discount")
    if management_risk > 60:
        risks.append("management_risk")
    if _safe_float(row.get("operating_cash_flow", 0.0)) < _safe_float(row.get("net_income", 0.0)) * 0.5:
        risks.append("cashflow_profit_mismatch")
    if _safe_float(row.get("receivables_growth", 0.0)) > _safe_float(row.get("revenue_growth", 0.0)) + 0.2:
        risks.append("receivables_growth_above_revenue")
    if _safe_float(row.get("valuation_bubble_score", 0.0)) >= 75.0:
        risks.append("valuation_bubble_risk")
    return tuple(risks)


def _required_follow_up(row: pd.Series) -> tuple[str, ...]:
    follow_up = []
    if _safe_float(row.get("theme_revenue_exposure", 0.0)) < 40.0:
        follow_up.append("theme_revenue_breakdown")
    if _safe_float(row.get("order_visibility_score", 50.0)) < 55.0:
        follow_up.append("order_and_customer_validation")
    if _safe_float(row.get("market_cap", 0.0)) <= 0.0:
        follow_up.append("market_cap_and_share_capital")
    return tuple(follow_up)
