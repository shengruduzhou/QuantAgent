from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.fundamental.market_valuation import enrich_market_valuation
from quantagent.v7.schemas import FundamentalDueDiligenceReport, FundamentalScore, FraudRiskScore
from quantagent.v7.scoring import fraud_confidence_multiplier


def build_fundamental_due_diligence(
    financials: pd.DataFrame,
    fundamental_scores: list[FundamentalScore],
    fraud_scores: list[FraudRiskScore],
    as_of_date: str,
) -> list[FundamentalDueDiligenceReport]:
    """Create per-symbol due-diligence reports from PIT statements and market valuation fields."""
    if financials.empty or not fundamental_scores:
        return []
    data = enrich_market_valuation(financials)
    latest = _latest_financial_rows(data)
    fundamentals = {score.symbol: score for score in fundamental_scores}
    fraud_by_symbol = {score.symbol: score for score in fraud_scores}
    reports: list[FundamentalDueDiligenceReport] = []
    for _, row in latest.iterrows():
        symbol = str(row["symbol"])
        score = fundamentals.get(symbol)
        if score is None:
            continue
        fraud = fraud_by_symbol.get(symbol)
        price = _optional_float(row.get("close", row.get("price")))
        total_shares = _optional_float(row.get("total_share_capital", row.get("total_shares")))
        market_cap = score.market_cap if score.market_cap is not None else _market_cap(price, total_shares, row.get("market_cap"))
        intrinsic = _intrinsic_value_per_share(row, score, price)
        margin = _margin_of_safety(price, intrinsic, score.margin_of_safety)
        valuation_flags = _valuation_flags(score)
        fraud_flags = fraud.risk_flags if fraud is not None else score.key_risks
        confidence = float(
            np.clip(
                score.confidence
                * fraud_confidence_multiplier(score.fraud_risk_score, "fundamental")
                * _data_completeness_multiplier(row),
                0.02,
                0.95,
            )
        )
        reports.append(
            FundamentalDueDiligenceReport(
                symbol=symbol,
                as_of_date=as_of_date,
                price=price,
                total_shares=total_shares,
                market_cap=market_cap,
                free_float_market_cap=score.free_float_market_cap,
                estimated_intrinsic_value_per_share=intrinsic,
                margin_of_safety=margin,
                quality_score=score.quality_score,
                growth_score=score.growth_score,
                valuation_score=score.valuation_score,
                earnings_visibility_score=score.earnings_visibility_score,
                fraud_risk_score=score.fraud_risk_score,
                confidence=confidence,
                investment_horizon_days=score.investment_horizon,
                valuation_flags=valuation_flags,
                fraud_flags=fraud_flags,
                required_follow_up=_required_follow_up(row, score, market_cap),
                rationale=score.rationale,
            )
        )
    return sorted(reports, key=lambda item: item.symbol)


def _latest_financial_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "report_date" not in frame.columns:
        return frame.drop_duplicates("symbol", keep="last") if "symbol" in frame.columns else frame
    data = frame.copy()
    data["report_date"] = pd.to_datetime(data["report_date"])
    return data.sort_values(["symbol", "report_date"]).groupby("symbol", sort=False).tail(1)


def _market_cap(price: float | None, shares: float | None, explicit: object) -> float | None:
    explicit_value = _optional_float(explicit)
    if explicit_value is not None and explicit_value > 0.0:
        return explicit_value
    if price is None or shares is None:
        return None
    return float(price * shares)


def _intrinsic_value_per_share(row: pd.Series, score: FundamentalScore, price: float | None) -> float | None:
    for column in ("intrinsic_value_per_share", "fair_value_per_share", "target_price"):
        value = _optional_float(row.get(column))
        if value is not None and value > 0.0:
            return value
    if price is None or price <= 0.0:
        return None
    margin = score.margin_of_safety
    if not np.isfinite(margin):
        margin = (score.valuation_score - 50.0) / 100.0
    return float(max(0.0, price * (1.0 + margin)))


def _margin_of_safety(price: float | None, intrinsic: float | None, fallback: float) -> float:
    if price is not None and price > 0.0 and intrinsic is not None:
        return float(intrinsic / price - 1.0)
    return float(fallback)


def _valuation_flags(score: FundamentalScore) -> tuple[str, ...]:
    flags = []
    if score.valuation_bubble_score >= 75.0:
        flags.append("valuation_bubble_risk")
    if score.industry_valuation_percentile is not None and score.industry_valuation_percentile >= 80.0:
        flags.append("expensive_vs_industry")
    if score.history_valuation_percentile is not None and score.history_valuation_percentile >= 80.0:
        flags.append("expensive_vs_history")
    if score.margin_of_safety < -0.20:
        flags.append("negative_margin_of_safety")
    if score.valuation_score >= 70.0 and score.margin_of_safety >= 0.0:
        flags.append("valuation_supportive")
    return tuple(flags)


def _required_follow_up(row: pd.Series, score: FundamentalScore, market_cap: float | None) -> tuple[str, ...]:
    follow_up = list(score.required_follow_up)
    if market_cap is None or market_cap <= 0.0:
        follow_up.append("market_cap_and_share_capital")
    if _optional_float(row.get("revenue")) is None:
        follow_up.append("revenue_statement")
    if _optional_float(row.get("operating_cash_flow")) is None:
        follow_up.append("cashflow_statement")
    if score.fraud_risk_score >= 60.0:
        follow_up.append("fraud_risk_manual_review")
    return tuple(sorted(set(follow_up)))


def _data_completeness_multiplier(row: pd.Series) -> float:
    required = ("revenue", "net_income", "operating_cash_flow", "total_assets", "close")
    available = sum(1 for column in required if column in row.index and row.get(column) is not None and not pd.isna(row.get(column)))
    return float(np.clip(0.55 + 0.09 * available, 0.55, 1.0))


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
