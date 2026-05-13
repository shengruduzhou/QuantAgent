from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.fundamental.forensic_accounting import fraud_risk_composite
from quantagent.v7.schemas import FraudRiskScore


def score_fraud_risk(frame: pd.DataFrame) -> list[FraudRiskScore]:
    """Score accounting and disclosure risk from point-in-time financial rows."""
    if frame.empty:
        return []
    data = frame.copy()
    composite = fraud_risk_composite(data) if {"symbol", "report_date"}.issubset(data.columns) else pd.DataFrame()
    if not composite.empty:
        latest = composite.sort_values(["symbol", "report_date"]).groupby("symbol", sort=False).tail(1).set_index("symbol")
    else:
        latest = pd.DataFrame()
    data["report_date"] = pd.to_datetime(data["report_date"]) if "report_date" in data.columns else pd.Timestamp("1970-01-01")
    data = data.sort_values(["symbol", "report_date"]).groupby("symbol", sort=False).tail(1)
    scores: list[FraudRiskScore] = []
    for _, row in data.iterrows():
        symbol = str(row["symbol"])
        comp = latest.loc[symbol] if symbol in latest.index else pd.Series(dtype=float)
        receivables = _score_gap(comp.get("receivable_gap", row.get("receivables_risk_score", np.nan)))
        inventory = _score_gap(comp.get("inventory_gap", row.get("inventory_risk_score", np.nan)))
        accruals = _score_gap(comp.get("accrual_ratio", row.get("accruals_quality_score", np.nan)))
        cashflow = _cashflow_risk(row)
        related = _score_ratio(row.get("related_party_amount"), row.get("revenue"))
        regulatory = 90.0 if bool(row.get("has_regulatory_penalty", False)) else float(row.get("regulatory_penalty_score", 10.0))
        audit = _audit_risk(str(row.get("audit_opinion", "standard")))
        overall = float(
            np.clip(
                0.18 * receivables
                + 0.14 * inventory
                + 0.18 * accruals
                + 0.18 * cashflow
                + 0.10 * related
                + 0.12 * regulatory
                + 0.10 * audit,
                0.0,
                100.0,
            )
        )
        scores.append(
            FraudRiskScore(
                symbol=symbol,
                beneish_m_score=_optional_float(row.get("beneish_m_score")),
                piotroski_f_score=_optional_float(row.get("piotroski_f_score")),
                altman_z_score=_optional_float(row.get("altman_z_score")),
                accruals_quality_score=100.0 - accruals,
                cashflow_quality_score=100.0 - cashflow,
                receivables_risk_score=receivables,
                inventory_risk_score=inventory,
                related_party_risk_score=related,
                regulatory_penalty_score=regulatory,
                audit_opinion_score=audit,
                overall_fraud_risk_score=overall,
                risk_flags=_fraud_flags(overall, regulatory, audit, cashflow, receivables, inventory),
            )
        )
    return scores


def _score_gap(value: object) -> float:
    if value is None or pd.isna(value):
        return 50.0
    return float(np.clip(50.0 + float(value) * 120.0, 0.0, 100.0))


def _score_ratio(numerator: object, denominator: object) -> float:
    if numerator is None or denominator is None or pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0.0:
        return 20.0
    return float(np.clip(float(numerator) / abs(float(denominator)) * 250.0, 0.0, 100.0))


def _cashflow_risk(row: pd.Series) -> float:
    cfo = float(row.get("operating_cash_flow", 0.0) or 0.0)
    net_income = float(row.get("net_income", 0.0) or 0.0)
    if net_income <= 0:
        return 55.0 if cfo < 0 else 35.0
    ratio = cfo / max(net_income, 1.0)
    return float(np.clip(80.0 - ratio * 55.0, 0.0, 100.0))


def _audit_risk(opinion: str) -> float:
    text = opinion.lower()
    if "adverse" in text or "disclaimer" in text or "non_standard" in text:
        return 95.0
    if "qualified" in text or "emphasis" in text:
        return 65.0
    return 10.0


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _fraud_flags(overall: float, regulatory: float, audit: float, cashflow: float, receivables: float, inventory: float) -> tuple[str, ...]:
    flags = []
    if overall > 80:
        flags.append("high_fraud_risk")
    elif overall >= 60:
        flags.append("medium_fraud_risk")
    if regulatory > 70:
        flags.append("regulatory_penalty")
    if audit > 60:
        flags.append("audit_opinion_risk")
    if cashflow > 70:
        flags.append("cashflow_profit_mismatch")
    if receivables > 70:
        flags.append("receivables_anomaly")
    if inventory > 70:
        flags.append("inventory_anomaly")
    return tuple(flags)
