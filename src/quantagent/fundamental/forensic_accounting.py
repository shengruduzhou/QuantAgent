from __future__ import annotations

import numpy as np
import pandas as pd


def receivable_growth_vs_revenue_growth(frame: pd.DataFrame) -> pd.Series:
    data = _prepare(frame)
    receivable_growth = data.groupby("symbol", sort=False)["receivables"].pct_change()
    revenue_growth = data.groupby("symbol", sort=False)["revenue"].pct_change()
    return (receivable_growth - revenue_growth).replace([np.inf, -np.inf], np.nan)


def inventory_growth_vs_cogs_growth(frame: pd.DataFrame) -> pd.Series:
    data = _prepare(frame)
    inventory_growth = data.groupby("symbol", sort=False)["inventory"].pct_change()
    cogs_growth = data.groupby("symbol", sort=False)["cogs"].pct_change()
    return (inventory_growth - cogs_growth).replace([np.inf, -np.inf], np.nan)


def accrual_ratio(frame: pd.DataFrame) -> pd.Series:
    data = _prepare(frame)
    return ((data["net_income"] - data["operating_cash_flow"]) / data["total_assets"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def cfo_to_net_income(frame: pd.DataFrame) -> pd.Series:
    data = _prepare(frame)
    return (data["operating_cash_flow"] / data["net_income"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def capex_pressure(frame: pd.DataFrame) -> pd.Series:
    data = _prepare(frame)
    return (data["capex"].abs() / data["operating_cash_flow"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def goodwill_impairment_risk(frame: pd.DataFrame) -> pd.Series:
    data = _prepare(frame)
    goodwill = data["goodwill"] if "goodwill" in data.columns else pd.Series(0.0, index=data.index)
    return (goodwill / data["total_assets"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def debt_maturity_pressure(frame: pd.DataFrame) -> pd.Series:
    data = _prepare(frame)
    short_debt = data["short_term_debt"] if "short_term_debt" in data.columns else pd.Series(np.nan, index=data.index)
    cash = data["cash"] if "cash" in data.columns else pd.Series(np.nan, index=data.index)
    return (short_debt / cash.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def related_party_transaction_risk(frame: pd.DataFrame) -> pd.Series:
    data = _prepare(frame)
    if "related_party_amount" not in data.columns:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return (data["related_party_amount"] / data["revenue"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def fraud_risk_composite(frame: pd.DataFrame) -> pd.DataFrame:
    data = _prepare(frame)
    metrics = pd.DataFrame(index=data.index)
    metrics["receivable_gap"] = receivable_growth_vs_revenue_growth(data)
    metrics["inventory_gap"] = inventory_growth_vs_cogs_growth(data)
    metrics["accrual_ratio"] = accrual_ratio(data)
    metrics["cfo_shortfall"] = -cfo_to_net_income(data)
    metrics["capex_pressure"] = capex_pressure(data)
    metrics["goodwill_risk"] = goodwill_impairment_risk(data)
    metrics["debt_pressure"] = debt_maturity_pressure(data)
    metrics["related_party_risk"] = related_party_transaction_risk(data)
    ranked = metrics.rank(pct=True)
    data["fraud_risk_composite"] = ranked.mean(axis=1, skipna=True)
    return pd.concat([data[["symbol", "report_date"]], metrics, data["fraud_risk_composite"]], axis=1)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["report_date"] = pd.to_datetime(data["report_date"])
    for column in [
        "receivables",
        "revenue",
        "inventory",
        "cogs",
        "net_income",
        "operating_cash_flow",
        "total_assets",
        "capex",
    ]:
        if column not in data.columns:
            data[column] = np.nan
    return data.sort_values(["symbol", "report_date"]).reset_index(drop=True)
