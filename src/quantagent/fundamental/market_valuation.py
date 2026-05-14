from __future__ import annotations

import numpy as np
import pandas as pd


def enrich_market_valuation(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive market-cap and relative valuation fields from PIT fundamentals."""
    if frame.empty:
        return frame.copy()
    data = frame.copy()
    close = _numeric(data, "close", fallback=_numeric(data, "price", default=np.nan))
    total_shares = _numeric(data, "total_share_capital", fallback=_numeric(data, "total_shares", default=np.nan))
    free_float_shares = _numeric(data, "free_float_shares", fallback=_numeric(data, "float_shares", default=np.nan))
    if "market_cap" not in data.columns:
        data["market_cap"] = close * total_shares
    if "free_float_market_cap" not in data.columns:
        data["free_float_market_cap"] = close * free_float_shares
    if "ps_ttm" not in data.columns and "revenue" in data.columns:
        data["ps_ttm"] = _ratio(data["market_cap"], _numeric(data, "revenue", default=np.nan))
    if "pe_ttm" not in data.columns and "net_income" in data.columns:
        data["pe_ttm"] = _ratio(data["market_cap"], _numeric(data, "net_income", default=np.nan))
    if "pb" not in data.columns and "book_value" in data.columns:
        data["pb"] = _ratio(data["market_cap"], _numeric(data, "book_value", default=np.nan))
    if "ev_ebitda" not in data.columns and "ebitda" in data.columns:
        enterprise_value = data["market_cap"] + _numeric(data, "total_debt", default=0.0) - _numeric(data, "cash", default=0.0)
        data["ev_ebitda"] = _ratio(enterprise_value, _numeric(data, "ebitda", default=np.nan))
    if "peg" not in data.columns:
        profit_growth = _numeric(data, "profit_growth", fallback=_numeric(data, "net_income_growth", default=np.nan))
        pe = _numeric(data, "pe_ttm", default=np.nan)
        data["peg"] = _ratio(pe, profit_growth.abs() * 100.0)
    data["industry_valuation_percentile"] = _relative_percentile(data, "industry")
    data["history_valuation_percentile"] = _history_percentile(data)
    data["valuation_bubble_score"] = _bubble_score(data)
    if "margin_of_safety" not in data.columns:
        percentile = data["industry_valuation_percentile"].fillna(data["history_valuation_percentile"]).fillna(50.0)
        data["margin_of_safety"] = (100.0 - percentile) / 100.0 - 0.35
    return data


def _relative_percentile(data: pd.DataFrame, group_column: str) -> pd.Series:
    valuation = _valuation_composite(data)
    if group_column not in data.columns:
        return valuation.rank(pct=True) * 100.0
    return valuation.groupby(data[group_column].fillna("unknown")).rank(pct=True) * 100.0


def _history_percentile(data: pd.DataFrame) -> pd.Series:
    valuation = _valuation_composite(data)
    if "symbol" not in data.columns or "report_date" not in data.columns:
        return valuation.rank(pct=True) * 100.0
    ordered = data.assign(_valuation=valuation).sort_values(["symbol", "report_date"])
    return ordered.groupby("symbol")["_valuation"].rank(pct=True).reindex(data.index).fillna(50.0) * 100.0


def _bubble_score(data: pd.DataFrame) -> pd.Series:
    industry_pct = data.get("industry_valuation_percentile", pd.Series(50.0, index=data.index)).fillna(50.0)
    history_pct = data.get("history_valuation_percentile", pd.Series(50.0, index=data.index)).fillna(50.0)
    revenue_growth = _numeric(data, "revenue_growth", default=0.0) * 100.0
    profit_growth = _numeric(data, "profit_growth", fallback=_numeric(data, "net_income_growth", default=0.0)) * 100.0
    growth_support = np.maximum(revenue_growth, profit_growth).clip(lower=0.0, upper=80.0)
    return (0.55 * industry_pct + 0.35 * history_pct - 0.40 * growth_support).clip(lower=0.0, upper=100.0)


def _valuation_composite(data: pd.DataFrame) -> pd.Series:
    pe = _numeric(data, "pe_ttm", default=25.0).clip(lower=0.0, upper=300.0).rank(pct=True)
    pb = _numeric(data, "pb", default=3.0).clip(lower=0.0, upper=30.0).rank(pct=True)
    ps = _numeric(data, "ps_ttm", fallback=_numeric(data, "ps", default=5.0)).clip(lower=0.0, upper=80.0).rank(pct=True)
    ev = _numeric(data, "ev_ebitda", default=15.0).clip(lower=0.0, upper=100.0).rank(pct=True)
    peg = _numeric(data, "peg", default=1.5).clip(lower=0.0, upper=20.0).rank(pct=True)
    return 0.30 * pe + 0.18 * pb + 0.18 * ps + 0.18 * ev + 0.16 * peg


def _numeric(data: pd.DataFrame, column: str, default: float | None = None, fallback: pd.Series | None = None) -> pd.Series:
    if column in data.columns:
        return pd.to_numeric(data[column], errors="coerce")
    if fallback is not None:
        return fallback
    return pd.Series(default, index=data.index, dtype="float64")


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator
