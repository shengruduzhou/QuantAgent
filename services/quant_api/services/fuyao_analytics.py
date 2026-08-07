"""Deterministic analytics for Fuyao best-practice views.

The functions here transform already-fetched official API payloads.  They do not
perform network requests and never fabricate missing fields.  Ratios with a
missing/zero denominator remain ``None`` as required by the upstream examples.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

import pandas as pd


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    n = _number(numerator)
    d = _number(denominator)
    if n is None or d is None or d == 0:
        return None
    value = n / d
    return value if isfinite(value) else None


def cashflow_quality(financial_health: Mapping[str, Any]) -> dict[str, Any]:
    """Build the official cash-flow-quality ratios on aligned annual periods."""
    statements = financial_health.get("statements") or {}
    income = _by_period(statements.get("income", []))
    balance = _by_period(statements.get("balance", []))
    cashflow = _by_period(statements.get("cashflow", []))
    periods = sorted(set(income) | set(balance) | set(cashflow), reverse=True)

    rows: list[dict[str, Any]] = []
    required = (
        "net_profit",
        "operating_income",
        "assets_total",
        "accounts_receivable",
        "cash",
        "total_debt",
        "act_cash_flow_net",
        "pay_fixed_assets_etc_cash",
    )
    for period in periods:
        merged: dict[str, Any] = {}
        report_dates: list[int] = []
        for source in (income.get(period), balance.get(period), cashflow.get(period)):
            if not source:
                continue
            merged.update(source)
            report_date = source.get("report_date_ms")
            if isinstance(report_date, int):
                report_dates.append(report_date)
        op_cash = _number(merged.get("act_cash_flow_net"))
        capex = _number(merged.get("pay_fixed_assets_etc_cash"))
        free_cash_flow = None if op_cash is None or capex is None else op_cash - capex
        present = sum(1 for field in required if _number(merged.get(field)) is not None)
        rows.append(
            {
                "periodEndMs": period,
                "reportDateMs": max(report_dates) if report_dates else None,
                "cashConversion": _ratio(merged.get("act_cash_flow_net"), merged.get("net_profit")),
                "freeCashFlow": free_cash_flow,
                "freeCashFlowMargin": _ratio(free_cash_flow, merged.get("operating_income")),
                "accrualRatio": _ratio(
                    None if _number(merged.get("net_profit")) is None or op_cash is None else _number(merged.get("net_profit")) - op_cash,
                    merged.get("assets_total"),
                ),
                "receivablePressure": _ratio(merged.get("accounts_receivable"), merged.get("operating_income")),
                "netCashRatio": _ratio(
                    None if _number(merged.get("cash")) is None or _number(merged.get("total_debt")) is None else _number(merged.get("cash")) - _number(merged.get("total_debt")),
                    merged.get("assets_total"),
                ),
                "fieldCompleteness": present / len(required),
                "missingFields": [field for field in required if _number(merged.get(field)) is None],
            }
        )

    return {
        "symbol": financial_health.get("symbol"),
        "source": financial_health.get("source"),
        "pitKey": "report_date_ms",
        "periodKey": "period_end_ms",
        "rows": rows,
        "formulas": {
            "cashConversion": "act_cash_flow_net / net_profit",
            "freeCashFlowMargin": "(act_cash_flow_net - pay_fixed_assets_etc_cash) / operating_income",
            "accrualRatio": "(net_profit - act_cash_flow_net) / assets_total",
            "receivablePressure": "accounts_receivable / operating_income",
            "netCashRatio": "(cash - total_debt) / assets_total",
        },
        "boundary": "Historical visibility is controlled by report_date_ms; zero/missing denominators stay null.",
        "provenance": financial_health.get("provenance"),
    }


def attention_price_resonance(
    *,
    symbol: str,
    benchmark: str,
    rank_rows: list[Mapping[str, Any]],
    stock_bars: list[Mapping[str, Any]],
    benchmark_bars: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Align raw hot-list ranks, stock price and benchmark on one daily axis."""
    rank = pd.DataFrame(rank_rows)
    stock = pd.DataFrame(stock_bars)
    bench = pd.DataFrame(benchmark_bars)
    if rank.empty or stock.empty or bench.empty:
        return {
            "symbol": symbol,
            "benchmark": benchmark,
            "rows": [],
            "spearman": None,
            "sampleSize": 0,
            "boundary": "Missing official history remains unavailable; no synthetic line is created.",
        }

    rank["date"] = pd.to_datetime(rank.get("date", rank.get("date_ms")), errors="coerce")
    if "date_ms" in rank and rank["date"].isna().all():
        rank["date"] = pd.to_datetime(rank["date_ms"], unit="ms", errors="coerce", utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
    stock["date"] = pd.to_datetime(stock["datetime"], errors="coerce")
    bench["date"] = pd.to_datetime(bench["datetime"], errors="coerce")
    stock = stock[["date", "close"]].rename(columns={"close": "stockClose"})
    bench = bench[["date", "close"]].rename(columns={"close": "benchmarkClose"})
    rank = rank[["date", "rank"]]
    merged = rank.merge(stock, on="date", how="inner").merge(bench, on="date", how="inner").sort_values("date")
    if merged.empty:
        return {"symbol": symbol, "benchmark": benchmark, "rows": [], "spearman": None, "sampleSize": 0}

    merged["rankImprovement"] = merged["rank"].shift(1) - merged["rank"]
    merged["stockReturn"] = pd.to_numeric(merged["stockClose"], errors="coerce").pct_change()
    merged["benchmarkReturn"] = pd.to_numeric(merged["benchmarkClose"], errors="coerce").pct_change()
    merged["relativeReturn"] = merged["stockReturn"] - merged["benchmarkReturn"]
    first_stock = _number(merged["stockClose"].iloc[0])
    first_bench = _number(merged["benchmarkClose"].iloc[0])
    merged["stockIndexed"] = merged["stockClose"] / first_stock * 100 if first_stock not in {None, 0} else pd.NA
    merged["benchmarkIndexed"] = merged["benchmarkClose"] / first_bench * 100 if first_bench not in {None, 0} else pd.NA
    usable = merged[["rankImprovement", "relativeReturn"]].dropna()
    spearman = usable["rankImprovement"].rank().corr(usable["relativeReturn"].rank()) if len(usable) >= 3 else None
    if spearman is not None and not isfinite(float(spearman)):
        spearman = None

    rows = []
    for record in merged.to_dict(orient="records"):
        rows.append({key: _json_value(value) for key, value in record.items()})
    return {
        "symbol": symbol,
        "benchmark": benchmark,
        "rows": rows,
        "spearman": None if spearman is None else float(spearman),
        "sampleSize": int(len(usable)),
        "rankAxis": "raw rank; smaller number means hotter; render inverted",
        "boundary": "Contemporaneous association only; it does not establish causality or future return.",
    }


def _by_period(rows: Any) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        period = row.get("period_end_ms")
        if isinstance(period, int):
            out[period] = dict(row)
    return out


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


__all__ = ["attention_price_resonance", "cashflow_quality"]
