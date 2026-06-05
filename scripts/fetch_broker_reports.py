#!/usr/bin/env python3
"""Fetch 券商研报 (broker research reports) into silver/broker_reports.

Source: akshare ``stock_research_report_em`` (东方财富 研报中心), queried per
symbol. Maps 机构->broker, 东财评级->rating, 报告名称->summary, 日期->announced_at,
then runs the project's ``BrokerReportBuilder`` (tier + credibility + gate)
and writes ``silver/broker_reports/broker_reports.parquet``.

Usage:
    python scripts/fetch_broker_reports.py \
        --symbols-from runtime/tmp/demo_preds_20260604.parquet \
        --lookback-days 90 --output-root runtime/data/v7
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from quantagent.data.broker import BrokerReportBuilder, BrokerReportConfig


def _code6(symbol: str) -> str:
    s = str(symbol).strip().upper()
    return s.split(".")[0].zfill(6) if "." in s else s.zfill(6)


def _fetch_symbol(ak, symbol: str, cutoff: pd.Timestamp) -> list[dict]:
    code = _code6(symbol)
    try:
        df = ak.stock_research_report_em(symbol=code)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    df = df.copy()
    df["announced_at"] = pd.to_datetime(df.get("日期"), errors="coerce")
    df = df[df["announced_at"].notna() & (df["announced_at"] >= cutoff)]
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                "broker": str(r.get("机构") or ""),
                "symbol": symbol,
                "announced_at": r["announced_at"],
                "rating": str(r.get("东财评级") or "n/a"),
                "summary": str(r.get("报告名称") or ""),
                "sector": str(r.get("行业") or ""),
                "source": "akshare:stock_research_report_em",
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--symbols", help="comma-separated symbols e.g. 000001.SZ,600519.SH")
    g.add_argument("--symbols-from", type=Path, help="parquet/csv with a 'symbol' column")
    ap.add_argument("--lookback-days", type=int, default=90)
    ap.add_argument("--max-symbols", type=int, default=80)
    ap.add_argument("--output-root", type=Path, default=Path("runtime/data/v7"))
    ap.add_argument("--min-events", type=int, default=3)
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        f = args.symbols_from
        df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        symbols = df["symbol"].astype(str).unique().tolist()
    symbols = symbols[: args.max_symbols]

    import akshare as ak

    cutoff = pd.Timestamp(dt.date.today() - dt.timedelta(days=args.lookback_days))
    all_rows: list[dict] = []
    for sym in symbols:
        all_rows.extend(_fetch_symbol(ak, sym, cutoff))
    raw = pd.DataFrame(all_rows)
    print(f"fetched {len(raw)} broker reports across {len(symbols)} symbols")
    if raw.empty:
        print("no broker reports in window; nothing to write")
        return 0

    cfg = BrokerReportConfig(
        source="akshare:stock_research_report_em",
        source_version=dt.date.today().strftime("%Y%m%d"),
        output_root=args.output_root,
        min_events=args.min_events,
    )
    builder = BrokerReportBuilder(cfg)
    result = builder.write(builder.build(raw))
    gate = result.coverage.get("gate", {})
    print(f"gate: {gate} | rows: {len(result.frame)}")
    print(f"silver: {result.output_paths.get('broker_reports')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
