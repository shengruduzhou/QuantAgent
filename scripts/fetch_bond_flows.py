#!/usr/bin/env python3
"""Fetch China bond-market data (国债收益率曲线 + 信用利差) into a raw frame
consumable by ``quantagent import-bond-flows-v7``.

Source: akshare ``bond_china_yield`` (中债收益率曲线). We extract the treasury
curve (中债国债收益率曲线) for 3m/1y/5y/10y points and derive a real credit
spread from the most-liquid high-grade credit benchmark (中债中短期票据 AAA)
minus the matching treasury tenor. Term spreads (10y-1y, 10y-3m) are derived
downstream by the BondFlowBuilder.

Usage:
    python scripts/fetch_bond_flows.py --lookback-days 120 \
        --output runtime/data/v7/raw/bond/bond_yields_raw.csv
    quantagent import-bond-flows-v7 \
        --input runtime/data/v7/raw/bond/bond_yields_raw.csv \
        --source akshare:bond_china_yield --min-days 30
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

TREASURY_CURVE = "中债国债收益率曲线"
# Most-liquid high-grade credit benchmark; used to derive a real credit spread.
CREDIT_CURVE_AAA = "中债中短期票据收益率曲线(AAA)"

_TENOR_MAP = {"3月": "yield_3m", "1年": "yield_1y", "5年": "yield_5y", "10年": "yield_10y"}


def _fetch_curve(ak, curve_name: str, start: str, end: str) -> pd.DataFrame:
    raw = ak.bond_china_yield(start_date=start, end_date=end)
    sub = raw[raw["曲线名称"] == curve_name].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["trade_date"] = pd.to_datetime(sub["日期"], errors="coerce")
    cols = {cn: col for cn, col in _TENOR_MAP.items() if cn in sub.columns}
    out = sub[["trade_date", *cols.keys()]].rename(columns=cols)
    for c in cols.values():
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["trade_date"]).sort_values("trade_date")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lookback-days", type=int, default=120)
    ap.add_argument("--start", default=None, help="YYYYMMDD (overrides lookback)")
    ap.add_argument("--end", default=None, help="YYYYMMDD (default: today)")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("runtime/data/v7/raw/bond/bond_yields_raw.csv"),
    )
    args = ap.parse_args()

    import akshare as ak

    end = dt.datetime.strptime(args.end, "%Y%m%d").date() if args.end else dt.date.today()
    start = (
        dt.datetime.strptime(args.start, "%Y%m%d").date()
        if args.start
        else end - dt.timedelta(days=args.lookback_days)
    )
    s, e = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    treasury = _fetch_curve(ak, TREASURY_CURVE, s, e)
    if treasury.empty:
        raise SystemExit(f"no treasury curve rows for {s}..{e}")
    credit = _fetch_curve(ak, CREDIT_CURVE_AAA, s, e)

    frame = treasury.copy()
    if not credit.empty and "yield_10y" in credit.columns:
        cred10 = credit[["trade_date", "yield_10y"]].rename(columns={"yield_10y": "_aaa_10y"})
        frame = frame.merge(cred10, on="trade_date", how="left")
        # Real credit spread: high-grade AAA credit yield minus treasury, in %.
        frame["credit_spread_aa"] = frame["_aaa_10y"] - frame["yield_10y"]
        frame = frame.drop(columns=["_aaa_10y"])

    frame["source"] = "akshare:bond_china_yield"
    frame["source_version"] = f"{s}_{e}"
    frame["fetched_at"] = pd.Timestamp.utcnow().tz_localize(None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    cov = {c: int(frame[c].notna().sum()) for c in frame.columns if c.startswith(("yield_", "credit_"))}
    print(f"wrote {len(frame)} rows -> {args.output}")
    print(f"date range: {frame['trade_date'].min().date()} .. {frame['trade_date'].max().date()}")
    print(f"non-null coverage: {cov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
