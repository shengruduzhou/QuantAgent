#!/usr/bin/env python3
"""Infer 国家队 (state-team) activity into silver/state_team_inference.

Signal: 中央汇金 / 证金 / 社保基金 / 国新 appearing in a stock's top-10
shareholders. Source: akshare ``stock_gdfx_top_10_em`` (东方财富 前十大股东),
scanned over a curated set of large-caps the state team is known to hold,
at recent quarter-end report dates. Feeds the project's
``StateTeamInferenceBuilder`` (which flags ``evidence_label='inferred'``).

Usage:
    python scripts/fetch_state_team.py --dates 20251231,20260331 \
        --output-root runtime/data/v7
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from quantagent.data.state_team import StateTeamInferenceBuilder, StateTeamInferenceConfig

# Large-caps the state team (汇金/证金/社保) is historically known to hold.
DEFAULT_SYMBOLS = [
    "601398.SH", "601939.SH", "601288.SH", "601988.SH", "601328.SH",  # big-4 + 交行
    "601318.SH", "601601.SH", "601628.SH", "601336.SH",               # 平安/太保/国寿/新华
    "600036.SH", "601166.SH", "600000.SH", "600016.SH",               # 招商/兴业/浦发/民生
    "600519.SH", "600028.SH", "601857.SH", "601088.SH", "600900.SH",  # 茅台/中石化/中石油/神华/长电
    "601668.SH", "601800.SH", "600050.SH", "601390.SH",               # 建筑/中交/联通/中铁
]


def _ak_symbol(symbol: str) -> str:
    code, _, ex = str(symbol).partition(".")
    return f"{ex.lower()}{code}" if ex else symbol


def _scan(ak, symbol: str, date: str) -> list[dict]:
    try:
        df = ak.stock_gdfx_top_10_em(symbol=_ak_symbol(symbol), date=date)
    except Exception:
        return []
    if df is None or df.empty or "股东名称" not in df.columns:
        return []
    share_col = "占总股本持股比例" if "占总股本持股比例" in df.columns else None
    rows = []
    for _, r in df.iterrows():
        pct = pd.to_numeric(str(r.get(share_col, "")).replace("%", ""), errors="coerce") if share_col else None
        rows.append(
            {
                "trade_date": pd.Timestamp(date),
                "symbol": symbol,
                "holder_name": str(r.get("股东名称") or ""),
                "share_pct": float(pct) if pct == pct else 0.0,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", help="comma list; default = curated state-team holdings")
    ap.add_argument("--dates", default="20251231,20260331", help="comma quarter-end report dates YYYYMMDD")
    ap.add_argument("--output-root", type=Path, default=Path("runtime/data/v7"))
    ap.add_argument("--min-events", type=int, default=3)
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else DEFAULT_SYMBOLS
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]

    import akshare as ak

    rows: list[dict] = []
    for d in dates:
        for sym in symbols:
            rows.extend(_scan(ak, sym, d))
    holders = pd.DataFrame(rows)
    n_state = 0
    if not holders.empty:
        from quantagent.data.state_team.builder import STATE_TEAM_HOLDER_KEYWORDS

        n_state = holders["holder_name"].apply(
            lambda h: any(k in str(h) for k in STATE_TEAM_HOLDER_KEYWORDS)
        ).sum()
    print(f"scanned {len(symbols)} symbols x {len(dates)} dates -> {len(holders)} holder rows, {n_state} state-team hits")
    if holders.empty:
        print("no holder data; nothing to write")
        return 0

    cfg = StateTeamInferenceConfig(
        source="akshare:stock_gdfx_top_10_em",
        source_version=dt.date.today().strftime("%Y%m%d"),
        output_root=args.output_root,
        min_events=args.min_events,
    )
    builder = StateTeamInferenceBuilder(cfg)
    result = builder.write(builder.build(top10_holders=holders))
    print(f"gate: {result.coverage.get('gate')} | events: {len(result.frame)}")
    print(f"silver: {result.output_paths.get('state_team_inference')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
