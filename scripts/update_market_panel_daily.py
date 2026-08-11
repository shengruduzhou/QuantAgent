#!/usr/bin/env python3
"""Incrementally append new trading days to silver/market_panel.parquet.

Historical tradability is point-in-time strict. TickFlow provides the OHLCV
bars, while ST/risk-warning state must come from a separately versioned dated
PIT evidence table accepted by ``quantagent.data.providers.st_pit``. The script
refuses to append when that evidence is absent/incomplete; a current ST snapshot
is never broadcast backwards into the canonical panel.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.data.providers.st_pit import (
    HistoricalSTCoverageError,
    attach_historical_st,
    load_historical_st_evidence,
)
from quantagent.quant_math.ashare import board_price_limit_vector


PANEL = Path("runtime/data/v7/silver/market_panel/market_panel.parquet")


def _tf_client():
    try:
        from dotenv import load_dotenv

        load_dotenv(".env", override=False)
    except Exception:
        pass
    import tickflow

    return tickflow.TickFlow(
        api_key=os.environ["TICKFLOW_API_KEY"],
        base_url=os.environ.get("TICKFLOW_API_ENDPOINT") or None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--active-window-days", type=int, default=30)
    ap.add_argument("--max-symbols", type=int, default=0, help="cap for smoke runs")
    ap.add_argument("--end", default=None, help="last date to fetch (default today)")
    ap.add_argument(
        "--historical-st-path",
        default=os.environ.get("QUANTAGENT_HISTORICAL_ST_PATH"),
        help="dated PIT ST table (parquet/csv/jsonl); required",
    )
    args = ap.parse_args()

    if not args.historical_st_path:
        raise SystemExit(
            "historical ST PIT evidence is required; set --historical-st-path or "
            "QUANTAGENT_HISTORICAL_ST_PATH. Current ST snapshots are not valid for historical append."
        )
    try:
        st_evidence = load_historical_st_evidence(args.historical_st_path)
    except HistoricalSTCoverageError as exc:
        raise SystemExit(f"historical ST PIT evidence invalid: {exc}") from exc

    panel = pd.read_parquet(PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    for provenance_column in ("is_st_provenance", "st_evidence_sha256"):
        if provenance_column not in panel.columns:
            panel[provenance_column] = pd.NA

    last = panel["trade_date"].max()
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.today().normalize()
    if end <= last:
        print(f"panel already at {last.date()} — nothing to do")
        return 0

    recent = panel[panel["trade_date"] >= last - pd.Timedelta(days=args.active_window_days)]
    symbols = sorted(recent["symbol"].astype(str).unique())
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]
    prev_close = recent.sort_values("trade_date").groupby("symbol")["close"].last()

    tf = _tf_client()
    start_fetch = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_fetch = end.strftime("%Y-%m-%d")
    start_ms = int((last - pd.Timedelta(days=3)).timestamp() * 1000)
    end_ms = int((end + pd.Timedelta(days=1)).timestamp() * 1000) - 1
    print(f"fetching {len(symbols)} symbols {start_fetch}..{end_fetch}", flush=True)

    rows: list[pd.DataFrame] = []
    failed = 0
    for i, sym in enumerate(symbols):
        try:
            k = tf.klines.get(
                sym,
                period="1d",
                start_time=start_ms,
                end_time=end_ms,
                adjust="none",
                as_dataframe=True,
            )
        except Exception:
            failed += 1
            continue
        if k is None or len(k) == 0:
            continue
        k = k.copy()
        k["symbol"] = sym
        if "volume" in k.columns:
            k["volume"] = pd.to_numeric(k["volume"], errors="coerce") * 100.0
        rows.append(k)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(symbols)} fetched", flush=True)
    if not rows:
        print(json.dumps({"appended": 0, "failed": failed}))
        return 0

    new = pd.concat(rows, ignore_index=True)
    new["trade_date"] = pd.to_datetime(new["trade_date"])
    new = new[(new["trade_date"] > last) & (new["trade_date"] <= end)]
    for column in ("open", "high", "low", "close", "volume", "amount"):
        if column in new.columns:
            new[column] = pd.to_numeric(new[column], errors="coerce")
    new = new.dropna(subset=["trade_date", "symbol", "close"])
    if new.empty:
        print(json.dumps({"appended": 0, "failed": failed}))
        return 0

    try:
        new = attach_historical_st(new, st_evidence)
    except HistoricalSTCoverageError as exc:
        raise SystemExit(
            "refusing market-panel append because historical ST coverage is incomplete: "
            f"{exc}"
        ) from exc

    new["is_suspended"] = new["volume"].fillna(0) <= 0
    new = new.sort_values(["symbol", "trade_date"])
    previous = new.groupby("symbol")["close"].shift(1)
    previous = previous.fillna(new["symbol"].map(prev_close))
    ratios = board_price_limit_vector(
        new["symbol"].astype(str),
        new["is_st"].astype(bool),
        trade_dates=new["trade_date"],
    )
    up_price = (previous * (1.0 + ratios)).round(2)
    down_price = (previous * (1.0 - ratios)).round(2)
    new["is_limit_up"] = (
        (new["close"].round(2) >= up_price - 0.005) & previous.notna()
    )
    new["is_limit_down"] = (
        (new["close"].round(2) <= down_price + 0.005) & previous.notna()
    )
    new["available_at"] = new["trade_date"] + pd.Timedelta(days=1)
    new["source"] = "tickflow_daily_append"
    new["source_type"] = "vendor_api"
    new["source_reliability"] = 0.9
    new["point_in_time_valid"] = True

    keep_cols = list(panel.columns)
    for column in keep_cols:
        if column not in new.columns:
            new[column] = np.nan
    new = new[keep_cols]

    backup = PANEL.with_name(
        f"market_panel.pre_{end.strftime('%Y%m%d')}.tail.parquet"
    )
    if not backup.exists():
        panel[panel["trade_date"] >= last - pd.Timedelta(days=5)].to_parquet(
            backup, index=False
        )
    merged = pd.concat([panel, new], ignore_index=True)
    merged = merged.drop_duplicates(["symbol", "trade_date"], keep="first")
    merged.to_parquet(PANEL, index=False)
    out = {
        "appended": int(len(new)),
        "failed_symbols": failed,
        "new_max_date": str(merged["trade_date"].max().date()),
        "dates_added": sorted(str(d.date()) for d in new["trade_date"].unique()),
        "historical_st_evidence_sha256": st_evidence.source_sha256,
        "historical_st_evidence_mode": st_evidence.mode,
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
