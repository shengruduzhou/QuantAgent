#!/usr/bin/env python3
"""Incrementally append new trading days to silver/market_panel.parquet.

Fetches daily klines from TickFlow for every symbol active in the panel's
recent history, derives the tradability flags with the SAME rules as
``enrich_market_panel.py`` (is_suspended = volume==0; is_limit_up/down by
prev-close band; is_st broadcast from the st_flags snapshot), and appends
rows for dates the panel does not yet have.

Idempotent: re-running on the same day adds nothing. A timestamped backup
of the panel tail is written before the first append of each day.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("runtime/data/v7/silver/market_panel/market_panel.parquet")
ST_FLAGS = Path("runtime/data/v7/silver/st_flags/st_flags.parquet")


def _tf_client():
    try:
        from dotenv import load_dotenv
        load_dotenv(".env", override=False)
    except Exception:
        pass
    import tickflow
    return tickflow.TickFlow(api_key=os.environ["TICKFLOW_API_KEY"],
                             base_url=os.environ.get("TICKFLOW_API_ENDPOINT") or None)


def _board_cap(symbol: str, is_st: bool) -> float:
    s = str(symbol).split(".")[0].zfill(6)
    if s.startswith(("30", "68")):
        return 0.20
    if s.startswith(("82", "83", "87", "88", "43", "92")):
        return 0.30
    return 0.05 if is_st else 0.10


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--active-window-days", type=int, default=30)
    ap.add_argument("--max-symbols", type=int, default=0, help="cap for smoke runs")
    ap.add_argument("--end", default=None, help="last date to fetch (default today)")
    args = ap.parse_args()

    panel = pd.read_parquet(PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
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

    st_map: dict[str, bool] = {}
    if ST_FLAGS.exists():
        st = pd.read_parquet(ST_FLAGS)
        st_map = dict(zip(st["symbol"].astype(str), st["is_st"].astype(bool)))

    tf = _tf_client()
    start_fetch = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_fetch = end.strftime("%Y-%m-%d")
    # SDK 2026-06 change: klines.get takes epoch-ms start_time/end_time (the old
    # start_date/end_date kwargs raise TypeError — this script was broken and the
    # panel froze at 2026-05-18; see FRESH_HOLDOUT_FREEZE_MANIFEST.md).
    # adjust="none" verified to match the panel's as-of-day price basis exactly
    # (600519 2026-05-18: close 1323.00 == panel; forward-adjusted would be 1292.31).
    start_ms = int((last + pd.Timedelta(days=1)).timestamp() * 1000)
    end_ms = int((end + pd.Timedelta(days=1)).timestamp() * 1000) - 1
    print(f"fetching {len(symbols)} symbols {start_fetch}..{end_fetch}", flush=True)

    rows: list[pd.DataFrame] = []
    failed = 0
    for i, sym in enumerate(symbols):
        try:
            k = tf.klines.get(sym, period="1d", start_time=start_ms,
                              end_time=end_ms, adjust="none", as_dataframe=True)
        except Exception:
            failed += 1
            continue
        if k is None or len(k) == 0:
            continue
        k = k.copy()
        k["symbol"] = sym
        # TickFlow daily volume is in lots (手); the panel stores shares (股).
        # Verified: 600519 2026-05-18 tickflow 49,661 lots vs panel 4,966,097 shares.
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
    for c in ("open", "high", "low", "close", "volume", "amount"):
        if c in new.columns:
            new[c] = pd.to_numeric(new[c], errors="coerce")
    new = new.dropna(subset=["trade_date", "symbol", "close"])
    if new.empty:
        print(json.dumps({"appended": 0, "failed": failed}))
        return 0

    # flags — identical semantics to enrich_market_panel.py
    new["is_st"] = new["symbol"].map(lambda s: bool(st_map.get(str(s), False)))
    new["is_st_provenance"] = "current_snapshot_broadcast"
    new["is_suspended"] = new["volume"].fillna(0) <= 0
    new = new.sort_values(["symbol", "trade_date"])
    pc = new.groupby("symbol")["close"].shift(1)
    first_mask = pc.isna()
    pc = pc.fillna(new["symbol"].map(prev_close))
    caps = new.apply(lambda r: _board_cap(r["symbol"], bool(r["is_st"])), axis=1)
    up_px = (pc * (1 + caps)).round(2)
    down_px = (pc * (1 - caps)).round(2)
    new["is_limit_up"] = (new["close"].round(2) >= up_px - 0.005) & pc.notna()
    new["is_limit_down"] = (new["close"].round(2) <= down_px + 0.005) & pc.notna()
    new["available_at"] = new["trade_date"] + pd.Timedelta(days=1)
    new["source"] = "tickflow_daily_append"
    new["source_type"] = "vendor_api"
    new["source_reliability"] = 0.9
    new["point_in_time_valid"] = True
    del first_mask

    keep_cols = [c for c in panel.columns]
    for c in keep_cols:
        if c not in new.columns:
            new[c] = np.nan
    new = new[keep_cols]

    backup = PANEL.with_name(f"market_panel.pre_{end.strftime('%Y%m%d')}.tail.parquet")
    if not backup.exists():
        panel[panel["trade_date"] >= last - pd.Timedelta(days=5)].to_parquet(backup, index=False)
    merged = pd.concat([panel, new], ignore_index=True)
    merged = merged.drop_duplicates(["symbol", "trade_date"], keep="first")
    merged.to_parquet(PANEL, index=False)
    out = {"appended": int(len(new)), "failed_symbols": failed,
           "new_max_date": str(merged['trade_date'].max().date()),
           "dates_added": sorted(str(d.date()) for d in new["trade_date"].unique())}
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
