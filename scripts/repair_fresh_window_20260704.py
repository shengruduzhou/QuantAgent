#!/usr/bin/env python3
"""One-off repair of the fresh-window ingest (2026-05-19..2026-07-02).

The 2026-07-03 ingest left two defects (see FRESH_HOLDOUT_FREEZE_MANIFEST.md):
  1. 2026-05-19 missing entirely — TickFlow drops the first day of a
     start_time-bounded request (empirically verified twice), so the appended
     block started at 05-20.
  2. ~1,281/3,653 symbols failed (throttling) on every day of the window.

Because update_market_panel_daily.py only appends dates > panel max, a rerun
cannot backfill. This script:
  * fetches count=40 daily bars (adjust="none", volume lots->shares x100) for
    every symbol active on 2026-05-18, with 3-attempt backoff;
  * inserts ONLY (symbol, date) rows missing from the panel within the window
    — existing rows are never overwritten;
  * recomputes is_suspended / is_limit_up / is_limit_down / is_st for ALL
    fresh-window rows from the completed price chain (the 07-03 run derived
    05-20 limit flags against the 05-18 close because 05-19 was absent);
  * writes a pre-repair tail backup, then rewrites the panel.

One-shot: safe to re-run (idempotent by construction), delete after the
freeze manifest is committed.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = Path("runtime/data/v7/silver/market_panel/market_panel.parquet")
ST_FLAGS = Path("runtime/data/v7/silver/st_flags/st_flags.parquet")
WIN_START = pd.Timestamp("2026-05-19")
WIN_END = pd.Timestamp("2026-07-02")
SEED_DATE = pd.Timestamp("2026-05-18")   # last pre-window day: prev-close seed


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


def fetch_with_retry(tf, sym: str, attempts: int = 3):
    for i in range(attempts):
        try:
            k = tf.klines.get(sym, period="1d", count=40, adjust="none", as_dataframe=True)
            return k
        except Exception:
            if i < attempts - 1:
                time.sleep((2, 5, 10)[i])
    return None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols-file", default=None,
                    help="targeted pass: newline-separated symbols to (re)fetch instead of all seed-date symbols")
    args = ap.parse_args()

    panel = pd.read_parquet(PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])

    seed = panel[panel["trade_date"] == SEED_DATE]
    if args.symbols_file:
        symbols = sorted(set(Path(args.symbols_file).read_text().split()))
    else:
        symbols = sorted(seed["symbol"].astype(str).unique())
    have = set(map(tuple, panel[(panel["trade_date"] >= WIN_START) & (panel["trade_date"] <= WIN_END)]
                   [["symbol", "trade_date"]].astype({"symbol": str}).itertuples(index=False, name=None)))
    print(f"symbols on seed date: {len(symbols)}; existing fresh rows: {len(have):,}", flush=True)

    st_map: dict[str, bool] = {}
    if ST_FLAGS.exists():
        st = pd.read_parquet(ST_FLAGS)
        st_map = dict(zip(st["symbol"].astype(str), st["is_st"].astype(bool)))

    tf = _tf_client()
    rows, failed = [], []
    for i, sym in enumerate(symbols):
        k = fetch_with_retry(tf, sym)
        if k is None:
            failed.append(sym)
        elif len(k):
            k = k.copy()
            k["symbol"] = sym
            k["trade_date"] = pd.to_datetime(k["trade_date"])
            k = k[(k["trade_date"] >= WIN_START) & (k["trade_date"] <= WIN_END)]
            k = k[[ (sym, d) not in have for d in k["trade_date"] ]]
            if len(k):
                k["volume"] = pd.to_numeric(k["volume"], errors="coerce") * 100.0  # lots -> shares
                rows.append(k[["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]])
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(symbols)} fetched (failed {len(failed)})", flush=True)

    if not rows:
        print(json.dumps({"repaired_rows": 0, "failed": len(failed)}))
        return 0
    new = pd.concat(rows, ignore_index=True)
    for c in ("open", "high", "low", "close", "volume", "amount"):
        new[c] = pd.to_numeric(new[c], errors="coerce")
    new = new.dropna(subset=["trade_date", "symbol", "close"])
    new["is_st"] = new["symbol"].map(lambda s: bool(st_map.get(str(s), False)))
    new["is_st_provenance"] = "current_snapshot_broadcast"
    new["available_at"] = new["trade_date"] + pd.Timedelta(days=1)
    new["source"] = "tickflow_daily_append_repair_20260704"
    new["source_type"] = "vendor_api"
    new["source_reliability"] = 0.9
    new["point_in_time_valid"] = True
    for c in panel.columns:
        if c not in new.columns:
            new[c] = np.nan
    new = new[list(panel.columns)]

    backup = PANEL.with_name("market_panel.pre_repair_20260704.tail.parquet")
    if not backup.exists():
        panel[panel["trade_date"] >= SEED_DATE - pd.Timedelta(days=5)].to_parquet(backup, index=False)

    merged = pd.concat([panel, new], ignore_index=True)
    merged = merged.drop_duplicates(["symbol", "trade_date"], keep="first")

    # ---- recompute flags for the ENTIRE fresh window from the completed chain
    win_mask = (merged["trade_date"] >= WIN_START) & (merged["trade_date"] <= WIN_END)
    chain = merged[(merged["trade_date"] >= SEED_DATE) & (merged["trade_date"] <= WIN_END)][
        ["symbol", "trade_date", "close", "volume", "is_st"]].sort_values(["symbol", "trade_date"]).copy()
    chain["prev_close"] = chain.groupby("symbol")["close"].shift(1)
    chain["is_st"] = chain["symbol"].map(lambda s: bool(st_map.get(str(s), False)))
    caps = chain.apply(lambda r: _board_cap(r["symbol"], bool(r["is_st"])), axis=1)
    up_px = (chain["prev_close"] * (1 + caps)).round(2)
    dn_px = (chain["prev_close"] * (1 - caps)).round(2)
    chain["is_limit_up"] = (chain["close"].round(2) >= up_px - 0.005) & chain["prev_close"].notna()
    chain["is_limit_down"] = (chain["close"].round(2) <= dn_px + 0.005) & chain["prev_close"].notna()
    chain["is_suspended"] = chain["volume"].fillna(0) <= 0
    flags = chain[chain["trade_date"] >= WIN_START].set_index(["symbol", "trade_date"])[
        ["is_limit_up", "is_limit_down", "is_suspended", "is_st"]]
    idx = merged.loc[win_mask].set_index(["symbol", "trade_date"]).index
    for col in ("is_limit_up", "is_limit_down", "is_suspended", "is_st"):
        merged.loc[win_mask, col] = flags[col].reindex(idx).fillna(False).to_numpy()

    merged = merged.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    merged.to_parquet(PANEL, index=False)

    win = merged[win_mask]
    report = {
        "repaired_rows_inserted": int(len(new)),
        "failed_symbols": failed[:50],
        "n_failed": len(failed),
        "fresh_days": int(win["trade_date"].nunique()),
        "coverage_min": int(win.groupby("trade_date")["symbol"].nunique().min()),
        "coverage_max": int(win.groupby("trade_date")["symbol"].nunique().max()),
        "has_20260519": bool((win["trade_date"] == WIN_START).any()),
        "limit_up_rate_mean": round(float(win.groupby("trade_date")["is_limit_up"].mean().mean()), 4),
    }
    print(json.dumps(report, ensure_ascii=False))
    Path("runtime/logs/repair_fresh_window_20260704.report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
