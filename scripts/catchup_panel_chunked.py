#!/usr/bin/env python3
"""Chunked, RESUMABLE panel catch-up (H-029 activation repair, 2026-07-19).

Why: the monolithic updater persists only at completion; under degraded
TickFlow throttling (~2s/call, measured 2026-07-18) a full 3.6k-symbol pass
takes ~2h+ and a timeout loses everything (CATCHUP_FAILED rc=124 precedent).

Design: fetch in chunks of 250 symbols; each chunk is staged immediately to
_staging_catchup/chunk_*.parquet; a restart skips symbols already staged
(idempotent); when all chunks are staged the panel is merged ONCE, flags are
rebuilt over the affected window with the proven repair semantics
(board-aware caps, volume lots->shares, surgical insert, dedup keep-first),
and staging is cleared. Tail backup written before the merge.

Usage: AI_quant_venv/bin/python3 scripts/catchup_panel_chunked.py [--end YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import repair_fresh_window_20260704 as rep  # noqa: E402  (proven helpers)

PANEL = REPO / "runtime/data/v7/silver/market_panel/market_panel.parquet"
ST_FLAGS = REPO / "runtime/data/v7/silver/st_flags/st_flags.parquet"
STAGING = PANEL.parent / "_staging_catchup"
CHUNK = 250


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--end", default=None, help="last date to fetch (default: today)")
    args = ap.parse_args()
    t0 = time.time()
    STAGING.mkdir(exist_ok=True)

    panel = pd.read_parquet(PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    pmax = panel["trade_date"].max()
    win_start = pmax + pd.Timedelta(days=1)
    win_end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    # AVAILABILITY CLAMP (incident 2026-07-21): TickFlow serves an IN-PROGRESS
    # daily bar during the session. A multi-hour pass that starts before 15:00
    # CST therefore stages partial bars (measured: 000001.SZ volume 124.3M vs
    # true close 175.5M, ratio 0.708) that are invisible to both coverage and
    # aggregate-volume checks. Never request a day whose close is unpublished.
    _cst = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    _last_available = _cst.normalize() if _cst.hour * 60 + _cst.minute >= 15 * 60 + 30 \
        else _cst.normalize() - pd.Timedelta(days=1)
    if win_end > _last_available:
        print(f"clamping window end {win_end.date()} -> {_last_available.date()} "
              f"(close not published yet at {_cst:%Y-%m-%d %H:%M} CST)", flush=True)
        win_end = _last_available
    if win_start > win_end:
        print(json.dumps({"appended": 0, "note": "panel already current"}))
        return 0
    # weekend/holiday early exit: no business day in the window => nothing can exist
    if len(pd.bdate_range(win_start, win_end)) == 0:
        print(json.dumps({"appended": 0, "note": "no business day in window (weekend/holiday)"}))
        return 0
    seed_syms = sorted(panel.loc[panel["trade_date"] == pmax, "symbol"].astype(str).unique())

    # staging is WINDOW-SCOPED: a manifest pins the window; any mismatch clears
    # staging so stale done-markers can never poison a different catch-up window
    manifest = STAGING / "window.json"
    wtag = f"{win_start.date()}_{win_end.date()}"
    if not manifest.exists() or json.loads(manifest.read_text()).get("window") != wtag:
        for f in list(STAGING.glob("chunk_*.parquet")) + list(STAGING.glob("done_*.json")):
            f.unlink()
        manifest.write_text(json.dumps({"window": wtag}))

    done_syms: set[str] = set()
    for f in sorted(STAGING.glob("done_*.json")):
        try:
            done_syms |= set(json.loads(f.read_text()))
        except Exception:
            f.unlink()
    for f in sorted(STAGING.glob("chunk_*.parquet")):
        try:
            done_syms |= set(pd.read_parquet(f, columns=["symbol"])["symbol"].astype(str))
        except Exception:
            f.unlink()  # corrupt partial chunk: refetch it
    todo = [s for s in seed_syms if s not in done_syms]
    print(f"window {win_start.date()}..{win_end.date()} | seed {len(seed_syms)} "
          f"| staged {len(done_syms)} | todo {len(todo)}", flush=True)

    tf = rep._tf_client()
    failed = []
    for ci in range(0, len(todo), CHUNK):
        chunk_syms = todo[ci:ci + CHUNK]
        rows = []
        for sym in chunk_syms:
            k = rep.fetch_with_retry(tf, sym)
            if k is None:
                failed.append(sym)
                continue
            if not len(k):
                continue
            k = k.copy()
            k["symbol"] = sym
            k["trade_date"] = pd.to_datetime(k["trade_date"])
            k = k[(k["trade_date"] >= win_start) & (k["trade_date"] <= win_end)]
            if len(k):
                k["volume"] = pd.to_numeric(k["volume"], errors="coerce") * 100.0  # lots -> shares
                rows.append(k[["symbol", "trade_date", "open", "high", "low", "close",
                               "volume", "amount"]])
        staged = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"])
        # done marker = attempted MINUS failed (failed symbols get retried on restart);
        # sidecar keeps chunk parquets pure OHLCV
        chunk_failed = set(failed) & set(chunk_syms)
        (STAGING / f"done_{ci//CHUNK:04d}.json").write_text(
            json.dumps(sorted(set(chunk_syms) - chunk_failed)))
        staged.to_parquet(STAGING / f"chunk_{ci//CHUNK:04d}_{int(time.time())}.parquet", index=False)
        print(f"  staged {ci + len(chunk_syms)}/{len(todo)} (failed {len(failed)}) "
              f"{time.time()-t0:.0f}s", flush=True)

    # ---- merge once
    chunks = [pd.read_parquet(f) for f in sorted(STAGING.glob("chunk_*.parquet"))]
    new = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    new = new.drop_duplicates(["symbol", "trade_date"], keep="first") if len(new) else new
    if not len(new):
        for f in list(STAGING.glob("chunk_*.parquet")) + list(STAGING.glob("done_*.json")):
            f.unlink()  # empty window: clear staging so markers never leak forward
        manifest.unlink(missing_ok=True)
        print(json.dumps({"appended": 0, "n_failed": len(failed), "note": "no new rows fetched"}))
        return 0
    for c in ("open", "high", "low", "close", "volume", "amount"):
        new[c] = pd.to_numeric(new[c], errors="coerce")
    new = new.dropna(subset=["trade_date", "symbol", "close"])
    st_map = {}
    if ST_FLAGS.exists():
        stf = pd.read_parquet(ST_FLAGS)
        scol = "symbol" if "symbol" in stf.columns else stf.columns[0]
        vcol = "is_st" if "is_st" in stf.columns else stf.columns[-1]
        st_map = dict(zip(stf[scol].astype(str), stf[vcol].astype(bool)))
    new["is_st"] = new["symbol"].map(lambda s: bool(st_map.get(str(s), False)))
    new["is_st_provenance"] = "current_snapshot_broadcast"
    new["available_at"] = new["trade_date"] + pd.Timedelta(days=1)
    new["source"] = "tickflow_catchup_chunked"
    new["source_type"] = "vendor_api"
    new["source_reliability"] = 0.9
    new["point_in_time_valid"] = True
    for c in panel.columns:
        if c not in new.columns:
            new[c] = np.nan
    new = new[list(panel.columns)]

    backup = PANEL.with_name(f"market_panel.pre_catchup_{win_start.date()}.tail.parquet")
    if not backup.exists():
        panel[panel["trade_date"] >= pmax - pd.Timedelta(days=5)].to_parquet(backup, index=False)
    merged = pd.concat([panel, new], ignore_index=True)
    merged = merged.drop_duplicates(["symbol", "trade_date"], keep="first")

    # ---- flag rebuild over the appended window (chain from 5d before window)
    win_mask = (merged["trade_date"] >= win_start) & (merged["trade_date"] <= win_end)
    chain = merged[(merged["trade_date"] >= pmax - pd.Timedelta(days=5))
                   & (merged["trade_date"] <= win_end)][
        ["symbol", "trade_date", "close", "volume", "is_st"]].sort_values(
        ["symbol", "trade_date"]).copy()
    chain["prev_close"] = chain.groupby("symbol")["close"].shift(1)
    chain["is_st"] = chain["symbol"].map(lambda s: bool(st_map.get(str(s), False)))
    caps = chain.apply(lambda r: rep._board_cap(r["symbol"], bool(r["is_st"])), axis=1)
    up_px = (chain["prev_close"] * (1 + caps)).round(2)
    dn_px = (chain["prev_close"] * (1 - caps)).round(2)
    chain["is_limit_up"] = (chain["close"].round(2) >= up_px - 0.005) & chain["prev_close"].notna()
    chain["is_limit_down"] = (chain["close"].round(2) <= dn_px + 0.005) & chain["prev_close"].notna()
    chain["is_suspended"] = chain["volume"].fillna(0) <= 0
    flags = chain[chain["trade_date"] >= win_start].set_index(["symbol", "trade_date"])[
        ["is_limit_up", "is_limit_down", "is_suspended", "is_st"]]
    idx = merged.loc[win_mask].set_index(["symbol", "trade_date"]).index
    for col in ("is_limit_up", "is_limit_down", "is_suspended", "is_st"):
        merged.loc[win_mask, col] = flags[col].reindex(idx).fillna(False).to_numpy()

    merged = merged.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    merged.to_parquet(PANEL, index=False)
    for f in list(STAGING.glob("chunk_*.parquet")) + list(STAGING.glob("done_*.json")):
        f.unlink()
    manifest.unlink(missing_ok=True)
    win = merged[win_mask]
    report = {"appended": int(len(new)), "n_failed": len(failed),
              "failed_sample": failed[:20],
              "new_max_date": str(merged["trade_date"].max().date()),
              "dates_added": [str(d.date()) for d in sorted(win["trade_date"].unique())],
              "coverage_per_date": {str(k.date()): int(v) for k, v in
                                    win.groupby("trade_date")["symbol"].nunique().items()},
              "runtime_s": round(time.time() - t0, 1)}
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
