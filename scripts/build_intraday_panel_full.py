#!/usr/bin/env python3
# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): input builder for the rejected intraday family.
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""Stage 2.5 (core): full per-minute causal panel for the HELD name-days only.

The 做T overlay only acts on held positions, so we only need per-minute causal
features for the (symbol, trade_date) pairs actually held by w210_k10 / w111_k5
in the OOS + 2026 windows that also have minute bars (~50% coverage). This is
far cheaper than the full 675 x all-days panel and is what stage3b consumes.

Causal only: features at minute m use information up to m (no future high/low/
VWAP/volume). Output -> intraday_panel_675.parquet.
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "../src"))
from quantagent.execution.intraday_features import build_causal_intraday_feature_frame

MIN_DIR = "runtime/data/v7/silver/minute_bars"
S1 = "runtime/reports/v89_closed_loop/stage1"
OUT = Path("runtime/data/v7/silver/intraday_2026"); OUT.mkdir(parents=True, exist_ok=True)
KEEP = ["symbol", "trade_date", "trade_time", "open", "close", "volume", "amount",
        "price_vs_vwap_z", "intraday_return",
        "rolling_return_5m", "rolling_volatility_10m", "distance_to_high_of_day", "distance_to_low_of_day"]
WIN_A, WIN_B = "2025-09-01", "2026-05-13"


def main() -> int:
    have = {os.path.basename(f).replace(".parquet", "") for f in glob.glob(MIN_DIR + "/*.parquet")}
    held = {}  # symbol -> set(dates)
    for book in ("w210_k10", "w111_k5"):
        pos = pd.read_parquet(f"{S1}/daily_{book}_positions.parquet")
        pos["trade_date"] = pd.to_datetime(pos["trade_date"])
        pos = pos[(pos["trade_date"] >= pd.Timestamp(WIN_A)) & (pos["trade_date"] <= pd.Timestamp(WIN_B))]
        for s, d in zip(pos["symbol"].astype(str), pos["trade_date"]):
            if s in have:
                held.setdefault(s, set()).add(d.normalize())
    print(f"held covered names={len(held)}  total name-days={sum(len(v) for v in held.values())}", flush=True)

    frames = []
    for i, (sym, dates) in enumerate(held.items()):
        try:
            b = pd.read_parquet(MIN_DIR + f"/{sym}.parquet",
                                columns=["symbol", "trade_time", "trade_date", "open", "high", "low", "close", "volume", "amount"])
        except Exception:
            continue
        b["trade_date"] = pd.to_datetime(b["trade_date"], errors="coerce").dt.normalize()
        b = b[b["trade_date"].isin(dates)]
        if b.empty:
            continue
        feat = build_causal_intraday_feature_frame(b, include_level2=False)
        if feat.empty:
            continue
        feat["trade_date"] = pd.to_datetime(feat["trade_date"]).dt.normalize()
        frames.append(feat[[c for c in KEEP if c in feat.columns]])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(held)} names done (rows so far ~{sum(len(f) for f in frames):,})", flush=True)
    if not frames:
        print("no per-minute features built"); return 1
    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(OUT / "intraday_panel_675.parquet", index=False)
    print(f"wrote {OUT/'intraday_panel_675.parquet'}: rows={len(full):,} "
          f"symbols={full['symbol'].nunique()} dates={full['trade_date'].nunique()} cols={list(full.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
