#!/usr/bin/env python3
# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): input builder for the rejected intraday family.
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""Build a CAUSAL 2026 intraday feature panel from the 675-symbol minute bars.

Output 1 (EOD snapshot): per (symbol, trade_date) the last causal-feature row of
the day -> usable as a NEXT-DAY daily signal (no leak: only that day's history).
Output 2 (per-minute): kept for the 做T/timing overlay on held names.

Coverage is limited to the 675 names with 2026 minute bars; this is the only
honest intraday set available for 2026 (rich panel is 2020-21 only).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "../src"))
from quantagent.execution.intraday_features import build_causal_intraday_feature_frame  # noqa: E402

MIN_DIR = "runtime/data/v7/silver/minute_bars"
OUT = Path("runtime/data/v7/silver/intraday_2026")
START, END = "2025-12-15", "2026-05-31"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(MIN_DIR + "/*.parquet"))
    print(f"{len(files)} symbol minute files", flush=True)
    eod_rows = []
    n_ok = 0
    for i, f in enumerate(files):
        try:
            df = pd.read_parquet(f, columns=["symbol", "trade_time", "trade_date", "open", "high", "low", "close", "volume", "amount"])
        except Exception:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df = df[(df["trade_date"] >= pd.Timestamp(START)) & (df["trade_date"] <= pd.Timestamp(END))]
        if df.empty:
            continue
        feat = build_causal_intraday_feature_frame(df, include_level2=False)
        if feat.empty:
            continue
        # EOD snapshot = last minute row per (symbol, trade_date)
        feat["trade_date"] = pd.to_datetime(feat["trade_date"])
        eod = feat.sort_values("trade_time").groupby(["symbol", "trade_date"]).tail(1)
        eod_rows.append(eod)
        n_ok += 1
        if (i + 1) % 100 == 0:
            print(f"  processed {i+1}/{len(files)} (ok={n_ok})", flush=True)
    if not eod_rows:
        print("no intraday features built"); return 1
    eod_all = pd.concat(eod_rows, ignore_index=True)
    keep = [c for c in eod_all.columns if c not in ("trade_time",)]
    eod_all[keep].to_parquet(OUT / "intraday_eod_2026.parquet", index=False)
    print(f"\nwrote {OUT/'intraday_eod_2026.parquet'}: rows={len(eod_all)} "
          f"symbols={eod_all['symbol'].nunique()} dates={eod_all['trade_date'].nunique()}")
    print("feature cols:", [c for c in keep if c not in ('symbol','trade_date')][:25])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
