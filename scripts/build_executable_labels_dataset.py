#!/usr/bin/env python3
"""Build the v8.7 training dataset with EXECUTABLE labels (memory-efficient).

Root cause being fixed: the v8 models were trained on close(t)->close(t+h)
labels over every row, so they learned to top-rank names whose day-t close
was a sealed limit-up (unbuyable; 5.2% of top-50 picks carrying ~13x the
next-day return of the rest) and ST names the strategy's own risk gate
rejects into cash. That alpha is phantom — it cannot be executed.

Changes vs the source dataset:

1. Every horizon h is re-labelled as the DELAY-1 executable return
       forward_return_{h}d := close(t+1+h) / close(t+1) - 1
   i.e. what the strict backtest (variant C) actually measures.
2. Rows with infeasible/gated ENTRY are dropped:
       is_st(t) | is_suspended(t) | is_suspended(t+1) | is_limit_up(t+1)
   so the model learns its cross-sectional ranking over exactly the pool
   it is allowed to pick from at inference time.

Implementation note: labels are computed on a 6-column slice and merged
back; loading the full 246-column frame and group-shifting it in place
OOM-killed a 64GB box.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HORIZONS = (1, 5, 20, 60, 120, 126)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        default="runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet",
    )
    ap.add_argument("--output", default=None)
    ap.add_argument("--panel", default="runtime/data/v7/silver/market_panel/market_panel.parquet")
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    args = ap.parse_args()

    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    out_path = Path(
        args.output
        or str(args.input).replace(".parquet", "").replace("_full_nosynth", "") + "_exec_v87.parquet"
    )

    # --- pass 1: labels + entry mask on a thin slice -------------------------
    # CRITICAL: tradability flags must come from the MARKET PANEL. The
    # training datasets carry stale all-False is_st/is_suspended/is_limit_up
    # columns (verified 2026-06-11: dataset rates 0.0 vs panel 6.6%/3.8%/1.1%).
    thin = pd.read_parquet(args.input, columns=["symbol", "trade_date", "close"])
    n0 = len(thin)
    thin["trade_date"] = pd.to_datetime(thin["trade_date"], errors="coerce")
    flags = pd.read_parquet(
        args.panel, columns=["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up", "is_limit_down"]
    )
    flags["trade_date"] = pd.to_datetime(flags["trade_date"], errors="coerce")
    thin = thin.merge(flags.drop(columns=["is_limit_down"]), on=["symbol", "trade_date"], how="left")
    thin = thin.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    g = thin.groupby("symbol", sort=False)

    close_next = g["close"].shift(-1)
    suspended = thin["is_suspended"].fillna(False).astype(bool)
    limit_up = thin["is_limit_up"].fillna(False).astype(bool)
    st = thin["is_st"].fillna(False).astype(bool)
    sym = thin["symbol"]
    suspended_next = suspended.groupby(sym, sort=False).shift(-1).astype("boolean").fillna(True).astype(bool)
    limit_up_next = limit_up.groupby(sym, sort=False).shift(-1).astype("boolean").fillna(True).astype(bool)
    entry_ok = ~(st | suspended | suspended_next | limit_up_next) & np.isfinite(close_next) & (close_next > 0)

    labels = thin[["symbol", "trade_date"]].copy()
    for h in horizons:
        exit_close = g["close"].shift(-(1 + h))
        labels[f"forward_return_{h}d"] = (exit_close / close_next - 1.0).astype("float32")
        labels[f"label_end_{h}d"] = g["trade_date"].shift(-(1 + h))
    labels = labels[entry_ok.values].reset_index(drop=True)
    del thin, g, close_next, exit_close
    label_cols = [f"forward_return_{h}d" for h in horizons]
    labels = labels.dropna(subset=label_cols, how="all").reset_index(drop=True)

    # --- pass 2: features without old labels, inner-join new labels ----------
    all_cols = pq.read_schema(args.input).names
    drop = {f"forward_return_{h}d" for h in HORIZONS} | {f"label_end_{h}d" for h in HORIZONS}
    # Stale flag columns are replaced by the real panel values below.
    drop |= {"is_st", "is_suspended", "is_limit_up", "is_limit_down"}
    feature_cols = [c for c in all_cols if c not in drop]
    features = pd.read_parquet(args.input, columns=feature_cols)
    features["trade_date"] = pd.to_datetime(features["trade_date"], errors="coerce")
    df = features.merge(labels, on=["symbol", "trade_date"], how="inner")
    del features, labels
    df = df.merge(flags, on=["symbol", "trade_date"], how="left")
    for c in ("is_st", "is_suspended", "is_limit_up", "is_limit_down"):
        df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    del flags

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    schema = {
        "source": str(args.input),
        "label_definition": "forward_return_{h}d = close(t+1+h)/close(t+1) - 1 (delay-1 executable)",
        "entry_filter": "dropped rows with is_st(t) | is_suspended(t) | is_suspended(t+1) | is_limit_up(t+1)",
        "horizons": horizons,
        "rows_input": int(n0),
        "rows_output": int(len(df)),
        "rows_dropped_pct": round(100.0 * (n0 - len(df)) / max(1, n0), 2),
        "dates": [str(df["trade_date"].min().date()), str(df["trade_date"].max().date())],
        "symbols": int(df["symbol"].nunique()),
        "output": str(out_path),
    }
    Path(str(out_path).replace(".parquet", "_label_schema.json")).write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(schema, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
