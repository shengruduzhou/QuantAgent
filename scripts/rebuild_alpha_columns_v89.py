#!/usr/bin/env python3
"""Rebuild the stale alpha101-native columns in the training dataset (→ v8.9).

Root cause (2026-06-14, diagnosed): the v8.8 dataset's alpha101-native columns
(alpha001..alpha101, e.g. the 11 cross-sectional-rank factors 003/013/015/016/
027/042/050/061/062/065/095) were materialized by an OLDER code path and do
NOT match the current `compute_alpha101` (hand-verified correct vs WorldQuant:
alpha003 diff=0.0). Universe was ruled out (full-silver / nosynth / gold all
give ~0.83). gtja/synth columns are already fresh (build_v88_dataset recomputes
them), so only the alphaNNN block is stale.

This script recomputes the alpha101-native block over the FULL silver panel
(the builder's ranking universe) with the current correct code and overwrites
those columns on the dataset keys, producing a fully self-consistent dataset
for the v8.9 retrain. Non-alpha columns (gtja/synth/idx/base/labels/flags) are
preserved byte-for-byte.

--smoke recomputes only a date slice and reports reproducibility (should be
1.0 by construction since train + future inference then share one code path).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SRC = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v88.parquet"
OUT = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89.parquet"


def _native_alpha_cols(columns) -> list[str]:
    out = []
    for c in columns:
        m = re.fullmatch(r"alpha(\d+)", str(c))
        if m and int(m.group(1)) <= 101:
            out.append(c)
    return sorted(out, key=lambda c: int(c.removeprefix("alpha")))


def _recompute(panel: pd.DataFrame, names: list[str], workers: int = 1) -> pd.DataFrame:
    from quantagent.factors.alpha101 import compute_alpha101
    wide = compute_alpha101(panel, names=names, wide=True, workers=workers)
    wide["trade_date"] = pd.to_datetime(wide["trade_date"])
    wide["symbol"] = wide["symbol"].astype(str)
    return wide


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--warmup-days", type=int, default=400)
    ap.add_argument("--workers", type=int, default=1,
                    help="factor-level parallel workers for compute_alpha101 (1 = serial)")
    ap.add_argument("--smoke", action="store_true", help="recompute one slice + report reproducibility")
    ap.add_argument("--smoke-date", default="2026-04-30")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    cols = pq.ParquetFile(args.src).schema_arrow.names
    alpha_cols = _native_alpha_cols(cols)
    print(f"alpha101-native columns to rebuild: {len(alpha_cols)}", flush=True)

    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "high",
                                            "low", "close", "volume", "amount"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["symbol"] = panel["symbol"].astype(str)

    if args.smoke:
        D = pd.Timestamp(args.smoke_date)
        lo = D - pd.Timedelta(days=args.warmup_days)
        sl = panel[(panel["trade_date"] >= lo) & (panel["trade_date"] <= D)]
        names = ["alpha003", "alpha013", "alpha050", "alpha016", "alpha095"]
        fresh = _recompute(sl, names, workers=args.workers)
        old = pd.read_parquet(args.src, columns=["symbol", "trade_date", *names],
                              filters=[("trade_date", ">=", D), ("trade_date", "<=", D)])
        old["trade_date"] = pd.to_datetime(old["trade_date"]); old["symbol"] = old["symbol"].astype(str)
        ff = fresh[fresh["trade_date"] == D]
        # self-consistency: recompute the SAME slice twice → must be identical
        fresh2 = _recompute(sl, names, workers=args.workers)
        ff2 = fresh2[fresh2["trade_date"] == D]
        j2 = ff.merge(ff2, on="symbol", suffixes=("_a", "_b"))
        print("--- self-consistency (recompute twice) ---")
        for c in names:
            d = float((j2[c + "_a"] - j2[c + "_b"]).abs().max())
            print(f"  {c}: max|diff| {d:.2e} {'OK' if d < 1e-9 else 'FAIL'}")
        j = old.merge(ff, on="symbol", suffixes=("_old", "_new"))
        print("--- vs stale v8.8 (expected LOW: that's the bug being fixed) ---")
        for c in names:
            sub = j[[c + "_old", c + "_new"]].dropna()
            corr = sub[c + "_old"].corr(sub[c + "_new"], method="spearman") if len(sub) > 30 else np.nan
            print(f"  {c}: spearman_vs_stale {corr:.4f} n={len(sub)}")
        print("\nv8.9 train + forward inference will BOTH use this code → reproducibility 1.0 by construction.")
        return 0

    # full rebuild
    print(f"recomputing alpha101-native over full silver panel (workers={args.workers}) ...", flush=True)
    fresh = _recompute(panel, alpha_cols, workers=args.workers)
    fresh = fresh[["symbol", "trade_date", *alpha_cols]]
    for c in alpha_cols:
        fresh[c] = fresh[c].astype("float32")

    print(f"loading source dataset {args.src} ...", flush=True)
    df = pd.read_parquet(args.src)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["symbol"] = df["symbol"].astype(str)
    df = df.drop(columns=[c for c in alpha_cols if c in df.columns])
    df = df.merge(fresh, on=["symbol", "trade_date"], how="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    schema = {"source": args.src, "rebuilt_alpha_native_cols": len(alpha_cols),
              "cols": alpha_cols, "rows": int(len(df)),
              "coverage_sample": {c: round(float(df[c].notna().mean()), 3) for c in alpha_cols[:6]},
              "note": "v8.9 retrain: --feature-policy judgment; alpha block now matches compute_alpha101 (correct)"}
    Path(str(out).replace(".parquet", "_schema.json")).write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(schema, ensure_ascii=False, indent=2)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
