#!/usr/bin/env python3
"""Augment the v8 training dataset with evidence + tickflow growth-financial features.

v8.5 feature augmentation (the 20→30 lever): join onto the alpha181 training dataset
  (a) the 5 evidence signals from core30 that have REAL cross-sectional variation over
      2018-2026 (core_policy_score / fundamental_quality_score / sector_resonance_score /
      old_dealer_risk_score / trend_strength_score; sentiment is empty hist, dip is sparse → skipped),
  (b) tickflow PIT growth financials (roe / gross_margin / net_margin / revenue_yoy /
      net_income_yoy) from build_tickflow_fin_features.py.
The new column names match the (now-extended) DEFAULT_SHORT_FEATURE_PATTERNS so
``train-v8-deep`` picks them up. Output is a drop-in replacement training dataset.

  python scripts/augment_training_dataset.py
  → runtime/data/v7/gold/training_dataset/training_dataset_alpha181_aug_v85.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BASE = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet"
CORE30 = "runtime/data/v7/gold/training_dataset/training_dataset_core30.parquet"
FIN = "runtime/data/v7/gold/training_dataset/tickflow_fin_features.parquet"
OUT = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_aug_v85.parquet"

EVIDENCE = ["core_policy_score", "fundamental_quality_score", "sector_resonance_score",
            "old_dealer_risk_score", "trend_strength_score"]
FINCOLS = ["roe", "gross_margin", "net_margin", "revenue_yoy", "net_income_yoy"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--core30", default=CORE30)
    ap.add_argument("--fin", default=FIN)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    df = pd.read_parquet(args.base)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    n0, c0 = df.shape
    print(f"base: {n0} rows, {c0} cols", flush=True)

    ev = pd.read_parquet(args.core30, columns=["symbol", "trade_date"] + EVIDENCE)
    ev["trade_date"] = pd.to_datetime(ev["trade_date"])
    df = df.merge(ev, on=["symbol", "trade_date"], how="left")
    print(f"+evidence {EVIDENCE}: now {df.shape[1]} cols", flush=True)

    if Path(args.fin).exists():
        fin = pd.read_parquet(args.fin)
        fin["trade_date"] = pd.to_datetime(fin["trade_date"])
        fin = fin[["symbol", "trade_date"] + [c for c in FINCOLS if c in fin.columns]]
        df = df.merge(fin, on=["symbol", "trade_date"], how="left")
        cov = {c: round(float(df[c].notna().mean()), 3) for c in FINCOLS if c in df.columns}
        print(f"+financials {FINCOLS}: now {df.shape[1]} cols; coverage={cov}", flush=True)
    else:
        print(f"WARN: {args.fin} not found — financial features skipped (run build_tickflow_fin_features.py)")

    assert len(df) == n0, f"row count changed {n0}->{len(df)} (merge fan-out!)"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {args.out}: {df.shape[0]} rows, {df.shape[1]} cols "
          f"(+{df.shape[1]-c0} new features)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
