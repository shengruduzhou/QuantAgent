#!/usr/bin/env python3
"""H-029 Stage 3: root-cause isolation for the drifting feature columns.

Tests (machine-readable evidence -> stage3_results.json):
  T1 sleeve membership of the drift columns (explains short/mid vs long)
  T2 NaN-mask asymmetry per drift column
  T3 warmup sensitivity: recompute drift columns with 630d (vs 420d) warmup
  T4 HYBRID: forward features with top-k drift columns replaced by gold
     values -> sleeve fidelity; k in {0 (baseline), 4, 6, all>thresh}
Golden window + population identical to stage 2 (preregistered 1b47c3d).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
from forward_daily_inference import _sleeve_features, build_feature_frame, rank_normalize  # noqa: E402
from exp029_fidelity import (  # noqa: E402
    GOLD, PANEL, QUAR, RUN, SLEEVES, W0, W1, daily_spear, predict_frame, score_vs_stored,
)

OUT = REPO / "runtime/reports/h029"
DRIFT = ["alpha045", "gtja032", "synth_007_3_046", "gtja036", "synth_001_2_020", "synth_002_001"]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()
    sleeve_feats = _sleeve_features(RUN)
    union = sorted({c for cols in sleeve_feats.values() for c in cols})
    res = {"T1_membership": {c: [sl for sl in SLEEVES if c in sleeve_feats[sl]] for c in DRIFT}}
    print(json.dumps(res["T1_membership"], indent=1), flush=True)

    gold = pd.read_parquet(GOLD, columns=["symbol", "trade_date"] + union,
                           filters=[("trade_date", ">=", W0), ("trade_date", "<=", W1)])
    gold["trade_date"] = pd.to_datetime(gold["trade_date"])
    assert gold["trade_date"].max() < QUAR
    stored = {}
    for sl in SLEEVES:
        s = pd.read_parquet(RUN / sl / "predictions.parquet")
        s["trade_date"] = pd.to_datetime(s["trade_date"])
        stored[sl] = s[(s["trade_date"] >= W0) & (s["trade_date"] <= W1)]

    cache = OUT / "fwd_raw_cache.parquet"
    if cache.exists():
        fwd_raw = pd.read_parquet(cache)
        fwd_raw["trade_date"] = pd.to_datetime(fwd_raw["trade_date"])
    else:
        panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "high", "low",
                                                "close", "volume", "amount"],
                                filters=[("trade_date", ">=", W0 - pd.Timedelta(days=420)),
                                         ("trade_date", "<=", W1)])
        panel["trade_date"] = pd.to_datetime(panel["trade_date"])
        target = pd.DatetimeIndex(sorted(gold["trade_date"].unique()))
        active = panel.loc[panel["trade_date"].isin(target) & panel["close"].gt(0), "symbol"].unique()
        fwd_raw = build_feature_frame(panel[panel["symbol"].isin(active)].copy(), union, target)
        fwd_raw.to_parquet(cache, index=False)
    inter = gold.merge(fwd_raw, on=["symbol", "trade_date"], suffixes=("_g", "_f"), how="inner")

    # T2 NaN asymmetry
    res["T2_nan_asymmetry"] = {
        c: {"nan_gold_only": int((inter[f"{c}_g"].isna() & inter[f"{c}_f"].notna()).sum()),
            "nan_fwd_only": int((inter[f"{c}_g"].notna() & inter[f"{c}_f"].isna()).sum())}
        for c in DRIFT}
    print("T2:", json.dumps(res["T2_nan_asymmetry"]), flush=True)

    # T3 warmup sensitivity (recompute drift columns with 630d warmup)
    panel_long = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "high", "low",
                                                 "close", "volume", "amount"],
                                 filters=[("trade_date", ">=", W0 - pd.Timedelta(days=630)),
                                          ("trade_date", "<=", W1)])
    panel_long["trade_date"] = pd.to_datetime(panel_long["trade_date"])
    target = pd.DatetimeIndex(sorted(gold["trade_date"].unique()))
    active = panel_long.loc[panel_long["trade_date"].isin(target)
                            & panel_long["close"].gt(0), "symbol"].unique()
    fwd_long = build_feature_frame(panel_long[panel_long["symbol"].isin(active)].copy(),
                                   DRIFT, target)
    il = gold[["symbol", "trade_date"] + DRIFT].merge(
        fwd_long, on=["symbol", "trade_date"], suffixes=("_g", "_f"), how="inner")
    res["T3_warmup630"] = {}
    for c in DRIFT:
        sp = daily_spear(il[f"{c}_g"], il[f"{c}_f"], il["trade_date"].to_numpy())
        res["T3_warmup630"][c] = {"daily_median_spearman_630d": round(float(sp.median()), 5)}
    print("T3:", json.dumps(res["T3_warmup630"]), flush=True)

    # T4 hybrid substitution (drift columns from GOLD, rest from forward)
    flags = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "is_st", "is_suspended"])
    flags["trade_date"] = pd.to_datetime(flags["trade_date"])
    fp = fwd_raw.merge(flags, on=["symbol", "trade_date"], how="left")
    bad = fp["is_st"].fillna(False).astype(bool) | fp["is_suspended"].fillna(False).astype(bool)
    fp = fp[~bad].drop(columns=["is_st", "is_suspended"]).copy()
    rows, daily_rows = [], []
    for k, cols in (("k0_baseline", []), ("k4", DRIFT[:4]), ("k6", DRIFT[:6])):
        hyb = fp.copy()
        if cols:
            gsub = gold[["symbol", "trade_date"] + cols]
            hyb = hyb.drop(columns=cols).merge(gsub, on=["symbol", "trade_date"], how="left")
        hn = rank_normalize(hyb, union)
        preds = predict_frame(hn, sleeve_feats, args.device)
        res[f"T4_{k}"] = score_vs_stored(preds, stored, f"T4_{k}", rows, daily_rows)
    (OUT / "stage3_results.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(daily_rows).to_csv(OUT / "stage3_daily.csv", index=False)
    print(f"done {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
