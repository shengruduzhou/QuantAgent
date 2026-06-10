#!/usr/bin/env python3
"""Fit + validate score calibration / conformal uncertainty on REAL v8 predictions (②).

Forward-safe: fit the calibrator on an EARLY window, apply + validate on a LATER window.
Validation = reliability: bin p_beat into deciles and compare to the realized
beat-the-universe rate (calibrated ⇒ they line up; reports an ECE-style gap). Emits a
calibrated predictions table + the latest-date per-symbol conformal_width for RiskGate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.ensemble.calibration import fit_calibrator

PRED = "runtime/reports/v8/deep/v8_full_v3_20260602_051048/short_5d/predictions.parquet"
CORE = "runtime/data/v7/gold/training_dataset/training_dataset_core30.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default=PRED)
    ap.add_argument("--calib-end", default="2025-06-30", help="fit on data <= this; validate after")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out-dir", default="runtime/reports/v8/calibration")
    args = ap.parse_args()

    p = pd.read_parquet(args.predictions)
    p["trade_date"] = pd.to_datetime(p["trade_date"])
    fwd = pd.read_parquet(CORE, columns=["symbol", "trade_date", "forward_return_5d"])
    fwd["trade_date"] = pd.to_datetime(fwd["trade_date"])
    df = p.merge(fwd, on=["symbol", "trade_date"], how="inner").dropna(subset=["alpha_score", "forward_return_5d"])

    cut = pd.Timestamp(args.calib_end)
    calib, test = df[df["trade_date"] <= cut], df[df["trade_date"] > cut]
    if calib.empty or test.empty:
        raise SystemExit("need data on both sides of --calib-end")
    cal = fit_calibrator(calib, alpha=args.alpha)
    out = cal.apply(test)

    # reliability: realized beat-rate per p_beat decile
    out["beat"] = (pd.to_numeric(out["forward_return_5d"], errors="coerce")
                   > out.groupby("trade_date")["forward_return_5d"].transform("mean")).astype(int)
    out["_pb"] = pd.cut(out["p_beat"], np.linspace(0, 1, 11), include_lowest=True)
    rel = out.groupby("_pb", observed=True).agg(pred=("p_beat", "mean"), realized=("beat", "mean"),
                                                n=("beat", "size")).dropna()
    ece = float((rel["pred"] - rel["realized"]).abs().mul(rel["n"]).sum() / rel["n"].sum())
    # does p_beat sort returns? top-decile vs bottom-decile realized beat rate
    lift = float(rel["realized"].iloc[-1] - rel["realized"].iloc[0]) if len(rel) >= 2 else float("nan")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    last_day = out["trade_date"].max()
    cw = out[out["trade_date"] == last_day].set_index("symbol")["conformal_width"]
    cw.to_frame("conformal_width").to_parquet(out_dir / "conformal_width_latest.parquet")
    out[["trade_date", "symbol", "alpha_score", "calib_rank", "p_beat", "conformal_width", "uncertainty"]] \
        .to_parquet(out_dir / "calibrated_predictions.parquet", index=False)
    summary = {
        "calib_end": args.calib_end, "alpha": args.alpha,
        "test_window": [str(test["trade_date"].min().date()), str(test["trade_date"].max().date())],
        "reliability_ece": round(ece, 4),
        "top_vs_bottom_decile_beat_lift": round(lift, 4),
        "bucket_conformal_width": [round(float(x), 5) for x in cal.bucket_width],
        "conformal_uncertainty_threshold_hint": round(float(np.median(cal.bucket_width)), 5),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nreliability (p_beat decile → realized beat rate):")
    print(rel.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
