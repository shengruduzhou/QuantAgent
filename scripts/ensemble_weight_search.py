#!/usr/bin/env python3
"""Structural lever #1: pick sleeve-blend weights + book concentration that
MAXIMIZE absolute CAGR (not Calmar), contamination-free.

Tune on a VALIDATION window, report on a never-seen HELD-OUT window. The
per-sleeve scores (short_5d_score/mid_5d_30d_score/long_30d_120d_score) are
cross-sectionally ranked per date, combined with candidate weights, and run
through the trusted strict backtest (`baseline_protocol`, variant C).

Selection metric = held-out-proxy CAGR on the validation window; the winner is
then re-run on the held-out window and exported as a UI backtest.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baseline_protocol as bp  # noqa: E402


def _ranked_sleeves(preds: pd.DataFrame) -> pd.DataFrame:
    out = preds[["trade_date", "symbol"]].copy()
    for col in ("short_5d_score", "mid_5d_30d_score", "long_30d_120d_score"):
        if col in preds.columns:
            out[col] = preds.groupby("trade_date")[col].rank(pct=True)
        else:
            out[col] = 0.0
    return out


def _cagr(preds_path: str, top_k: int, start: str, end: str) -> dict:
    res = bp.evaluate(preds_path, top_k=top_k, start=start, end=end, slippage_bps=8.0,
                      variants=["C_flags_eligible_delay1"], score_column="composite_score")
    c = res["variants"]["C_flags_eligible_delay1"]
    return {"cagr": c["ann"], "maxdd": c["maxDD"], "sharpe": c["sharpe"],
            "calmar": (c["ann"] / c["maxDD"] if c["maxDD"] else 0.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True, help="ensemble_composite.parquet with per-sleeve scores")
    ap.add_argument("--val-start", default="2024-08-28")
    ap.add_argument("--val-end", default="2025-08-31")
    # No defaults: the old 2025-09-01+ defaults silently consumed the (now
    # quarantined) holdout. bp.evaluate() fails closed on quarantined windows.
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--test-end", required=True)
    ap.add_argument("--top-k-grid", default="10,20,30")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    tmpdir = outdir / "_tmp"; tmpdir.mkdir(exist_ok=True)
    preds = pd.read_parquet(args.predictions)
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    ranked = _ranked_sleeves(preds)
    top_ks = [int(k) for k in args.top_k_grid.split(",") if k.strip()]

    # weight grid over (short, mid, long); 0 drops a sleeve.
    weight_grid = [
        (1, 1, 1), (1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0),
        (1, 2, 0), (1, 1, 0.5), (2, 1, 1), (1, 2, 1),
    ]

    results = []
    for (ws, wm, wl), k in itertools.product(weight_grid, top_ks):
        score = ws * ranked["short_5d_score"] + wm * ranked["mid_5d_30d_score"] + wl * ranked["long_30d_120d_score"]
        frame = ranked[["trade_date", "symbol"]].copy()
        frame["composite_score"] = score.to_numpy()
        tmp = tmpdir / f"w{ws}_{wm}_{wl}_k{k}.parquet"
        frame.to_parquet(tmp, index=False)
        val = _cagr(str(tmp), k, args.val_start, args.val_end)
        results.append({"weights": (ws, wm, wl), "top_k": k, "val": val})
        print(f"w={ws},{wm},{wl} k={k}  VAL CAGR {val['cagr']:+.2%} (Calmar {val['calmar']:.2f})", flush=True)

    # SELECT BY MAX CAGR on validation (user goal = max absolute return).
    best = max(results, key=lambda r: r["val"]["cagr"])
    ws, wm, wl = best["weights"]; k = best["top_k"]
    print(f"\n>>> best-by-VAL-CAGR: weights={best['weights']} top_k={k} val_CAGR={best['val']['cagr']:+.2%}", flush=True)

    # Re-run winner on the HELD-OUT window (honest report) + export UI backtest.
    score = ws * ranked["short_5d_score"] + wm * ranked["mid_5d_30d_score"] + wl * ranked["long_30d_120d_score"]
    frame = ranked[["trade_date", "symbol"]].copy(); frame["composite_score"] = score.to_numpy()
    win_path = outdir / "winner_predictions.parquet"; frame.to_parquet(win_path, index=False)
    test = bp.evaluate(str(win_path), top_k=k, start=args.test_start, end=args.test_end, slippage_bps=8.0,
                       variants=["C_flags_eligible_delay1"], score_column="composite_score",
                       save_backtest_dir=str(outdir / "winner_heldout"))
    tc = test["variants"]["C_flags_eligible_delay1"]
    # equal-weight composite baseline on held-out for comparison
    base = _cagr(args.predictions, k, args.test_start, args.test_end)
    summary = {
        "best_weights": list(best["weights"]), "best_top_k": k,
        "val_cagr": best["val"]["cagr"],
        "heldout": {"cagr": tc["ann"], "maxDD": tc["maxDD"], "sharpe": tc["sharpe"],
                    "calmar": (tc["ann"] / tc["maxDD"] if tc["maxDD"] else 0.0)},
        "heldout_equalweight_composite": base,
        "all_results": results,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nHELD-OUT winner: CAGR {tc['ann']:+.2%} maxDD {tc['maxDD']:.2%} (vs equal-wt composite CAGR {base['cagr']:+.2%})")
    print(f"wrote {outdir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
