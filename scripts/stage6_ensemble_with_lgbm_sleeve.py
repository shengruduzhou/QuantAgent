#!/usr/bin/env python3
"""Add the cross-sectional LightGBM as a 4th SLEEVE to the v8.9 ensemble book.

Optimise the BOOK (not rank-IC): blend the 3 v8.9 deep sleeves
(short_5d / mid_5d_30d / long_30d_120d) + a new LightGBM sleeve, cross-sectionally
ranked, with weights tuned by MAX after-cost CAGR on a VALIDATION window and
reported on a never-seen HELD-OUT window via the strict backtest (baseline_protocol
variant-C). Contamination-free (val/held-out split).

Honest controlled comparison: the best 4-sleeve blend (LightGBM weight > 0) vs the
best 3-sleeve blend (LightGBM weight 0) on the SAME overlap rows + same held-out
window. Only if the 4-sleeve held-out CAGR beats the 3-sleeve does the LightGBM
sleeve add tradable value.

Usage:
    AI_quant_venv/bin/python3 scripts/stage6_ensemble_with_lgbm_sleeve.py \
        --sleeve3 runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/ensemble_composite.parquet \
        --lgbm-preds runtime/stage6_classical_walkforward_8fold/wf/walkforward_predictions.parquet \
        --output-dir runtime/stage6_ensemble_lgbm_sleeve
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

SLEEVES = ["short_5d_score", "mid_5d_30d_score", "long_30d_120d_score", "lgbm_score"]


def _ranked(work: pd.DataFrame) -> pd.DataFrame:
    out = work[["trade_date", "symbol"]].copy()
    for col in SLEEVES:
        out[col] = work.groupby("trade_date")[col].rank(pct=True) if col in work.columns else 0.0
    return out


def _cagr(path: str, k: int, start: str, end: str, save_dir: str | None = None) -> dict:
    res = bp.evaluate(path, top_k=k, start=start, end=end, slippage_bps=8.0,
                      variants=["C_flags_eligible_delay1"], score_column="composite_score",
                      save_backtest_dir=save_dir)
    c = res["variants"]["C_flags_eligible_delay1"]
    return {"cagr": c["ann"], "maxDD": c["maxDD"], "sharpe": c["sharpe"],
            "calmar": (c["ann"] / c["maxDD"] if c["maxDD"] else 0.0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sleeve3", required=True)
    ap.add_argument("--lgbm-preds", required=True)
    ap.add_argument("--lgbm-score-col", default="alpha_5d")
    ap.add_argument("--val-start", default="2024-08-28")
    ap.add_argument("--val-end", default="2025-08-31")
    ap.add_argument("--test-start", default="2025-09-01")
    ap.add_argument("--test-end", default="2026-04-30")
    ap.add_argument("--top-k-grid", default="10,20")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    tmp = outdir / "_tmp"; tmp.mkdir(exist_ok=True)

    s3 = pd.read_parquet(args.sleeve3)
    s3["trade_date"] = pd.to_datetime(s3["trade_date"])
    lg = pd.read_parquet(args.lgbm_preds, columns=["symbol", "trade_date", args.lgbm_score_col])
    lg["trade_date"] = pd.to_datetime(lg["trade_date"])
    lg = lg.rename(columns={args.lgbm_score_col: "lgbm_score"})
    # Inner-merge → both books evaluated on IDENTICAL rows (fair comparison).
    work = s3.merge(lg, on=["symbol", "trade_date"], how="inner")
    print(f"overlap rows={len(work):,} dates={work['trade_date'].nunique()} "
          f"span {work['trade_date'].min().date()}->{work['trade_date'].max().date()}", flush=True)
    ranked = _ranked(work)
    top_ks = [int(k) for k in args.top_k_grid.split(",") if k.strip()]

    # (short, mid, long, lgbm). lgbm=0 → 3-sleeve baseline; lgbm>0 → 4-sleeve.
    weight_grid = [
        (1, 1, 0, 0), (1, 1, 1, 0), (2, 1, 0, 0), (1, 2, 0, 0),         # 3-sleeve baselines
        (1, 1, 0, 1), (1, 1, 0, 2), (1, 1, 1, 1), (1, 2, 0, 1),
        (2, 1, 0, 1), (0, 0, 0, 1), (1, 1, 0, 0.5), (1, 1, 1, 2),       # +lgbm sleeve
    ]
    results = []
    for (ws, wm, wl, wg), k in itertools.product(weight_grid, top_ks):
        score = (ws * ranked["short_5d_score"] + wm * ranked["mid_5d_30d_score"]
                 + wl * ranked["long_30d_120d_score"] + wg * ranked["lgbm_score"])
        frame = ranked[["trade_date", "symbol"]].copy(); frame["composite_score"] = score.to_numpy()
        path = tmp / f"w{ws}_{wm}_{wl}_{wg}_k{k}.parquet"; frame.to_parquet(path, index=False)
        val = _cagr(str(path), k, args.val_start, args.val_end)
        results.append({"weights": (ws, wm, wl, wg), "top_k": k, "has_lgbm": wg > 0,
                        "val_cagr": val["cagr"], "val_calmar": val["calmar"], "_path": str(path)})
        print(f"w={ws},{wm},{wl},{wg} k={k}  VAL CAGR {val['cagr']:+.2%} (Calmar {val['calmar']:.2f})", flush=True)

    def _best(rows):
        return max(rows, key=lambda r: r["val_cagr"]) if rows else None
    best_4 = _best([r for r in results if r["has_lgbm"]])
    best_3 = _best([r for r in results if not r["has_lgbm"]])

    out = {"overlap_dates": int(work["trade_date"].nunique()),
           "test_window": [args.test_start, args.test_end], "results": results}
    for tag, best in (("with_lgbm_4sleeve", best_4), ("baseline_3sleeve", best_3)):
        held = _cagr(best["_path"], best["top_k"], args.test_start, args.test_end,
                     save_dir=str(outdir / f"{tag}_heldout"))
        out[tag] = {"weights": list(best["weights"]), "top_k": best["top_k"],
                    "val_cagr": best["val_cagr"], "heldout": held}
        print(f"\n{tag}: weights={best['weights']} k={best['top_k']} "
              f"VAL {best['val_cagr']:+.2%} -> HELD-OUT CAGR {held['cagr']:+.2%} "
              f"(maxDD {held['maxDD']:.2%}, Calmar {held['calmar']:.2f}, Sharpe {held['sharpe']:.2f})", flush=True)

    delta = out["with_lgbm_4sleeve"]["heldout"]["cagr"] - out["baseline_3sleeve"]["heldout"]["cagr"]
    out["lgbm_sleeve_heldout_cagr_delta"] = delta
    out["verdict"] = ("LightGBM sleeve ADDS value (held-out CAGR improves)" if delta > 0
                      else "LightGBM sleeve does NOT help (held-out CAGR no better)")
    (outdir / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n=== VERDICT: {out['verdict']} (Δheld-out CAGR {delta:+.2%}) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
