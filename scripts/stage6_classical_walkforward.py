#!/usr/bin/env python3
"""Stage 6 feature/model fix: cross-sectional-feature LightGBM walk-forward.

Identical dataset / window / pinned 242-feature schema / folds as the deep-MLP
run (scripts/stage6_full_walkforward.py), so the ONLY difference is:
  * features transformed per-day cross-sectionally (rank), and
  * model = LightGBM (NaN-native, no complete-case attrition) instead of the
    globally-standardised MLP.
This isolates the hypothesis: does the global-standardisation flaw explain the
MLP's ~0 rank-IC?  Emits OOS predictions in the standard shape → feed straight
into stage6_policy_search.py / baseline_protocol.py / walk_forward_eval.

Usage:
    AI_quant_venv/bin/python3 scripts/stage6_classical_walkforward.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quantagent.training.splitters import WalkForwardSplitConfig
from quantagent.training.walk_forward_classical import ClassicalWFConfig, run_walk_forward_classical
from quantagent.training.walk_forward_eval import evaluate_walk_forward_oos

GOLD = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean.parquet"
SCHEMA = "runtime/stage6_full_walkforward/feature_schema.json"   # the SAME pinned 242-feature contract
OUT = Path("runtime/stage6_classical_walkforward")
START = "2020-10-01"
HORIZONS = (1, 5, 20)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--cross-sectional", default="rank")
    ap.add_argument("--label-transform", default="raw")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-splits", type=int, default=6)
    ap.add_argument("--start", default=START)
    ap.add_argument("--min-train-days", type=int, default=500)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = json.loads(Path(SCHEMA).read_text())
    feats = schema["feature_columns"]
    label_cols = [f"forward_return_{h}d" for h in HORIZONS]
    keep = ["symbol", "trade_date", *label_cols, *feats]

    df = pd.read_parquet(GOLD, columns=keep)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df[df["trade_date"] >= args.start].reset_index(drop=True)
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce").astype("float32")
    print(f"[dataset] rows={len(df):,} symbols={df['symbol'].nunique()} dates={df['trade_date'].nunique()} "
          f"features={len(feats)} schema_hash={schema['schema_hash'][:12]}", flush=True)

    cfg = ClassicalWFConfig(
        horizons=HORIZONS, model=args.model, cross_sectional=args.cross_sectional,
        label_transform=args.label_transform, feature_version=f"{args.model}-csrank@cov099",
    )
    split = WalkForwardSplitConfig(mode="purged", n_splits=args.n_splits, min_train_days=args.min_train_days,
                                   valid_size_days=120, embargo_days=5, purge_days=20)
    print(f"[run] model={args.model} cross_sectional={args.cross_sectional} label={args.label_transform}", flush=True)
    res = run_walk_forward_classical(df, feature_schema_path=SCHEMA, config=cfg,
                                     split_config=split, output_dir=str(out_dir / "wf"))

    labels = df[["symbol", "trade_date", *label_cols]]
    ev = evaluate_walk_forward_oos(res.oos_predictions, labels, horizons=HORIZONS,
                                   output_dir=str(out_dir / "wf" / "eval"))
    summary = {
        "model": args.model, "cross_sectional": args.cross_sectional, "label_transform": args.label_transform,
        "schema_hash": res.schema_hash, "n_folds": int(len(res.fold_metadata)),
        "oos_rows": int(len(res.oos_predictions)), "manifest_path": res.manifest_path,
        "overall_ic": ev.overall.to_dict("records"), "coverage": ev.coverage,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n=== OVERALL OOS rank-IC (cross-sectional LightGBM) ===")
    print(ev.overall.to_string(index=False))
    print("\nby fold (h=5):")
    print(ev.metrics_by_fold[ev.metrics_by_fold.horizon == 5].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
