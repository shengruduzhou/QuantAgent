#!/usr/bin/env python3
"""Stage 6/7 controlled full-universe production walk-forward training.

Full A-share tradable universe, alpha181 + discovered factors (from the
v8.9+7clean gold dataset), horizons (1,5,20), purged walk-forward over
2020-10 → latest, schema-locked to ONE pinned feature_schema.json, on GPU
(require_gpu). The feature set is coverage-floored (>= --coverage-floor on the
window) so the deep trainer's complete-case dropna keeps the bulk of rows
instead of being zeroed out by sparse broadcast columns (e.g. monthly macro).

Objective: a truthful full-universe OOS rank-IC (read via the rank-IC gate,
not a dressed-up backtest). Records schema hash, feature version, fold window,
seed, factor set, horizon, backend and manifest path for every checkpoint /
prediction.

Usage:
    AI_quant_venv/bin/python3 scripts/stage6_full_walkforward.py [--preflight-only]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.data.dataset_builder.v7_training_dataset import _schema_hash
from quantagent.training.splitters import WalkForwardSplitConfig
from quantagent.training.v7_deep_trainer import (
    V7DeepAlphaTrainerConfig,
    run_walk_forward_deep_training,
)
from quantagent.training.walk_forward_eval import evaluate_walk_forward_oos

GOLD = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean.parquet"
OUT = Path("runtime/stage6_full_walkforward")
START = "2020-10-01"
HORIZONS = (1, 5, 20)
LABEL_PREFIXES = ("forward_return_", "forward_excess_", "forward_rank_", "forward_tradable_", "label_end_")
FORBIDDEN = {"open", "high", "low", "close", "volume", "amount"}
META = {"symbol", "trade_date", "available_at", "source", "source_type",
        "source_reliability", "point_in_time_valid", "is_st_provenance"}
FEATURE_VERSION = "alpha181+disc23@cov099"


def _curated_features(win: pd.DataFrame, coverage_floor: float) -> list[str]:
    num = win.select_dtypes(include=[np.number, bool]).columns
    cand = [c for c in num if not c.startswith(LABEL_PREFIXES) and c not in FORBIDDEN and c not in META]
    numwin = win[cand].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    fr = numwin.notna().mean()
    feats = [c for c in cand if float(fr[c]) >= coverage_floor and win[c].nunique(dropna=True) > 1]
    return sorted(feats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--coverage-floor", type=float, default=0.99)
    ap.add_argument("--min-train-days", type=int, default=500)
    ap.add_argument("--valid-size-days", type=int, default=120)
    ap.add_argument("--n-splits", type=int, default=6)
    ap.add_argument("--purge-days", type=int, default=20)   # == max horizon, no label leak
    ap.add_argument("--embargo-days", type=int, default=5)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1729)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(GOLD)
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    win = df[df["trade_date"] >= START].reset_index(drop=True)
    del df   # free the pre-window copy; the walk-forward driver will copy `win` once
    feats = _curated_features(win, args.coverage_floor)
    label_cols = [f"forward_return_{h}d" for h in HORIZONS]
    schema_hash = _schema_hash(feats, label_cols, HORIZONS)
    schema = {
        "feature_version": FEATURE_VERSION,
        "schema_hash": schema_hash,
        "feature_columns": feats,
        "label_columns": label_cols,
        "horizons": list(HORIZONS),
        "coverage_floor": args.coverage_floor,
        "n_alpha181": len([c for c in feats if c.startswith("alpha")]),
        "n_discovered": len([c for c in feats if c.startswith(("synth_", "rd_", "llm_", "gp_"))]),
    }
    schema_path = OUT / "feature_schema.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    # RAM discipline (not architecture): keep ONLY the pinned features + labels +
    # entity cols, and downcast features to float32. The full 327-col float64
    # frame + the driver's copy + per-fold dropna copies OOM a 62 GB host on the
    # largest expanding folds; the pruned float32 frame is ~1/3 the size.
    keep = ["symbol", "trade_date", *label_cols, *feats]
    train_df = win[keep].copy()
    train_df[feats] = train_df[feats].apply(pd.to_numeric, errors="coerce").astype("float32")
    del win
    n_symbols = int(train_df["symbol"].nunique())

    # Estimated complete-case survival + fold count + artifact size.
    labok = pd.to_numeric(train_df["forward_return_5d"], errors="coerce").notna()
    survive = int((train_df[feats].notna().all(axis=1) & labok).sum())
    n_dates = int(train_df["trade_date"].nunique())
    est_folds = min(args.n_splits, max(0, (n_dates - args.min_train_days) // args.valid_size_days))
    est_oos_rows = est_folds * args.valid_size_days * n_symbols

    base_config = V7DeepAlphaTrainerConfig(
        horizons=HORIZONS, hidden_sizes=(128, 64), dropout=0.10,
        learning_rate=1e-3, weight_decay=1e-4, batch_size=2048,
        max_epochs=args.max_epochs, early_stopping_patience=5,
        rank_loss_weight=0.5, device="cuda", require_gpu=True, use_torch=True,
        log_gpu_memory=True, seed=args.seed,
    )
    split_config = WalkForwardSplitConfig(
        mode="purged", n_splits=args.n_splits, min_train_days=args.min_train_days,
        valid_size_days=args.valid_size_days, embargo_days=args.embargo_days, purge_days=args.purge_days,
    )

    preflight = {
        "dataset": GOLD,
        "window": [START, str(train_df["trade_date"].max().date())],
        "window_rows": int(len(train_df)),
        "window_symbols": n_symbols,
        "window_dates": n_dates,
        "feature_count": len(feats),
        "n_alpha181": schema["n_alpha181"],
        "n_discovered": schema["n_discovered"],
        "coverage_floor": args.coverage_floor,
        "complete_case_rows": survive,
        "complete_case_pct": round(survive / max(1, len(train_df)) * 100, 1),
        "schema_hash": schema_hash,
        "schema_path": str(schema_path),
        "est_folds": int(est_folds),
        "est_oos_rows_upper": int(est_oos_rows),
        "est_predictions_parquet_mb_upper": round(est_oos_rows * 13 * 8 / 1e6, 0),
        "horizons": list(HORIZONS),
        "split_config": vars(split_config) if hasattr(split_config, "__dict__") else None,
        "base_config_seed": args.seed,
        "require_gpu": True,
    }
    print("=== PREFLIGHT ===")
    print(json.dumps(preflight, ensure_ascii=False, indent=2, default=str))
    print("\n=== EXACT RUN CONFIG ===")
    print("base_config:", base_config)
    print("split_config:", split_config)
    (OUT / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if args.preflight_only:
        print("\n[preflight-only] not launching training.")
        return 0

    print("\n=== LAUNCHING FULL WALK-FORWARD TRAINING (GPU) ===", flush=True)
    wf = run_walk_forward_deep_training(
        train_df, feature_schema_path=str(schema_path),
        base_config=base_config, split_config=split_config,
        output_dir=str(OUT / "wf"),
    )
    labels = train_df[["symbol", "trade_date", *label_cols]]
    ev = evaluate_walk_forward_oos(wf.oos_predictions, labels, horizons=HORIZONS,
                                   output_dir=str(OUT / "wf" / "eval"))
    summary = {
        "preflight": preflight,
        "n_folds": int(len(wf.fold_metadata)),
        "oos_rows": int(len(wf.oos_predictions)),
        "gpu_peak_mb_max": wf.run_manifest.get("gpu_peak_mb_max"),
        "manifest_path": wf.manifest_path,
        "overall_ic": ev.overall.to_dict("records"),
        "coverage": ev.coverage,
        "fold_plan": wf.fold_metadata.to_dict("records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n=== DONE ===")
    print("OVERALL OOS rank-IC by horizon:")
    print(ev.overall.to_string(index=False))
    print("\nby fold:")
    print(ev.metrics_by_fold.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
