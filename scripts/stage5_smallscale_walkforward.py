#!/usr/bin/env python3
# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): one-shot stage research; conclusion recorded in stage3_to_6_report.md / memory.
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""Stage 5 item 4: real small-scale GPU walk-forward training run.

Subsets the real silver panel to a handful of liquid, board-diverse symbols
over a recent window, builds a board-aware gold dataset with a pinned feature
schema, and runs schema-locked walk-forward deep-alpha training ON GPU
(require_gpu=True, sequential per fold). Emits the run manifest, fold plan and
OOS predictions under ``runtime/walkforward_smallscale/``.

Read-only on the source panel; this is a real training run (the daily data
trust gate is cleared), not a unit test.

Usage:
    AI_quant_venv/bin/python3 scripts/stage5_smallscale_walkforward.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantagent.data.dataset_builder import V7TrainingDatasetConfig, build_v7_training_dataset_artifact
from quantagent.data.v7_label_builder import build_forward_return_labels
from quantagent.training.splitters import WalkForwardSplitConfig
from quantagent.training.v7_deep_trainer import (
    V7DeepAlphaTrainerConfig,
    run_walk_forward_deep_training,
)

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
OUT = Path("runtime/walkforward_smallscale")
START, END = "2022-01-01", "2026-05-18"
HORIZONS = (1, 5)
N_PER_BOARD = 15


def _board(symbol: str) -> str:
    code = str(symbol).split(".")[0]
    if code.startswith(("300", "301")):
        return "chinext"
    if code.startswith(("688", "689")):
        return "star"
    if str(symbol).endswith(".BJ") or code.startswith(("8", "4")):
        return "bse"
    return "main"


def main() -> int:
    cols = ["symbol", "trade_date", "open", "high", "low", "close",
            "volume", "amount", "available_at", "is_st"]
    panel = pd.read_parquet(PANEL, columns=cols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel = panel[(panel["trade_date"] >= START) & (panel["trade_date"] <= END)]

    # Pick the most liquid names per board so we exercise board-aware limits
    # (main 10% vs ChiNext 20%) on real prices.
    liq = panel.groupby("symbol")["amount"].median().dropna().sort_values(ascending=False)
    picks: dict[str, list[str]] = {}
    for sym in liq.index:
        b = _board(sym)
        picks.setdefault(b, [])
        if len(picks[b]) < N_PER_BOARD:
            picks[b].append(sym)
    symbols = [s for board in ("main", "chinext", "star", "bse") for s in picks.get(board, [])]
    panel = panel[panel["symbol"].isin(symbols)].sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    labels = build_forward_return_labels(panel, horizons=HORIZONS).frame
    OUT.mkdir(parents=True, exist_ok=True)
    panel_path, labels_path = OUT / "panel_subset.parquet", OUT / "labels_subset.parquet"
    panel.to_parquet(panel_path, index=False)
    labels.to_parquet(labels_path, index=False)

    built = build_v7_training_dataset_artifact(
        V7TrainingDatasetConfig(
            market_panel_path=str(panel_path),
            labels_path=str(labels_path),
            output_path=str(OUT / "gold_dataset.parquet"),
            horizons=HORIZONS,
            min_rows=200, min_symbols=5, min_dates=50,
            feature_version="v8.9-smallscale",
            factor_library="basic",
        )
    )
    print(f"[dataset] rows={len(built.dataset)} symbols={built.dataset['symbol'].nunique()} "
          f"features={len(built.feature_schema['feature_columns'])} schema_hash={built.feature_schema['schema_hash'][:12]}")

    wf = run_walk_forward_deep_training(
        built.dataset,
        feature_schema_path=str(built.feature_schema_path),
        base_config=V7DeepAlphaTrainerConfig(
            horizons=HORIZONS, hidden_sizes=(64, 32), dropout=0.10,
            max_epochs=20, early_stopping_patience=4, batch_size=1024,
            device="cuda", require_gpu=True, use_torch=True, seed=1729,
        ),
        split_config=WalkForwardSplitConfig(
            mode="purged", n_splits=4, min_train_days=250, valid_size_days=120,
            embargo_days=2, purge_days=5,
        ),
        output_dir=str(OUT / "wf"),
    )

    summary = {
        "symbols": len(symbols),
        "board_breakdown": {b: len(v) for b, v in picks.items()},
        "dataset_rows": int(len(built.dataset)),
        "feature_count": len(wf.feature_columns),
        "schema_hash": wf.schema_hash,
        "feature_version": wf.feature_version,
        "n_folds": int(len(wf.fold_metadata)),
        "oos_rows": int(len(wf.oos_predictions)),
        "gpu_peak_mb_max": wf.run_manifest.get("gpu_peak_mb_max"),
        "manifest_path": wf.manifest_path,
        "fold_plan": wf.fold_metadata.to_dict("records"),
    }
    print(json.dumps(summary, default=str, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
