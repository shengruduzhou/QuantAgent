"""Driver: train v8 deep (FT-Transformer GPU) across 3 horizons + ensemble.

Spawns three sequential ``train-v8-deep`` invocations (one per horizon
class), then blends their OOS predictions into a single composite via
``HorizonEnsembleWeights``. Designed to run inside the GPU tmux session.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime


HORIZONS = ("short_5d", "mid_5d_30d", "long_30d_120d")


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_horizon(
    *,
    python_bin: str,
    horizon: str,
    output_dir: pathlib.Path,
    common_args: list[str],
) -> int:
    cmd = [
        python_bin, "-m", "quantagent.cli", "train-v8-deep",
        "--horizon-class", horizon,
        "--output-dir", str(output_dir),
        *common_args,
    ]
    print(f"[{_ts()}] HORIZON {horizon} → {output_dir}", flush=True)
    print(f"[{_ts()}] cmd: {' '.join(cmd[:6])} ... ({len(cmd)} args)", flush=True)
    proc = subprocess.run(cmd)
    print(f"[{_ts()}] horizon={horizon} exit={proc.returncode}", flush=True)
    return proc.returncode


def blend(output_root: pathlib.Path) -> None:
    import pandas as pd
    from quantagent.training.horizon_models import (
        HorizonClass, HorizonEnsembleWeights, ensemble_horizon_predictions,
    )

    predictions: dict[HorizonClass, pd.DataFrame] = {}
    used: list[str] = []
    for hz in HORIZONS:
        p = output_root / hz / "predictions.parquet"
        if not p.exists():
            print(f"[skip] {hz}: missing {p}", flush=True)
            continue
        long_df = pd.read_parquet(p)
        # Ensure required columns
        if "alpha_score" not in long_df.columns:
            score_cols = [c for c in long_df.columns if c not in ("trade_date", "symbol")]
            if score_cols:
                long_df = long_df.rename(columns={score_cols[0]: "alpha_score"})
        predictions[HorizonClass(hz)] = long_df[["trade_date", "symbol", "alpha_score"]]
        used.append(hz)
        print(f"[load] {hz}: {len(long_df)} rows", flush=True)
    if not predictions:
        print("no horizon outputs to ensemble", flush=True)
        return
    weights = HorizonEnsembleWeights()
    blended = ensemble_horizon_predictions(predictions, weights=weights)
    out = output_root / "ensemble_composite.parquet"
    blended.to_parquet(out)
    summary = {
        "n_rows": int(len(blended)),
        "weights": weights.as_dict(),
        "horizons_used": used,
        "generated_at": _ts(),
    }
    (output_root / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    print(f"[wrote] {out}", flush=True)
    print(f"[wrote] {output_root / 'ensemble_summary.json'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="v8 deep GPU sweep across 3 horizons")
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--symbols-file", required=True, type=pathlib.Path)
    parser.add_argument("--dataset-path", required=True, type=pathlib.Path)
    parser.add_argument("--silver-panel-path", required=True, type=pathlib.Path)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    parser.add_argument("--embargo-days", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--d-token", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dates-per-step", type=int, default=8,
                        help="trading days per optimisation step; lower → smaller activation footprint")
    parser.add_argument("--train-micro-batch", type=int, default=None,
                        help="cap rows per fwd/bwd within a date chunk")
    parser.add_argument("--cross-sectional-norm", default="rank",
                        help="per-date feature normalisation: rank | zscore | none")
    parser.add_argument("--label-norm", default="1", help="1=per-date label winsor+zscore, 0=raw")
    parser.add_argument("--attention-dropout", type=float, default=0.10)
    parser.add_argument("--ffn-dropout", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    common = [
        "--dataset-path", str(args.dataset_path),
        "--silver-panel-path", str(args.silver_panel_path),
        "--symbols-file", str(args.symbols_file),
        "--train-start", args.train_start,
        "--train-end", args.train_end,
        "--test-end", args.test_end,
        "--embargo-days", str(args.embargo_days),
        "--top-k", str(args.top_k),
        "--max-epochs", str(args.max_epochs),
        "--batch-size", str(args.batch_size),
        "--d-token", str(args.d_token),
        "--n-blocks", str(args.n_blocks),
        "--n-heads", str(args.n_heads),
        "--dates-per-step", str(args.dates_per_step),
        "--cross-sectional-norm", str(args.cross_sectional_norm),
        "--attention-dropout", str(args.attention_dropout),
        "--ffn-dropout", str(args.ffn_dropout),
        "--weight-decay", str(args.weight_decay),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--learning-rate", str(args.learning_rate),
        "--require-gpu",
    ]
    common += ["--label-norm" if str(args.label_norm) == "1" else "--no-label-norm"]
    if args.train_micro_batch is not None:
        common += ["--train-micro-batch", str(args.train_micro_batch)]

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"[{_ts()}] sweep start. output_root={args.output_root}", flush=True)
    t0 = time.time()
    failed: list[str] = []
    for hz in HORIZONS:
        hz_dir = args.output_root / hz
        hz_dir.mkdir(parents=True, exist_ok=True)
        rc = run_horizon(
            python_bin=args.python_bin,
            horizon=hz, output_dir=hz_dir,
            common_args=common,
        )
        if rc != 0:
            failed.append(hz)
            print(f"[{_ts()}] horizon {hz} failed; continuing", flush=True)
    print(f"[{_ts()}] horizon sweep done in {time.time() - t0:.1f}s; failed={failed}", flush=True)
    print(f"[{_ts()}] ensemble blend …", flush=True)
    blend(args.output_root)
    print(f"[{_ts()}] sweep complete. total_elapsed={time.time() - t0:.1f}s", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
