"""Average per-fold OOS predictions across the 3 v10 seed runs and
run the deployed-sleeve backtest on the ensemble.

Reads:
  runtime/models/v7_alpha_full_universe_nosynth_v10_seed{1729,4096,8191}/
    walk_forward/fold_NNN/fold_HHHd_oos_predictions.parquet

Writes:
  runtime/models/v7_alpha_full_universe_nosynth_v10/
    walk_forward/fold_NNN/fold_HHHd_oos_predictions.parquet  (ensemble avg)
  runtime/reports/sleeve_replay_v10/  (sleeve backtest output)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SEEDS = [1729, 4096, 8191]
BASE_OUT = Path("runtime/models/v7_alpha_full_universe_nosynth_v10")
SEED_DIRS = [Path(f"runtime/models/v7_alpha_full_universe_nosynth_v10_seed{seed}") for seed in SEEDS]


def average_fold_horizon(seed_files: list[Path], output: Path) -> None:
    """Average the `prediction` column across N seed parquets, keep all other
    columns from the first file. The (trade_date, symbol) keys must match.
    """
    frames = [pd.read_parquet(p) for p in seed_files if p.exists()]
    if not frames:
        return
    base = frames[0].copy()
    if len(frames) == 1:
        out = base
    else:
        pred_stack = pd.concat(
            [
                f.set_index(["trade_date", "symbol"])["prediction"].rename(f"p_{i}")
                for i, f in enumerate(frames)
            ],
            axis=1,
        )
        avg_pred = pred_stack.mean(axis=1)
        merged = base.set_index(["trade_date", "symbol"]).copy()
        merged["prediction"] = avg_pred.reindex(merged.index).values
        out = merged.reset_index()
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)


def main() -> None:
    available_seeds = [d for d in SEED_DIRS if (d / "walk_forward").exists()]
    print(f"available seed dirs: {[str(d) for d in available_seeds]}")
    if not available_seeds:
        raise SystemExit("no seed walk_forward outputs found; run scripts/run_v10_ensemble.sh first")

    # Use seed-0 as the reference for fold layout
    reference = available_seeds[0] / "walk_forward"
    fold_dirs = sorted([p for p in reference.glob("fold_*") if p.is_dir()])
    print(f"reference fold count: {len(fold_dirs)}")

    ensemble_out = BASE_OUT / "walk_forward"
    ensemble_out.mkdir(parents=True, exist_ok=True)

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        for parquet in sorted(fold_dir.glob("fold_*_oos_predictions.parquet")):
            horizon_file = parquet.name
            seed_files = [d / "walk_forward" / fold_name / horizon_file for d in available_seeds]
            output = ensemble_out / fold_name / horizon_file
            average_fold_horizon(seed_files, output)
        print(f"  averaged {fold_name}")

    print("ensemble OOS predictions written; running deployed-sleeve backtest")
    os.environ["QA_PROBE_DIR"] = str(BASE_OUT)
    os.environ["QA_REPLAY_OUT"] = "runtime/reports/sleeve_replay_v10"
    # Import and run replay_horizon_sleeves
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "replay_horizon_sleeves",
        str(Path(__file__).parent / "replay_horizon_sleeves.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
