"""Driver: train v8 pipeline across short/mid/long horizons, then ensemble.

Spawns three sequential ``train-v8-pipeline`` invocations through
subprocess, then blends their target_weights into a single composite
prediction frame. All artifacts land under ``--output-root``.

This replaces the shell heredoc version which lost the multi-kilobyte
symbol list to shell quoting.
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
    symbols_file: pathlib.Path,
    silver_panel: pathlib.Path,
    start_date: str,
    end_date: str,
    top_k: int,
    ga_population: int,
    ga_generations: int,
    horizon: str,
    output_dir: pathlib.Path,
) -> int:
    """Invoke the v8 CLI for ONE horizon. Returns the CLI exit code."""
    symbols = symbols_file.read_text(encoding="utf-8").strip()
    if not symbols:
        raise RuntimeError(f"symbols file empty: {symbols_file}")
    cmd = [
        python_bin, "-m", "quantagent.cli", "train-v8-pipeline",
        "--symbols", symbols,
        "--start-date", start_date, "--end-date", end_date,
        "--use-silver-panel", str(silver_panel),
        "--horizon-class", horizon,
        "--top-k", str(top_k),
        "--ga-population", str(ga_population),
        "--ga-generations", str(ga_generations),
        "--output-dir", str(output_dir),
    ]
    print(f"[{_ts()}] running horizon={horizon} → {output_dir}", flush=True)
    proc = subprocess.run(cmd, capture_output=False)
    print(f"[{_ts()}] horizon={horizon} exit={proc.returncode}", flush=True)
    return proc.returncode


def blend_ensemble(output_root: pathlib.Path) -> None:
    import pandas as pd
    from quantagent.training.horizon_models import (
        HorizonClass, HorizonEnsembleWeights, ensemble_horizon_predictions,
    )

    predictions: dict[HorizonClass, pd.DataFrame] = {}
    horizons_used: list[str] = []
    for hz in HORIZONS:
        p = output_root / hz / "target_weights.parquet"
        if not p.exists():
            print(f"[skip] {hz}: missing {p}", flush=True)
            continue
        wide = pd.read_parquet(p)
        long_df = wide.stack().rename("alpha_score").reset_index()
        long_df.columns = ["trade_date", "symbol", "alpha_score"]
        predictions[HorizonClass(hz)] = long_df
        horizons_used.append(hz)
        print(f"[load] {hz}: {len(long_df)} rows", flush=True)
    if not predictions:
        print("no horizon outputs to ensemble", flush=True)
        return
    weights = HorizonEnsembleWeights()
    blended = ensemble_horizon_predictions(predictions, weights=weights)
    out_parquet = output_root / "ensemble_composite.parquet"
    blended.to_parquet(out_parquet)
    summary = {
        "n_rows": int(len(blended)),
        "weights": weights.as_dict(),
        "horizons_used": horizons_used,
        "generated_at": _ts(),
    }
    (output_root / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    print(f"[wrote] {out_parquet}", flush=True)
    print(f"[wrote] {output_root / 'ensemble_summary.json'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="v8 pipeline sweep across 3 horizons")
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--symbols-file", required=True, type=pathlib.Path)
    parser.add_argument("--silver-panel", required=True, type=pathlib.Path)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--ga-population", type=int, default=48)
    parser.add_argument("--ga-generations", type=int, default=20)
    args = parser.parse_args()

    print(f"[{_ts()}] sweep start", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    failed: list[str] = []
    for hz in HORIZONS:
        hz_dir = args.output_root / hz
        hz_dir.mkdir(parents=True, exist_ok=True)
        rc = run_horizon(
            python_bin=args.python_bin,
            symbols_file=args.symbols_file,
            silver_panel=args.silver_panel,
            start_date=args.start_date,
            end_date=args.end_date,
            top_k=args.top_k,
            ga_population=args.ga_population,
            ga_generations=args.ga_generations,
            horizon=hz,
            output_dir=hz_dir,
        )
        if rc != 0:
            failed.append(hz)
            print(f"[{_ts()}] horizon {hz} failed; continuing to next", flush=True)
    print(f"[{_ts()}] horizon sweep done in {time.time() - t0:.1f}s; failed={failed}", flush=True)
    print(f"[{_ts()}] starting ensemble blend", flush=True)
    blend_ensemble(args.output_root)
    print(f"[{_ts()}] sweep complete; total elapsed={time.time() - t0:.1f}s", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
