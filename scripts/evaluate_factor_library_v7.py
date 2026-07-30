#!/usr/bin/env python3
"""Evaluate a complete PIT factor library without mutating the registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantagent.data.dataset_builder import (
    V7TrainingDatasetConfig,
    build_v7_training_dataset_artifact,
)
from quantagent.factors.experiment import (
    FactorScreeningConfig,
    chronological_calibration_slice,
    evaluate_factor_library,
    factor_columns_from_report,
)
from quantagent.data.io import read_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-panel-path", required=True)
    parser.add_argument("--labels-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--factor-library",
        default="all_reviewed",
        choices=("all_reviewed", "basic", "alpha101", "alpha181", "cicc_ashare80"),
    )
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--calibration-days", type=int, default=252)
    parser.add_argument("--holdout-days", type=int, default=60)
    parser.add_argument("--min-finite-ratio", type=float, default=0.30)
    parser.add_argument("--min-abs-rank-ic", type=float, default=0.005)
    parser.add_argument("--min-abs-rank-icir", type=float, default=0.10)
    parser.add_argument("--min-abs-monotonicity", type=float, default=0.15)
    parser.add_argument("--max-pairwise-correlation", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    dataset_path = output / "factor_dataset.parquet"
    result = build_v7_training_dataset_artifact(
        V7TrainingDatasetConfig(
            market_panel_path=args.market_panel_path,
            labels_path=args.labels_path,
            output_path=str(dataset_path),
            horizons=(args.horizon_days,),
            factor_library=args.factor_library,
            min_rows=1,
            min_symbols=1,
            min_dates=1,
        )
    )
    frame = read_frame(result.output_path)
    calibration, split = chronological_calibration_slice(
        frame,
        calibration_days=args.calibration_days,
        holdout_days=args.holdout_days,
    )
    factor_report = result.summary.get("factor_report")
    columns = factor_columns_from_report(
        factor_report if isinstance(factor_report, dict) else None
    )
    evaluation = evaluate_factor_library(
        calibration,
        columns,
        f"forward_return_{args.horizon_days}d",
        output,
        config=FactorScreeningConfig(
            min_finite_ratio=args.min_finite_ratio,
            min_abs_rank_ic=args.min_abs_rank_ic,
            min_abs_rank_icir=args.min_abs_rank_icir,
            min_abs_monotonicity=args.min_abs_monotonicity,
            max_pairwise_correlation=args.max_pairwise_correlation,
        ),
    )
    print(json.dumps({**evaluation.to_dict(), "calibration": split}, ensure_ascii=False))


if __name__ == "__main__":
    main()
