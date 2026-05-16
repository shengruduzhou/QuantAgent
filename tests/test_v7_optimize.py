"""Tests for the V7 parameter optimisation harness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.training.optimize import OptimizationConfig, run_alpha_param_search


def _toy_training_dataset(num_days: int = 80, num_symbols: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(31)
    dates = pd.bdate_range("2024-01-02", periods=num_days)
    rows: list[dict[str, object]] = []
    for symbol_idx in range(num_symbols):
        symbol = f"S{symbol_idx}"
        for day in dates:
            feature_a = rng.standard_normal()
            feature_b = rng.standard_normal()
            label = 0.05 * feature_a - 0.02 * feature_b + rng.standard_normal() * 0.10
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": day,
                    "available_at": day,
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "forward_return_1d": label,
                    "forward_return_5d": label * 1.05,
                }
            )
    return pd.DataFrame(rows)


def test_grid_search_writes_report_and_picks_best(tmp_path: Path):
    dataset = _toy_training_dataset()
    config = OptimizationConfig(
        parameter_space={
            "model": ["ridge"],
            "min_train_rows": [50, 100],
        },
        sampler="grid",
        output_dir=str(tmp_path),
        train_kwargs={"horizons": (1, 5)},
    )
    result = run_alpha_param_search(dataset, config)
    assert result.report_path.exists()
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["best_candidate"]
    assert len(payload["trials"]) == 2
    # Each trial recorded a metrics block.
    assert all("metrics" in trial for trial in payload["trials"])


def test_random_search_respects_n_trials(tmp_path: Path):
    dataset = _toy_training_dataset()
    config = OptimizationConfig(
        parameter_space={
            "model": ["ridge"],
            "min_train_rows": [50, 100, 200],
        },
        sampler="random",
        n_trials=4,
        seed=42,
        output_dir=str(tmp_path),
        train_kwargs={"horizons": (1, 5)},
    )
    result = run_alpha_param_search(dataset, config)
    assert len(result.trials) == 4
