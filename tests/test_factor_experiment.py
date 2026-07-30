from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quantagent.factors.experiment import (
    FactorScreeningConfig,
    chronological_calibration_slice,
    evaluate_factor_library,
    factor_columns_from_report,
)


def _frame(days: int = 40, symbols: int = 20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(7)
    for day in range(days):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
        for index in range(symbols):
            signal = index / symbols + day * 0.001
            rows.append(
                {
                    "trade_date": date,
                    "symbol": f"{index:06d}.SZ",
                    "amount": 1_000_000 + index * 10_000,
                    "good": signal,
                    "duplicate": signal * 2.0,
                    "noise": rng.normal(),
                    "forward_return_5d": signal * 0.03 + rng.normal(scale=0.001),
                }
            )
    return pd.DataFrame(rows)


def test_factor_evaluation_persists_metrics_and_prunes_correlation(tmp_path) -> None:
    result = evaluate_factor_library(
        _frame(),
        ["good", "duplicate", "noise"],
        "forward_return_5d",
        tmp_path,
        config=FactorScreeningConfig(
            min_abs_rank_ic=0.01,
            min_abs_rank_icir=0.01,
            min_abs_monotonicity=0.01,
            max_pairwise_correlation=0.80,
        ),
    )
    assert len(result.summary) == 3
    assert len({"good", "duplicate"} & set(result.selected_factors)) == 1
    assert (tmp_path / "factor_summary.csv").exists()
    assert (tmp_path / "factor_correlation.csv").exists()
    payload = json.loads((tmp_path / "factor_selection.json").read_text())
    assert payload["correlationEvidence"]["factorCount"] == 3


def test_calibration_window_excludes_later_holdout() -> None:
    calibration, evidence = chronological_calibration_slice(
        _frame(days=40),
        calibration_days=25,
        holdout_days=10,
    )
    assert calibration["trade_date"].nunique() == 25
    assert evidence["holdoutDates"] == 15
    assert calibration["trade_date"].max() < _frame(days=40)["trade_date"].max()


def test_factor_columns_flatten_mixed_library_report() -> None:
    report = {
        "library": "all_reviewed",
        "members": [
            {"library": "alpha101", "added_columns": ["a", "b"]},
            {"library": "alpha181", "added_columns": ["b", "c"]},
        ],
    }
    assert factor_columns_from_report(report) == ["a", "b", "c"]
