"""Tests for the V7 parameter optimisation harness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.training.optimize import (
    OptimizationConfig,
    _extract_metrics,
    run_alpha_param_search,
)


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
    # Each trial recorded a metrics block and an explicit eligibility decision.
    assert all("metrics" in trial for trial in payload["trials"])
    assert all("eligible" in trial for trial in payload["trials"])
    assert payload["eligible_trial_count"] + payload["rejected_trial_count"] == 2


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


def test_rejected_high_score_can_never_become_champion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An extreme but statistically invalid trial must not win parameter mining."""
    dataset = _toy_training_dataset(num_days=20, num_symbols=2)

    def fake_training(_dataset, config):
        if config.min_train_rows == 50:
            # The tempting result: huge IC, but zero valid folds.
            return {
                "metrics": {
                    "rank_ic_mean": 0.90,
                    "rank_ic_stability": 1.0,
                    "fold_count": 0,
                }
            }
        return {
            "metrics": {
                "rank_ic_mean": 0.12,
                "rank_ic_stability": 0.8,
                "fold_count": 3,
            }
        }

    monkeypatch.setattr(
        "quantagent.training.v7_experiment.run_v7_training_experiment",
        fake_training,
    )
    result = run_alpha_param_search(
        dataset,
        OptimizationConfig(
            parameter_space={"min_train_rows": [50, 100]},
            objective="rank_ic_mean",
            min_folds=2,
            output_dir=str(tmp_path),
        ),
    )

    assert result.best_candidate == {"min_train_rows": 100}
    assert result.trials[0]["eligible"] is False
    assert result.trials[1]["eligible"] is True
    assert result.trials[0]["score"] > result.trials[1]["score"]
    assert any("fold_count" in reason for reason in result.trials[0]["rejection_reasons"])


def test_metric_names_preserve_financial_semantics() -> None:
    metrics = _extract_metrics(
        {
            "metrics": {
                "turnover_adjusted_net_return": 0.20,
                "max_drawdown": -0.10,
                "rank_ic_mean": 0.03,
            }
        }
    )

    assert metrics["return_to_drawdown"] == pytest.approx(2.0)
    # These used to be fabricated aliases.  A real Sharpe/hit-rate must come
    # from actual return observations, not be synthesized from unrelated fields.
    assert "sharpe_like" not in metrics
    assert "hit_rate" not in metrics


def test_misleading_legacy_objective_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="synthetic objectives"):
        run_alpha_param_search(
            _toy_training_dataset(num_days=10, num_symbols=2),
            OptimizationConfig(
                parameter_space={"model": ["ridge"]},
                objective="sharpe_like",
                output_dir=str(tmp_path),
            ),
        )
