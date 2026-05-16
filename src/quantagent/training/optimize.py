"""Grid / random search for V7 alpha and portfolio hyperparameters.

The search loop is intentionally simple and deterministic. For each
candidate hyperparameter combination the executor:

1. Trains the alpha model on the training window via
   ``run_v7_training_experiment``.
2. Evaluates walk-forward metrics: rank IC mean, rank IC stability,
   turnover-adjusted return after cost, max drawdown, hit rate.
3. Records the candidate, metrics and any constraint-violation count
   into a report written under ``reports/v7/optimization/``.

The optimiser does **not** touch live trading and obeys the same
acceptance gates as the standalone trainer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SearchSpace = dict[str, Sequence[object]]


@dataclass(frozen=True)
class OptimizationConfig:
    parameter_space: SearchSpace
    objective: str = "rank_ic_mean"
    mode: str = "max"  # "max" or "min"
    n_trials: int | None = None
    seed: int = 1729
    sampler: str = "grid"  # "grid" or "random"
    output_dir: str = "reports/v7/optimization"
    train_kwargs: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationResult:
    best_candidate: dict[str, object]
    best_metrics: dict[str, float]
    trials: list[dict[str, object]]
    report_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "best_candidate": dict(self.best_candidate),
            "best_metrics": dict(self.best_metrics),
            "trials": [dict(trial) for trial in self.trials],
            "report_path": str(self.report_path),
        }


def _iter_candidates(config: OptimizationConfig) -> Iterable[dict[str, object]]:
    keys = list(config.parameter_space.keys())
    if not keys:
        yield {}
        return
    values = [list(config.parameter_space[k]) for k in keys]
    if config.sampler == "grid":
        for combo in product(*values):
            yield dict(zip(keys, combo))
    elif config.sampler == "random":
        rng = random.Random(config.seed)
        trials = config.n_trials or 16
        for _ in range(trials):
            yield {key: rng.choice(value) for key, value in zip(keys, values)}
    else:
        raise ValueError(f"unsupported sampler: {config.sampler}")


def run_alpha_param_search(
    dataset: pd.DataFrame,
    config: OptimizationConfig,
) -> OptimizationResult:
    """Run a grid / random search over alpha training hyperparameters.

    The search delegates training to
    :func:`quantagent.training.v7_experiment.run_v7_training_experiment`
    and reads metrics from its returned payload, so any model supported
    by the V7 trainer (ridge, elastic_net, lightgbm, xgboost) can be
    optimised through this entry point.
    """
    if dataset is None or dataset.empty:
        raise ValueError("optimization requires a non-empty dataset")
    from quantagent.training.v7_experiment import V7TrainingConfig, run_v7_training_experiment

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials: list[dict[str, object]] = []
    best_score: float | None = None
    best_candidate: dict[str, object] | None = None
    best_metrics: dict[str, float] | None = None

    for trial_id, candidate in enumerate(_iter_candidates(config)):
        kwargs = dict(config.train_kwargs)
        kwargs.update(candidate)
        kwargs.setdefault("output_dir", str(output_dir / f"trial_{trial_id:03d}"))
        result = run_v7_training_experiment(dataset, V7TrainingConfig(**kwargs))
        metrics = _extract_metrics(result)
        score = float(metrics.get(config.objective, float("nan")))
        trial = {
            "trial_id": trial_id,
            "candidate": candidate,
            "metrics": metrics,
            "score": score,
        }
        trials.append(trial)
        if np.isnan(score):
            continue
        if best_score is None or (config.mode == "max" and score > best_score) or (config.mode == "min" and score < best_score):
            best_score = score
            best_candidate = dict(candidate)
            best_metrics = dict(metrics)

    if best_candidate is None:
        best_candidate = {}
        best_metrics = {}
    report_path = output_dir / "optimization_report.json"
    report_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "best_candidate": best_candidate,
                "best_metrics": best_metrics,
                "trials": trials,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    return OptimizationResult(
        best_candidate=best_candidate,
        best_metrics=best_metrics or {},
        trials=trials,
        report_path=report_path,
    )


def _extract_metrics(training_result: object) -> dict[str, float]:
    if isinstance(training_result, dict):
        metrics_block = training_result.get("metrics")
    else:
        metrics_block = getattr(training_result, "metrics", None)
    if not isinstance(metrics_block, dict):
        return {}
    flat: dict[str, float] = {}
    for key, value in metrics_block.items():
        if isinstance(value, bool):
            flat[str(key)] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            flat[str(key)] = float(value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, bool):
                    flat[f"{key}.{sub_key}"] = 1.0 if sub_value else 0.0
                elif isinstance(sub_value, (int, float)):
                    flat[f"{key}.{sub_key}"] = float(sub_value)
    return flat


__all__ = [
    "OptimizationConfig",
    "OptimizationResult",
    "SearchSpace",
    "run_alpha_param_search",
]
