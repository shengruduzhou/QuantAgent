"""Grid / random search for V7 alpha and portfolio hyperparameters.

The optimiser is a *research selection* layer, not a live-trading shortcut.
Every candidate is trained/evaluated out of sample by the V7 experiment and a
candidate that fails an anti-overfit requirement is ineligible to become the
champion, regardless of its raw objective value.

Metric names in this module deliberately preserve their financial meaning.  In
particular, a return/drawdown ratio is not called a Sharpe ratio and the sign of
mean Rank IC is not called a hit rate.  If a true Sharpe or hit-rate metric is
produced by the trainer it is passed through unchanged; otherwise it is left
unavailable rather than fabricated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
import json
from pathlib import Path
import random
import subprocess
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from quantagent.config.paths import quant_paths


SearchSpace = dict[str, Sequence[object]]


@dataclass(frozen=True)
class OptimizationConfig:
    parameter_space: SearchSpace
    objective: str = "rank_ic_mean"
    mode: str = "max"  # "max" or "min"
    n_trials: int | None = None
    seed: int = 1729
    sampler: str = "grid"  # "grid" or "random"
    output_dir: str = field(default_factory=lambda: str(quant_paths().reports / "v7" / "optimization"))
    train_kwargs: dict[str, object] = field(default_factory=dict)
    min_folds: int = 1
    stability_threshold: float = float("-inf")


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
    if any(not value for value in values):
        raise ValueError("every optimization parameter must have at least one candidate value")
    if config.sampler == "grid":
        for combo in product(*values):
            yield dict(zip(keys, combo))
    elif config.sampler == "random":
        rng = random.Random(config.seed)
        trials = config.n_trials or 16
        if trials <= 0:
            raise ValueError("n_trials must be positive")
        for _ in range(trials):
            yield {key: rng.choice(value) for key, value in zip(keys, values)}
    else:
        raise ValueError(f"unsupported sampler: {config.sampler}")


def run_alpha_param_search(
    dataset: pd.DataFrame,
    config: OptimizationConfig,
) -> OptimizationResult:
    """Run a governed grid/random search over V7 training hyperparameters.

    A raw objective value never overrides a rejection.  This matters because
    parameter mining tends to make the most extreme-looking trial the most
    attractive one; allowing a trial with too few folds or unstable IC to win
    would invert the purpose of the anti-overfit gate.
    """
    if dataset is None or dataset.empty:
        raise ValueError("optimization requires a non-empty dataset")
    if config.objective not in _SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"unsupported objective {config.objective}; supported: {sorted(_SUPPORTED_OBJECTIVES)}. "
            "The old synthetic objectives sharpe_like/information_ratio_like/hit_rate were removed "
            "because they did not represent those financial statistics."
        )
    if config.mode not in {"max", "min"}:
        raise ValueError("mode must be 'max' or 'min'")
    if config.min_folds < 1:
        raise ValueError("min_folds must be >= 1")

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
        rejection_reasons = _trial_rejection_reasons(metrics, config)
        eligible = not rejection_reasons
        if not eligible:
            metrics["anti_overfit_rejected"] = 1.0

        score = float(metrics.get(config.objective, float("nan")))
        if not np.isfinite(score):
            rejection_reasons.append(f"objective_{config.objective}_not_finite")
            eligible = False

        trial = {
            "trial_id": trial_id,
            "candidate": candidate,
            "metrics": metrics,
            "score": score,
            "eligible": eligible,
            "rejection_reasons": list(dict.fromkeys(rejection_reasons)),
        }
        trials.append(trial)

        # Critical governance boundary: a rejected trial can be recorded and
        # inspected, but it can never be promoted as the best candidate.
        if not eligible:
            continue
        if best_score is None or (config.mode == "max" and score > best_score) or (config.mode == "min" and score < best_score):
            best_score = score
            best_candidate = dict(candidate)
            best_metrics = dict(metrics)

    if best_candidate is None:
        best_candidate = {}
        best_metrics = {}

    eligible_trial_count = sum(bool(trial["eligible"]) for trial in trials)
    report_path = output_dir / "optimization_report.json"
    report_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "git_commit": _git_commit(),
                "dataset_rows": int(len(dataset)),
                "date_range": _date_range(dataset),
                "best_candidate": best_candidate,
                "best_metrics": best_metrics,
                "eligible_trial_count": eligible_trial_count,
                "rejected_trial_count": len(trials) - eligible_trial_count,
                "selection_policy": "only finite, anti-overfit-eligible trials may become champion",
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


_SUPPORTED_OBJECTIVES = {
    "rank_ic_mean",
    "rank_ic_stability",
    "turnover_adjusted_net_return",
    "max_drawdown",
    "return_to_drawdown",
    # These are accepted only when the trainer actually emits the named
    # statistic.  _extract_metrics never manufactures them.
    "sharpe",
    "information_ratio",
    "hit_rate",
    "annualised_return",
    "excess_annualised_return",
}


def _trial_rejection_reasons(metrics: dict[str, float], config: OptimizationConfig) -> list[str]:
    reasons: list[str] = []
    if metrics.get("anti_overfit_rejected", 0.0) > 0.0:
        reasons.append("trainer_anti_overfit_rejected")

    fold_count = metrics.get("fold_count")
    if fold_count is None or not np.isfinite(fold_count):
        reasons.append("fold_count_missing")
    elif fold_count < config.min_folds:
        reasons.append(f"fold_count={fold_count:g}<min_folds={config.min_folds}")

    stability = metrics.get("rank_ic_stability")
    if config.stability_threshold != float("-inf"):
        if stability is None or not np.isfinite(stability):
            reasons.append("rank_ic_stability_missing")
        elif stability < config.stability_threshold:
            reasons.append(
                f"rank_ic_stability={stability:.6g}<threshold={config.stability_threshold:.6g}"
            )
    return reasons


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

    # A useful diagnostic, but name it for what it is.  This is deliberately
    # *not* called Sharpe: no volatility estimate appears in the denominator.
    net_return = flat.get("turnover_adjusted_net_return")
    max_drawdown = flat.get("max_drawdown")
    if net_return is not None and max_drawdown is not None and np.isfinite(net_return) and np.isfinite(max_drawdown):
        drawdown_abs = abs(float(max_drawdown))
        flat["return_to_drawdown"] = (
            float(net_return) / drawdown_abs if drawdown_abs > 1e-12 else float("nan")
        )
    return flat


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip() or None


def _date_range(dataset: pd.DataFrame) -> dict[str, str | None]:
    if "trade_date" not in dataset.columns or dataset.empty:
        return {"start": None, "end": None}
    dates = pd.to_datetime(dataset["trade_date"], errors="coerce").dropna()
    if dates.empty:
        return {"start": None, "end": None}
    return {"start": str(dates.min().date()), "end": str(dates.max().date())}


__all__ = [
    "OptimizationConfig",
    "OptimizationResult",
    "SearchSpace",
    "run_alpha_param_search",
]
