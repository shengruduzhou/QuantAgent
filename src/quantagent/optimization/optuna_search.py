"""Optuna hyperparameter search for the V7 alpha loop.

This layer optimises research artefacts only. It never emits orders and
does not enable live trading; portfolio knobs are persisted as candidate
metadata for downstream target-weight construction and paper backtests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.training.v7_experiment import V7TrainingConfig, run_v7_training_experiment


@dataclass(frozen=True)
class OptunaSearchConfig:
    study_name: str = "v7_alpha"
    storage: str | None = None
    n_trials: int = 100
    model: str = "ft_transformer"
    horizons: tuple[int, ...] = (1, 5, 20, 60, 120, 126)
    split_mode: str = "rolling"
    min_train_rows: int = 1000
    min_train_days: int = 120
    valid_size_days: int = 20
    rolling_train_days: int = 756
    embargo_days: int = 5
    n_splits: int = 4
    ft_max_epochs: int = 60
    ft_batch_size: int = 8192
    ft_device: str = "cuda"
    require_gpu: bool = True
    output_dir: str = field(default_factory=lambda: str(quant_paths().reports / "v7" / "optuna"))
    seed: int = 1729
    max_drawdown_penalty_threshold: float = 0.20


@dataclass(frozen=True)
class OptunaSearchResult:
    study_name: str
    storage: str
    best_params: dict[str, Any]
    best_value: float | None
    best_hp_path: str
    trials_path: str
    study_dashboard_hint: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_search_space(trial: Any) -> dict[str, Any]:
    """Build the Phase 4.1 search space from an Optuna trial."""

    return {
        "ft_d_token": trial.suggest_categorical("ft_d_token", [64, 128, 192, 256]),
        "ft_n_blocks": trial.suggest_int("ft_n_blocks", 3, 6),
        "ft_attention_dropout": trial.suggest_float("ft_attention_dropout", 0.0, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True),
        "ft_weight_decay": trial.suggest_float("ft_weight_decay", 1e-6, 1e-2, log=True),
        "top_k": trial.suggest_int("top_k", 20, 100),
        "alpha_threshold": trial.suggest_float("alpha_threshold", -0.005, 0.02),
        "confidence_floor": trial.suggest_float("confidence_floor", 0.40, 0.70),
        "selection_top_k_max": trial.suggest_int("selection_top_k_max", 30, 150),
        "max_sector_weight": trial.suggest_float("max_sector_weight", 0.15, 0.35),
        "max_turnover": trial.suggest_float("max_turnover", 0.10, 0.40),
        "cost_bps": trial.suggest_float("cost_bps", 5.0, 20.0),
        "factor_weight_policy": trial.suggest_float("factor_weight_policy", 0.0, 1.0),
        "factor_weight_fundamental": trial.suggest_float("factor_weight_fundamental", 0.0, 1.0),
        "factor_weight_technical": trial.suggest_float("factor_weight_technical", 0.0, 1.0),
        "factor_weight_flow": trial.suggest_float("factor_weight_flow", 0.0, 1.0),
        "factor_weight_news": trial.suggest_float("factor_weight_news", 0.0, 1.0),
    }


def run_optuna_hp_search(dataset: pd.DataFrame, config: OptunaSearchConfig | None = None) -> OptunaSearchResult:
    cfg = config or OptunaSearchConfig()
    if dataset is None or dataset.empty:
        raise ValueError("Optuna HP search requires a non-empty PIT training dataset")
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("hp-search requires optuna; install the optimization environment first") from exc

    output_dir = Path(cfg.output_dir) / cfg.study_name
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = cfg.storage or f"sqlite:///{output_dir / 'study.db'}"
    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name=cfg.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    def objective(trial: Any) -> float:
        params = build_search_space(trial)
        training_params = _training_params(params)
        trial_dir = output_dir / f"trial_{trial.number:04d}"
        result = run_v7_training_experiment(
            dataset,
            V7TrainingConfig(
                model=cfg.model,
                horizons=cfg.horizons,
                split_mode=cfg.split_mode,
                min_train_rows=cfg.min_train_rows,
                min_train_days=cfg.min_train_days,
                valid_size_days=cfg.valid_size_days,
                rolling_train_days=cfg.rolling_train_days,
                embargo_days=cfg.embargo_days,
                n_splits=cfg.n_splits,
                ft_max_epochs=cfg.ft_max_epochs,
                ft_batch_size=cfg.ft_batch_size,
                ft_device=cfg.ft_device,
                require_gpu=cfg.require_gpu,
                output_dir=str(trial_dir),
                cost_bps=float(params["cost_bps"]),
                **training_params,
            ),
        )
        metrics = result.metrics
        score = _score(metrics, cfg.max_drawdown_penalty_threshold)
        trial.set_user_attr("metrics", _jsonable(metrics))
        trial.set_user_attr("portfolio_params", _portfolio_params(params))
        trial.set_user_attr("factor_group_weights", _normalised_factor_weights(params))
        trial.set_user_attr("artifact_dir", result.output_dir)
        return score

    study.optimize(objective, n_trials=cfg.n_trials)
    best = study.best_trial if study.best_trial is not None else None
    best_payload = {
        "study_name": cfg.study_name,
        "storage": storage,
        "best_value": float(best.value) if best is not None and best.value is not None else None,
        "best_params": dict(best.params) if best is not None else {},
        "portfolio_params": best.user_attrs.get("portfolio_params", {}) if best is not None else {},
        "factor_group_weights": best.user_attrs.get("factor_group_weights", {}) if best is not None else {},
        "metrics": best.user_attrs.get("metrics", {}) if best is not None else {},
        "artifact_dir": best.user_attrs.get("artifact_dir") if best is not None else None,
        "git_commit": _git_commit(),
    }
    best_path = output_dir / "best_hp.json"
    best_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    trials_path = output_dir / "trials.csv"
    study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs")).to_csv(trials_path, index=False)
    return OptunaSearchResult(
        study_name=cfg.study_name,
        storage=storage,
        best_params=best_payload["best_params"],
        best_value=best_payload["best_value"],
        best_hp_path=str(best_path),
        trials_path=str(trials_path),
        study_dashboard_hint=f"optuna-dashboard {storage}",
    )


def _training_params(params: dict[str, Any]) -> dict[str, Any]:
    keys = {"ft_d_token", "ft_n_blocks", "ft_attention_dropout", "learning_rate", "ft_weight_decay"}
    return {key: params[key] for key in keys}


def _portfolio_params(params: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "top_k",
        "alpha_threshold",
        "confidence_floor",
        "selection_top_k_max",
        "max_sector_weight",
        "max_turnover",
        "cost_bps",
    }
    return {key: params[key] for key in keys if key in params}


def _normalised_factor_weights(params: dict[str, Any]) -> dict[str, float]:
    raw = {
        key.removeprefix("factor_weight_"): float(value)
        for key, value in params.items()
        if key.startswith("factor_weight_")
    }
    total = sum(raw.values())
    if total <= 0:
        return {key: 1.0 / max(1, len(raw)) for key in raw}
    return {key: value / total for key, value in raw.items()}


def _score(metrics: dict[str, object], dd_threshold: float) -> float:
    sharpe = float(
        metrics.get(
            "sharpe_like",
            metrics.get("information_ratio_like", metrics.get("rank_ic_stability", metrics.get("rank_ic_mean", 0.0))),
        )
        or 0.0
    )
    drawdown = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    return sharpe - 2.0 * max(0.0, drawdown - float(dd_threshold))


def _jsonable(metrics: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(metrics, ensure_ascii=False, default=str))


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip() or None


__all__ = ["OptunaSearchConfig", "OptunaSearchResult", "build_search_space", "run_optuna_hp_search"]
