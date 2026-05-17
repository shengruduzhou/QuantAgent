"""Genetic factor-mask and ensemble-weight evolution for V7.

The GA searches feature subsets, horizon blend weights and model ensemble
weights against walk-forward validation metrics. It remains a research
loop: the output is a genome/config JSON, not orders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.training.v7_experiment import V7TrainingConfig, run_v7_training_experiment


@dataclass(frozen=True)
class FactorEvolutionConfig:
    generations: int = 30
    population: int = 60
    seed_from_optuna: str | None = "v7_alpha"
    output_dir: str = field(default_factory=lambda: str(quant_paths().reports / "v7" / "optuna"))
    min_train_rows: int = 1000
    split_mode: str = "rolling"
    valid_size_days: int = 20
    min_train_days: int = 120
    rolling_train_days: int = 756
    embargo_days: int = 5
    n_splits: int = 4
    model: str = "ridge"
    seed: int = 1729


@dataclass(frozen=True)
class FactorEvolutionResult:
    best_genome: dict[str, Any]
    best_genome_path: str
    history_path: str
    rows: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_factor_evolution(dataset: pd.DataFrame, config: FactorEvolutionConfig | None = None) -> FactorEvolutionResult:
    cfg = config or FactorEvolutionConfig()
    if dataset is None or dataset.empty:
        raise ValueError("factor evolution requires a non-empty PIT training dataset")
    try:
        from deap import base, creator, tools
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("evolve-factors requires deap; install the optimization environment first") from exc

    output_dir = Path(cfg.output_dir)
    if cfg.seed_from_optuna:
        output_dir = output_dir / cfg.seed_from_optuna
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    feature_columns = _feature_columns(dataset)
    if not feature_columns:
        raise ValueError("factor evolution found no numeric feature columns")
    hp_seed = _load_hp_seed(output_dir)
    horizons = _available_horizons(dataset)
    genome_size = len(feature_columns) + len(horizons) + 4

    if not hasattr(creator, "V7Fitness"):
        creator.create("V7Fitness", base.Fitness, weights=(1.0, 1.0))
    if not hasattr(creator, "V7Individual"):
        creator.create("V7Individual", list, fitness=creator.V7Fitness)

    toolbox = base.Toolbox()
    toolbox.register("individual", _make_individual, creator.V7Individual, len(feature_columns), len(horizons), rng)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", _mate, feature_count=len(feature_columns), horizon_count=len(horizons))
    toolbox.register("mutate", _mutate, feature_count=len(feature_columns), rng=rng)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", _evaluate, dataset=dataset, feature_columns=feature_columns, horizons=horizons, cfg=cfg, hp_seed=hp_seed)

    population = toolbox.population(n=cfg.population)
    history: list[dict[str, object]] = []
    for generation in range(cfg.generations + 1):
        invalid = [ind for ind in population if not ind.fitness.valid]
        for individual in invalid:
            individual.fitness.values = toolbox.evaluate(individual)
            history.append(_history_row(generation, individual, feature_columns, horizons))
        if generation == cfg.generations:
            break
        offspring = tools.selBest(population, len(population))
        offspring = [creator.V7Individual(ind) for ind in offspring]
        for left, right in zip(offspring[::2], offspring[1::2]):
            if rng.random() < 0.7:
                toolbox.mate(left, right)
                del left.fitness.values, right.fitness.values
        for mutant in offspring:
            if rng.random() < 0.3:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid_offspring = [ind for ind in offspring if not ind.fitness.valid]
        for individual in invalid_offspring:
            individual.fitness.values = toolbox.evaluate(individual)
            history.append(_history_row(generation, individual, feature_columns, horizons))
        population = toolbox.select(population + offspring, cfg.population)

    best = tools.selBest(population, 1)[0]
    best_payload = _decode(best, feature_columns, horizons)
    best_payload["fitness"] = {
        "sharpe_proxy": float(best.fitness.values[0]),
        "calmar_proxy": float(best.fitness.values[1]),
    }
    best_path = output_dir / "best_genome.json"
    best_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    history_frame = pd.DataFrame(history)
    history_path = output_dir / "ga_history.parquet"
    try:
        history_frame.to_parquet(history_path, index=False)
    except Exception:
        history_path = history_path.with_suffix(".csv")
        history_frame.to_csv(history_path, index=False)
    return FactorEvolutionResult(
        best_genome=best_payload,
        best_genome_path=str(best_path),
        history_path=str(history_path),
        rows=int(len(history_frame)),
    )


def _make_individual(individual_type: Any, feature_count: int, horizon_count: int, rng: random.Random) -> Any:
    mask = [1 if rng.random() < 0.75 else 0 for _ in range(feature_count)]
    if not any(mask):
        mask[rng.randrange(feature_count)] = 1
    horizon = _simplex(horizon_count, rng)
    ensemble = _simplex(4, rng)
    return individual_type(mask + horizon + ensemble)


def _mate(left: list[float], right: list[float], *, feature_count: int, horizon_count: int) -> tuple[list[float], list[float]]:
    if feature_count >= 2:
        p1, p2 = sorted(random.sample(range(feature_count), 2))
        left[p1:p2], right[p1:p2] = right[p1:p2], left[p1:p2]
    start = feature_count
    end = feature_count + horizon_count + 4
    for i in range(start, end):
        a, b = float(left[i]), float(right[i])
        gamma = random.uniform(-0.3, 1.3)
        left[i] = (1 - gamma) * a + gamma * b
        right[i] = gamma * a + (1 - gamma) * b
    _renormalise_slice(left, feature_count, feature_count + horizon_count)
    _renormalise_slice(right, feature_count, feature_count + horizon_count)
    _renormalise_slice(left, feature_count + horizon_count, end)
    _renormalise_slice(right, feature_count + horizon_count, end)
    return left, right


def _mutate(individual: list[float], *, feature_count: int, rng: random.Random) -> tuple[list[float]]:
    for i in range(feature_count):
        if rng.random() < 0.05:
            individual[i] = 0 if int(individual[i]) else 1
    if not any(int(v) for v in individual[:feature_count]):
        individual[rng.randrange(feature_count)] = 1
    for i in range(feature_count, len(individual)):
        if rng.random() < 0.20:
            individual[i] = max(0.0, float(individual[i]) + rng.gauss(0.0, 0.05))
    return (individual,)


def _evaluate(
    individual: list[float],
    *,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    horizons: tuple[int, ...],
    cfg: FactorEvolutionConfig,
    hp_seed: dict[str, Any],
) -> tuple[float, float]:
    decoded = _decode(individual, feature_columns, horizons)
    selected = tuple(decoded["selected_features"])
    if not selected:
        return -1e6, -1e6
    params = dict(hp_seed.get("best_params", {}))
    safe_params = {k: v for k, v in params.items() if k in {"ft_d_token", "ft_n_blocks", "ft_attention_dropout", "learning_rate", "ft_weight_decay"}}
    try:
        result = run_v7_training_experiment(
            dataset,
            V7TrainingConfig(
                model=cfg.model,
                horizons=horizons,
                feature_columns=selected,
                min_train_rows=cfg.min_train_rows,
                split_mode=cfg.split_mode,
                valid_size_days=cfg.valid_size_days,
                min_train_days=cfg.min_train_days,
                rolling_train_days=cfg.rolling_train_days,
                embargo_days=cfg.embargo_days,
                n_splits=cfg.n_splits,
                output_dir=str(Path(cfg.output_dir) / "ga_eval"),
                **safe_params,
            ),
        )
    except Exception:
        return -1e6, -1e6
    metrics = result.metrics
    ret = float(metrics.get("turnover_adjusted_net_return", metrics.get("rank_ic_mean", 0.0)) or 0.0)
    dd = abs(float(metrics.get("max_drawdown", 0.0) or 0.0)) + 1e-9
    sharpe = float(metrics.get("sharpe_like", metrics.get("rank_ic_stability", 0.0)) or 0.0)
    calmar = ret / dd
    return sharpe, calmar


def _decode(individual: list[float], feature_columns: list[str], horizons: tuple[int, ...]) -> dict[str, Any]:
    feature_count = len(feature_columns)
    horizon_count = len(horizons)
    selected = [name for flag, name in zip(individual[:feature_count], feature_columns) if int(flag)]
    horizon_weights = _normalise(individual[feature_count : feature_count + horizon_count])
    ensemble_weights = _normalise(individual[feature_count + horizon_count : feature_count + horizon_count + 4])
    return {
        "selected_features": selected,
        "factor_binary_mask": [int(v) for v in individual[:feature_count]],
        "horizon_blend": {str(h): float(w) for h, w in zip(horizons, horizon_weights)},
        "ensemble": {name: float(w) for name, w in zip(("ridge", "lightgbm", "xgboost", "ft_transformer"), ensemble_weights)},
    }


def _history_row(generation: int, individual: Any, feature_columns: list[str], horizons: tuple[int, ...]) -> dict[str, object]:
    decoded = _decode(individual, feature_columns, horizons)
    return {
        "generation": generation,
        "fitness_sharpe_proxy": float(individual.fitness.values[0]),
        "fitness_calmar_proxy": float(individual.fitness.values[1]),
        "selected_feature_count": len(decoded["selected_features"]),
        "genome": json.dumps(decoded, ensure_ascii=False, sort_keys=True),
    }


def _feature_columns(dataset: pd.DataFrame) -> list[str]:
    labels = {c for c in dataset.columns if c.startswith("forward_return_") or c.startswith("label_end_")}
    forbidden = labels | {"symbol", "trade_date", "available_at", "ann_date", "report_period"}
    return [c for c in dataset.select_dtypes(include=[np.number, bool]).columns if c not in forbidden]


def _available_horizons(dataset: pd.DataFrame) -> tuple[int, ...]:
    horizons = []
    for column in dataset.columns:
        if column.startswith("forward_return_") and column.endswith("d"):
            horizons.append(int(column.removeprefix("forward_return_").removesuffix("d")))
    return tuple(sorted(set(horizons))) or (1, 5, 20, 60, 120, 126)


def _load_hp_seed(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "best_hp.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _simplex(size: int, rng: random.Random) -> list[float]:
    raw = [rng.random() for _ in range(max(1, size))]
    total = sum(raw)
    return [value / total for value in raw]


def _normalise(values: list[float]) -> list[float]:
    clean = [max(0.0, float(v)) for v in values]
    total = sum(clean)
    if total <= 0:
        return [1.0 / max(1, len(clean)) for _ in clean]
    return [v / total for v in clean]


def _renormalise_slice(values: list[float], start: int, end: int) -> None:
    normalised = _normalise(values[start:end])
    values[start:end] = normalised


__all__ = ["FactorEvolutionConfig", "FactorEvolutionResult", "run_factor_evolution"]
