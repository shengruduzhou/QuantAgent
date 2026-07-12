"""Leakage-aware GA for factor-blend weights.

This module is a research optimiser, not a production backtest. It trains a
factor-weight chromosome on each expanding walk-forward train fold and reports
its performance on the following OOS fold. Portfolio fitness is computed from
non-overlapping forward-return cohorts, with explicit label-horizon purging and
transaction-cost deductions.

The production acceptance path remains ``scripts/baseline_protocol.py``
variant C, which applies the full A-share execution model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from quantagent.optimization.multi_objective_loss import (
    LossComponents,
    LossWeights,
    compute_multi_objective_loss,
)


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 24
    generations: int = 10
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    mutation_sigma: float = 0.10
    elitism: int = 2
    top_k: int = 20
    random_seed: int = 17
    label_horizon_days: int = 1
    transaction_cost_bps: float = 8.0
    min_label_coverage: float = 0.80
    min_cohort_observations: int = 2

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be >= 4")
        if not 1 <= self.elitism < self.population_size:
            raise ValueError("elitism must be in [1, population_size)")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.label_horizon_days < 1:
            raise ValueError("label_horizon_days must be >= 1")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be >= 0")
        if not 0 < self.min_label_coverage <= 1:
            raise ValueError("min_label_coverage must be in (0, 1]")
        if self.min_cohort_observations < 2:
            raise ValueError("min_cohort_observations must be >= 2")


@dataclass(frozen=True)
class WalkForwardConfig:
    n_folds: int = 4
    embargo_days: int = 5
    label_horizon_days: int = 1
    min_train_days: int = 60
    min_test_days: int = 20

    @property
    def effective_gap_days(self) -> int:
        return max(int(self.embargo_days), int(self.label_horizon_days))

    def __post_init__(self) -> None:
        if self.n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")
        if self.label_horizon_days < 1:
            raise ValueError("label_horizon_days must be >= 1")
        if self.min_train_days < 2 or self.min_test_days < 2:
            raise ValueError("min_train_days and min_test_days must be >= 2")


@dataclass(frozen=True)
class GAFoldResult:
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_weights: dict[str, float]
    best_loss: float
    components: dict[str, float]


@dataclass(frozen=True)
class GAOptimizationResult:
    best_weights: dict[str, float]
    best_loss: float
    components: dict[str, float]
    fold_results: list[GAFoldResult]

    def to_dict(self) -> dict:
        return {
            "best_weights": dict(self.best_weights),
            "best_loss": float(self.best_loss),
            "components": dict(self.components),
            "fold_results": [
                {
                    "fold_index": row.fold_index,
                    "train_start": str(row.train_start),
                    "train_end": str(row.train_end),
                    "test_start": str(row.test_start),
                    "test_end": str(row.test_end),
                    "best_weights": dict(row.best_weights),
                    "best_loss": float(row.best_loss),
                    "components": dict(row.components),
                }
                for row in self.fold_results
            ],
        }


def purged_walk_forward_splits(
    dates: Sequence[pd.Timestamp],
    config: WalkForwardConfig,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Build expanding walk-forward folds with label-aware purging."""
    if len(dates) == 0:
        return []
    ordered = pd.DatetimeIndex(sorted({pd.Timestamp(value) for value in dates}))
    gap = config.effective_gap_days
    required = config.min_train_days + gap + config.min_test_days
    if len(ordered) < required:
        return []

    available_for_tests = len(ordered) - config.min_train_days - gap
    test_window = max(config.min_test_days, available_for_tests // config.n_folds)
    splits: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cursor = config.min_train_days
    for _ in range(config.n_folds):
        test_start_idx = cursor + gap
        if test_start_idx + config.min_test_days > len(ordered):
            break
        test_end_idx = min(len(ordered) - 1, test_start_idx + test_window - 1)
        splits.append(
            (
                ordered[0],
                ordered[cursor - 1],
                ordered[test_start_idx],
                ordered[test_end_idx],
            )
        )
        cursor = test_end_idx + 1
    return splits


def _normalise(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = float(clipped.sum())
    if total <= 0 or not np.isfinite(total):
        clipped = np.ones_like(clipped, dtype=float)
        total = float(clipped.sum())
    return clipped / total


def _initial_population(
    n_factors: int,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    population = rng.uniform(0.0, 1.0, size=(size, n_factors))
    return np.apply_along_axis(_normalise, 1, population)


def _validate_unique_keys(frame: pd.DataFrame, name: str) -> None:
    duplicated = frame.duplicated(["trade_date", "symbol"], keep=False)
    if bool(duplicated.any()):
        sample = frame.loc[duplicated, ["trade_date", "symbol"]].head(5).to_dict("records")
        raise ValueError(f"{name} has duplicate trade_date/symbol keys: {sample}")


def _aggregate_components(items: list[LossComponents]) -> LossComponents:
    if not items:
        return compute_multi_objective_loss(pd.Series(dtype=float))
    fields = tuple(items[0].as_dict().keys())
    medians = {
        field: float(np.median([item.as_dict()[field] for item in items]))
        for field in fields
    }
    return LossComponents(**medians)


def _cohort_loss(
    selected: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    horizon_days: int,
    transaction_cost_bps: float,
    loss_weights: LossWeights | None,
) -> LossComponents:
    cohort = selected[selected["trade_date"].isin(dates)].copy()
    if cohort.empty:
        return compute_multi_objective_loss(pd.Series(dtype=float))

    gross = cohort.groupby("trade_date", sort=True).apply(
        lambda group: float((group["forward_return"] * group["weight"]).sum()),
        include_groups=False,
    )
    weight_panel = cohort.pivot_table(
        index="trade_date",
        columns="symbol",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()
    turnover = weight_panel.diff().abs().sum(axis=1) / 2.0
    if not turnover.empty:
        turnover.iloc[0] = float(weight_panel.iloc[0].abs().sum())
    turnover = turnover.reindex(gross.index).fillna(0.0)

    cost_rate = turnover * (float(transaction_cost_bps) / 10_000.0)
    net = gross - cost_rate
    periods = max(1, int(round(252 / horizon_days)))
    return compute_multi_objective_loss(
        net,
        avg_daily_turnover=float(turnover.mean() / horizon_days),
        transaction_cost_rate=float(cost_rate.mean()),
        weights=loss_weights,
        periods=periods,
    )


def _evaluate_chromosome(
    weights: np.ndarray,
    factor_names: list[str],
    factor_panel: pd.DataFrame,
    forward_panel: pd.DataFrame,
    *,
    config: GAConfig,
    loss_weights: LossWeights | None,
) -> tuple[float, LossComponents]:
    """Score one chromosome without overlapping-return compounding."""
    required = ["trade_date", "symbol", *factor_names]
    work = factor_panel[required].copy()
    for column in factor_names:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=factor_names)
    if work.empty:
        empty = compute_multi_objective_loss(pd.Series(dtype=float))
        return float("inf"), empty

    work["score"] = work[factor_names].to_numpy(dtype=float) @ weights
    work = work.sort_values(
        ["trade_date", "score", "symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    work["rank"] = work.groupby("trade_date", sort=False).cumcount()
    selected = work[work["rank"] < config.top_k][["trade_date", "symbol", "score"]].copy()
    if selected.empty:
        empty = compute_multi_objective_loss(pd.Series(dtype=float))
        return float("inf"), empty

    labels = forward_panel[["trade_date", "symbol", "forward_return"]].copy()
    labels["forward_return"] = pd.to_numeric(labels["forward_return"], errors="coerce")
    selected = selected.merge(labels, on=["trade_date", "symbol"], how="left", validate="one_to_one")
    available = selected["forward_return"].notna()
    coverage = available.groupby(selected["trade_date"]).transform("mean")
    selected = selected[available & (coverage >= config.min_label_coverage)].copy()
    if selected.empty:
        empty = compute_multi_objective_loss(pd.Series(dtype=float))
        return float("inf"), empty

    labelled_counts = selected.groupby("trade_date")["symbol"].transform("size")
    selected["weight"] = 1.0 / labelled_counts.astype(float)
    valid_dates = pd.DatetimeIndex(sorted(selected["trade_date"].unique()))

    cohort_losses: list[LossComponents] = []
    horizon = config.label_horizon_days
    for offset in range(min(horizon, len(valid_dates))):
        cohort_dates = valid_dates[offset::horizon]
        if len(cohort_dates) < config.min_cohort_observations:
            continue
        cohort_losses.append(
            _cohort_loss(
                selected,
                cohort_dates,
                horizon_days=horizon,
                transaction_cost_bps=config.transaction_cost_bps,
                loss_weights=loss_weights,
            )
        )
    if not cohort_losses:
        empty = compute_multi_objective_loss(pd.Series(dtype=float))
        return float("inf"), empty
    components = _aggregate_components(cohort_losses)
    return float(components.total), components


def _crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = rng.random(parent_a.shape) < 0.5
    return _normalise(np.where(mask, parent_a, parent_b))


def _mutate(
    genome: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return _normalise(genome + rng.normal(0.0, sigma, size=genome.shape))


def optimize_factor_weights_ga(
    *,
    factor_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: Sequence[str],
    ga_config: GAConfig | None = None,
    wf_config: WalkForwardConfig | None = None,
    loss_weights: LossWeights | None = None,
) -> GAOptimizationResult:
    """Fit fold-local GA chromosomes and aggregate them without OOS selection."""
    ga_cfg = ga_config or GAConfig()
    wf_cfg = wf_config or WalkForwardConfig(label_horizon_days=ga_cfg.label_horizon_days)
    if wf_cfg.label_horizon_days != ga_cfg.label_horizon_days:
        raise ValueError("GAConfig and WalkForwardConfig label_horizon_days must match")

    names = list(factor_names)
    if not names:
        raise ValueError("factor_names is empty")
    missing = [name for name in names if name not in factor_panel.columns]
    if missing:
        raise ValueError(f"factor_panel missing columns: {missing}")

    fp = factor_panel.copy()
    fp["trade_date"] = pd.to_datetime(fp["trade_date"], errors="coerce")
    fp = fp.dropna(subset=["trade_date", "symbol"]).reset_index(drop=True)
    fr = forward_returns.copy()
    fr["trade_date"] = pd.to_datetime(fr["trade_date"], errors="coerce")
    fr = fr.dropna(subset=["trade_date", "symbol"]).reset_index(drop=True)
    if "forward_return" not in fr.columns:
        raise ValueError("forward_returns must include 'forward_return'")
    _validate_unique_keys(fp, "factor_panel")
    _validate_unique_keys(fr, "forward_returns")

    splits = purged_walk_forward_splits(sorted(fp["trade_date"].unique()), wf_cfg)
    if not splits:
        raise ValueError("not enough dates for the requested label-aware walk-forward configuration")

    rng = np.random.default_rng(ga_cfg.random_seed)
    fold_results: list[GAFoldResult] = []
    fold_chromosomes: list[np.ndarray] = []
    fold_components: list[LossComponents] = []

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(splits):
        train_panel = fp[fp["trade_date"].between(train_start, train_end)].reset_index(drop=True)
        train_fwd = fr[fr["trade_date"].between(train_start, train_end)].reset_index(drop=True)
        test_panel = fp[fp["trade_date"].between(test_start, test_end)].reset_index(drop=True)
        test_fwd = fr[fr["trade_date"].between(test_start, test_end)].reset_index(drop=True)
        if train_panel.empty or test_panel.empty:
            continue

        population = _initial_population(len(names), ga_cfg.population_size, rng)
        for _ in range(ga_cfg.generations):
            fitness = np.array(
                [
                    _evaluate_chromosome(
                        chromosome,
                        names,
                        train_panel,
                        train_fwd,
                        config=ga_cfg,
                        loss_weights=loss_weights,
                    )[0]
                    for chromosome in population
                ],
                dtype=float,
            )
            order = np.argsort(fitness)
            elites = population[order[: ga_cfg.elitism]]
            parent_pool = order[: max(2, ga_cfg.population_size // 2)]
            next_population = [row.copy() for row in elites]
            while len(next_population) < ga_cfg.population_size:
                first, second = rng.choice(parent_pool, size=2, replace=False)
                child = (
                    _crossover(population[first], population[second], rng)
                    if rng.random() < ga_cfg.crossover_rate
                    else population[first].copy()
                )
                if rng.random() < ga_cfg.mutation_rate:
                    child = _mutate(child, ga_cfg.mutation_sigma, rng)
                next_population.append(child)
            population = np.asarray(next_population, dtype=float)

        train_fitness = np.array(
            [
                _evaluate_chromosome(
                    chromosome,
                    names,
                    train_panel,
                    train_fwd,
                    config=ga_cfg,
                    loss_weights=loss_weights,
                )[0]
                for chromosome in population
            ],
            dtype=float,
        )
        if not np.isfinite(train_fitness).any():
            continue
        best = population[int(np.nanargmin(train_fitness))]
        oos_loss, oos_components = _evaluate_chromosome(
            best,
            names,
            test_panel,
            test_fwd,
            config=ga_cfg,
            loss_weights=loss_weights,
        )
        if not np.isfinite(oos_loss):
            continue

        fold_chromosomes.append(best)
        fold_components.append(oos_components)
        fold_results.append(
            GAFoldResult(
                fold_index=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_weights={name: float(weight) for name, weight in zip(names, best)},
                best_loss=float(oos_loss),
                components=oos_components.as_dict(),
            )
        )

    if not fold_chromosomes:
        raise ValueError("all walk-forward folds were unevaluable after integrity gates")

    aggregate = _normalise(np.median(np.vstack(fold_chromosomes), axis=0))
    components = _aggregate_components(fold_components)
    return GAOptimizationResult(
        best_weights={name: float(weight) for name, weight in zip(names, aggregate)},
        best_loss=float(components.total),
        components=components.as_dict(),
        fold_results=fold_results,
    )


def save_optimisation_artifacts(
    result: GAOptimizationResult,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "factor_weights": target / "factor_weights.json",
        "walk_forward_backtest": target / "walk_forward_backtest.json",
        "metrics": target / "metrics.json",
    }
    paths["factor_weights"].write_text(
        json.dumps(result.best_weights, indent=2), encoding="utf-8"
    )
    paths["walk_forward_backtest"].write_text(
        json.dumps(
            [
                {
                    "fold_index": row.fold_index,
                    "train_start": str(row.train_start),
                    "train_end": str(row.train_end),
                    "test_start": str(row.test_start),
                    "test_end": str(row.test_end),
                    "best_weights": row.best_weights,
                    "best_loss": row.best_loss,
                    "components": row.components,
                }
                for row in result.fold_results
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["metrics"].write_text(
        json.dumps(
            {
                "best_loss": result.best_loss,
                "components": result.components,
                "aggregation": "coordinate_median_of_fold_train_champions",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "GAConfig",
    "GAFoldResult",
    "GAOptimizationResult",
    "WalkForwardConfig",
    "optimize_factor_weights_ga",
    "purged_walk_forward_splits",
    "save_optimisation_artifacts",
]
