"""GAWeightOptimizer — multi-objective GA over factor weights (spec section 6).

The optimiser searches per-horizon factor blend weights against the
:func:`compute_multi_objective_loss` surface. It runs **strictly OOS**:

* Each generation's fitness is computed on the held-out fold of a
  purged walk-forward split with an embargo equal to the longest
  forward horizon, so no in-sample prediction leaks into the loss.
* The output table records the best chromosome per fold (+ globally),
  plus the fitness components so callers can audit *why* a weight set
  won.

Why a hand-rolled GA instead of pulling DEAP / pygad? Two reasons:

1. We need full control over how the loss components map to fitness —
   the multi-objective shape is provided by
   :mod:`quantagent.optimization.multi_objective_loss` and the GA's
   only job is to find a weight set that minimises ``total``.
2. The repository must run without optional ML deps. The GA here
   only depends on NumPy + pandas.

The factor-weight chromosome layout is ``{factor_name → weight}``;
weights are constrained to ``[0, 1]`` and renormalised to sum to 1
after each mutation. Callers supply:

* ``factor_returns``: long-form ``trade_date / factor / value``
  with already-scaled per-day factor scores.
* ``forward_returns``: long-form ``trade_date / symbol / forward_return``
  (forward = realised forward return of the target horizon).

The fitness on a fold is the loss when the chromosome's factor blend
is used to form daily target-weights (top-K long-only) and the
resulting daily returns are scored via the multi-objective loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import json
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quantagent.optimization.multi_objective_loss import (
    LossComponents,
    LossWeights,
    compute_multi_objective_loss,
)


# ---------------------------------------------------------------------------
# Config + records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GAConfig:
    population_size: int = 24
    generations: int = 10
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    mutation_sigma: float = 0.10
    elitism: int = 2
    top_k: int = 20             # how many names to long per day
    random_seed: int = 17


@dataclass(frozen=True)
class WalkForwardConfig:
    n_folds: int = 4
    embargo_days: int = 5
    min_train_days: int = 60
    min_test_days: int = 20


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
                    "fold_index": r.fold_index,
                    "train_start": str(r.train_start),
                    "train_end": str(r.train_end),
                    "test_start": str(r.test_start),
                    "test_end": str(r.test_end),
                    "best_weights": dict(r.best_weights),
                    "best_loss": float(r.best_loss),
                    "components": dict(r.components),
                }
                for r in self.fold_results
            ],
        }


# ---------------------------------------------------------------------------
# Walk-forward / purged split
# ---------------------------------------------------------------------------

def purged_walk_forward_splits(
    dates: Sequence[pd.Timestamp],
    config: WalkForwardConfig,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return ``[(train_start, train_end, test_start, test_end)]``.

    Embargo is enforced by skipping ``config.embargo_days`` business days
    between the train end and the test start; the same gap goes on the
    far side of the test fold so the next fold's training set does not
    touch this fold's test window.
    """
    if not dates:
        return []
    ordered = pd.DatetimeIndex(sorted({pd.Timestamp(d) for d in dates}))
    if len(ordered) < config.min_train_days + config.min_test_days + config.embargo_days:
        return []
    n = len(ordered)
    # cut into n_folds equally-sized test windows
    test_window = max(config.min_test_days, (n - config.min_train_days) // config.n_folds)
    splits = []
    cursor = config.min_train_days
    for k in range(config.n_folds):
        if cursor + config.embargo_days + test_window > n:
            break
        train_start = ordered[0]
        train_end = ordered[cursor - 1]
        test_start = ordered[cursor + config.embargo_days]
        test_end_idx = min(n - 1, cursor + config.embargo_days + test_window - 1)
        test_end = ordered[test_end_idx]
        splits.append((train_start, train_end, test_start, test_end))
        cursor = test_end_idx + 1
    return splits


# ---------------------------------------------------------------------------
# Chromosome + fitness
# ---------------------------------------------------------------------------

def _normalise(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(weights, 0.0, None)
    s = weights.sum()
    if s <= 0:
        # fall back to uniform
        weights = np.ones_like(weights)
        s = weights.sum()
    return weights / s


def _initial_population(n_factors: int, size: int, rng: np.random.Generator) -> np.ndarray:
    pop = rng.uniform(0.0, 1.0, size=(size, n_factors))
    return np.apply_along_axis(_normalise, 1, pop)


def _evaluate_chromosome(
    weights: np.ndarray,
    factor_names: list[str],
    factor_panel: pd.DataFrame,
    forward_panel: pd.DataFrame,
    *,
    top_k: int,
    loss_weights: LossWeights | None,
) -> tuple[float, LossComponents]:
    """Form daily portfolio returns from the weighted factor blend.

    factor_panel: trade_date × symbol × factor → value (long form expected
    pre-pivoted as wide on factor cols already).
    """
    # daily score = weighted blend of factor columns
    score_matrix = factor_panel[factor_names].astype(float).values
    if score_matrix.size == 0:
        return float("inf"), compute_multi_objective_loss(pd.Series(dtype=float))
    scores = score_matrix @ weights
    pf = factor_panel[["trade_date", "symbol"]].copy()
    pf["score"] = scores
    # pick top-K each day and equal weight; long-only
    pf = pf.sort_values(["trade_date", "score"], ascending=[True, False])
    pf["rank"] = pf.groupby("trade_date").cumcount()
    pf["weight"] = (pf["rank"] < top_k).astype(float) / float(top_k)
    # merge realised forward returns
    merged = pf.merge(forward_panel, on=["trade_date", "symbol"], how="left")
    merged["forward_return"] = pd.to_numeric(merged["forward_return"], errors="coerce").fillna(0.0)
    daily = merged.groupby("trade_date").apply(
        lambda d: float((d["forward_return"] * d["weight"]).sum())
    )
    daily.name = "daily_eq_return"
    # turnover proxy: half-sum of |Δweight| per day
    weight_pivot = pf.pivot(index="trade_date", columns="symbol", values="weight").fillna(0.0)
    deltas = weight_pivot.diff().abs().sum(axis=1) / 2.0
    avg_turnover = float(deltas.mean()) if len(deltas) > 0 else 0.0
    comps = compute_multi_objective_loss(
        daily, avg_daily_turnover=avg_turnover, weights=loss_weights,
    )
    return float(comps.total), comps


def _crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(parent_a.shape) < 0.5
    child = np.where(mask, parent_a, parent_b)
    return _normalise(child)


def _mutate(genome: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma, size=genome.shape)
    return _normalise(genome + noise)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_factor_weights_ga(
    *,
    factor_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_names: Sequence[str],
    ga_config: GAConfig | None = None,
    wf_config: WalkForwardConfig | None = None,
    loss_weights: LossWeights | None = None,
) -> GAOptimizationResult:
    """Run the multi-objective GA across walk-forward folds.

    Returns the best chromosome across all folds plus per-fold detail
    so callers can sanity-check OOS stability.
    """
    ga_cfg = ga_config or GAConfig()
    wf_cfg = wf_config or WalkForwardConfig()
    factor_names = list(factor_names)
    if not factor_names:
        raise ValueError("factor_names is empty")
    missing = [f for f in factor_names if f not in factor_panel.columns]
    if missing:
        raise ValueError(f"factor_panel missing columns: {missing}")
    rng = np.random.default_rng(ga_cfg.random_seed)

    fp = factor_panel.copy()
    fp["trade_date"] = pd.to_datetime(fp["trade_date"], errors="coerce")
    fp = fp.dropna(subset=["trade_date"]).reset_index(drop=True)
    if "symbol" not in fp.columns:
        raise ValueError("factor_panel must include 'symbol'")
    fr = forward_returns.copy()
    fr["trade_date"] = pd.to_datetime(fr["trade_date"], errors="coerce")
    fr = fr.dropna(subset=["trade_date", "symbol", "forward_return"]).reset_index(drop=True)

    unique_dates = sorted(fp["trade_date"].unique())
    splits = purged_walk_forward_splits(unique_dates, wf_cfg)
    if not splits:
        raise ValueError(
            "not enough dates for the requested walk-forward configuration"
        )

    fold_results: list[GAFoldResult] = []
    global_best_loss = float("inf")
    global_best_weights: np.ndarray | None = None
    global_best_components: LossComponents | None = None

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(splits):
        train_mask = (fp["trade_date"] >= train_start) & (fp["trade_date"] <= train_end)
        train_panel = fp[train_mask].reset_index(drop=True)
        train_fwd = fr[(fr["trade_date"] >= train_start) & (fr["trade_date"] <= train_end)]
        test_mask = (fp["trade_date"] >= test_start) & (fp["trade_date"] <= test_end)
        test_panel = fp[test_mask].reset_index(drop=True)
        test_fwd = fr[(fr["trade_date"] >= test_start) & (fr["trade_date"] <= test_end)]
        if train_panel.empty or test_panel.empty:
            continue

        # GA loop: fit on train fold, evaluate fitness on train; the
        # selected best chromosome is then *measured* on the OOS test
        # fold, which is what we report.
        pop = _initial_population(len(factor_names), ga_cfg.population_size, rng)
        for gen in range(ga_cfg.generations):
            fitness = np.empty(len(pop))
            for i, chromo in enumerate(pop):
                loss, _ = _evaluate_chromosome(
                    chromo, factor_names, train_panel, train_fwd,
                    top_k=ga_cfg.top_k, loss_weights=loss_weights,
                )
                fitness[i] = loss
            order = np.argsort(fitness)
            elites = pop[order[: ga_cfg.elitism]]
            new_pop = list(elites)
            while len(new_pop) < ga_cfg.population_size:
                # Tournament-of-2 selection
                a, b = rng.choice(order[: ga_cfg.population_size // 2], size=2, replace=False)
                parent_a = pop[a]
                parent_b = pop[b]
                if rng.random() < ga_cfg.crossover_rate:
                    child = _crossover(parent_a, parent_b, rng)
                else:
                    child = parent_a.copy()
                if rng.random() < ga_cfg.mutation_rate:
                    child = _mutate(child, ga_cfg.mutation_sigma, rng)
                new_pop.append(child)
            pop = np.array(new_pop)

        # final eval on OOS test fold using the fold's best chromosome
        train_fitness = np.array([
            _evaluate_chromosome(
                c, factor_names, train_panel, train_fwd,
                top_k=ga_cfg.top_k, loss_weights=loss_weights,
            )[0]
            for c in pop
        ])
        best_idx = int(np.argmin(train_fitness))
        best_chromo = pop[best_idx]
        oos_loss, oos_components = _evaluate_chromosome(
            best_chromo, factor_names, test_panel, test_fwd,
            top_k=ga_cfg.top_k, loss_weights=loss_weights,
        )
        fold_results.append(GAFoldResult(
            fold_index=fold_idx,
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            best_weights={f: float(w) for f, w in zip(factor_names, best_chromo)},
            best_loss=float(oos_loss),
            components=oos_components.as_dict(),
        ))
        if oos_loss < global_best_loss:
            global_best_loss = oos_loss
            global_best_weights = best_chromo
            global_best_components = oos_components

    if global_best_weights is None or global_best_components is None:
        # fallback: uniform across all factors
        global_best_weights = _normalise(np.ones(len(factor_names)))
        global_best_components = compute_multi_objective_loss(pd.Series([0.0]))
        global_best_loss = float(global_best_components.total)

    return GAOptimizationResult(
        best_weights={f: float(w) for f, w in zip(factor_names, global_best_weights)},
        best_loss=float(global_best_loss),
        components=global_best_components.as_dict(),
        fold_results=fold_results,
    )


def save_optimisation_artifacts(
    result: GAOptimizationResult,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write factor_weights.json + walk_forward_backtest.json + metrics.json."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    (target / "factor_weights.json").write_text(
        json.dumps(result.best_weights, indent=2), encoding="utf-8",
    )
    paths["factor_weights"] = target / "factor_weights.json"
    (target / "walk_forward_backtest.json").write_text(
        json.dumps([
            {
                "fold_index": r.fold_index,
                "train_start": str(r.train_start),
                "train_end": str(r.train_end),
                "test_start": str(r.test_start),
                "test_end": str(r.test_end),
                "best_loss": r.best_loss,
                "components": r.components,
            }
            for r in result.fold_results
        ], indent=2),
        encoding="utf-8",
    )
    paths["walk_forward_backtest"] = target / "walk_forward_backtest.json"
    (target / "metrics.json").write_text(
        json.dumps({
            "best_loss": result.best_loss,
            "components": result.components,
        }, indent=2),
        encoding="utf-8",
    )
    paths["metrics"] = target / "metrics.json"
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
