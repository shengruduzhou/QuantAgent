"""Genetic-programming symbolic regression for factor discovery.

Evolves expression trees in the :mod:`quantagent.factors.expr` DSL to
maximise rank-IC against forward-return labels. Output: top-K
discovered :class:`~quantagent.factors.expr.FactorDefinition`s that
can be registered alongside Alpha101.

Unlike :mod:`quantagent.optimization.factor_evolution`, which evolves
binary masks over **existing** columns, this module discovers **new
formulas** the human authors did not write.

Operators are restricted to the DSL primitives in
:mod:`quantagent.factors.expr` so the no-look-ahead and per-symbol
time-series guarantees are preserved automatically.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from quantagent.factors import expr as E

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Operator and terminal pools                                                 #
# --------------------------------------------------------------------------- #

_BINARY_OPS: tuple = (E.Add, E.Sub, E.Mul, E.Div)
_UNARY_OPS: tuple = (E.Abs, E.Sign, E.Log, E.CsZscore)
# Time-series operators are listed as (class_or_factory, takes_window: bool).
_TS_OPS: tuple = (E.TsMean, E.TsStd, E.TsSum, E.TsMax, E.TsMin, E.TsRank, E.DecayLinear)
_TS_BINARY_OPS: tuple = (E.TsCorr, E.TsCov)
_DELAY_DELTA_OPS: tuple = (E.Delay, E.Delta)
_TS_WINDOWS: tuple[int, ...] = (3, 5, 10, 20, 30, 60)
_DELAY_PERIODS: tuple[int, ...] = (1, 2, 3, 5, 10)

_PRICE_TERMINALS: tuple = (E.Open, E.High, E.Low, E.Close, E.Vwap)
_VOLUME_TERMINALS: tuple = (E.Volume, E.Amount)
_CONSTANTS: tuple = tuple(E.Constant(c) for c in (0.5, 1.0, 2.0, 5.0))

_MARKET_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SymbolicGAConfig:
    population: int = 80
    generations: int = 20
    max_depth: int = 4
    min_depth: int = 2
    tournament_size: int = 5
    crossover_prob: float = 0.7
    mutation_prob: float = 0.25
    elitism: int = 4
    top_k: int = 20
    label_column: str = "forward_return_5d"
    complexity_penalty: float = 5e-4
    min_finite_ratio: float = 0.3
    max_correlation: float = 0.85
    validation_fraction: float = 0.25
    min_validation_rank_ic: float = 0.0
    fitness_sample_dates: int = 400
    fitness_sample_symbols: int = 500
    seed: int = 1729
    # Fraction of the initial population seeded from Alpha101-style templates
    # (and mutated variants of them) instead of pure random trees.
    warm_start_fraction: float = 0.4
    # Weight of |daily ICIR| in fitness, alongside |mean rank-IC|.
    icir_weight: float = 0.05
    # Existing factor columns (must be present in the merged panel) that
    # candidates are decorrelated against at selection time.
    reference_columns: tuple[str, ...] = ()
    max_reference_correlation: float = 0.7
    # Fraction of each new generation replaced by fresh random trees to
    # keep diversity (quality-diversity style anti-premature-convergence).
    random_injection_rate: float = 0.10


@dataclass(frozen=True)
class SynthesisResult:
    definitions: list[E.FactorDefinition]
    leaderboard: pd.DataFrame  # name, expression, train_rank_ic, validation_rank_ic, complexity, finite_ratio
    history: pd.DataFrame      # generation, best_fitness, mean_fitness, mean_complexity


# --------------------------------------------------------------------------- #
# Tree generation                                                             #
# --------------------------------------------------------------------------- #


def _random_terminal(rng: random.Random) -> E.Expr:
    pool: list = list(_PRICE_TERMINALS) + list(_VOLUME_TERMINALS)
    if rng.random() < 0.15:
        return rng.choice(_CONSTANTS)
    return rng.choice(pool)


def _random_tree(rng: random.Random, depth: int, *, force_internal: bool = False) -> E.Expr:
    """Sample a random expression tree of bounded depth."""
    if depth <= 0:
        return _random_terminal(rng)
    if not force_internal and rng.random() < 0.18:
        return _random_terminal(rng)

    roll = rng.random()
    if roll < 0.28:
        op = rng.choice(_BINARY_OPS)
        left = _random_tree(rng, depth - 1)
        right = _random_tree(rng, depth - 1)
        return op(left, right)
    if roll < 0.38:
        op = rng.choice(_UNARY_OPS)
        return op(_random_tree(rng, depth - 1))
    if roll < 0.68:
        op = rng.choice(_TS_OPS)
        window = rng.choice(_TS_WINDOWS)
        return op(_random_tree(rng, depth - 1), window)
    if roll < 0.78:
        op = rng.choice(_TS_BINARY_OPS)
        window = rng.choice(_TS_WINDOWS)
        return op(_random_tree(rng, depth - 1), _random_tree(rng, depth - 1), window)
    if roll < 0.90:
        op = rng.choice(_DELAY_DELTA_OPS)
        period = rng.choice(_DELAY_PERIODS)
        return op(_random_tree(rng, depth - 1), period)
    return E.Rank(_random_tree(rng, depth - 1))


def _warm_start_templates() -> list[E.Expr]:
    """Alpha101/191-style structural seeds known to carry signal in A-shares.

    These encode reversal, volume-price divergence, decayed momentum,
    liquidity, intraday positioning and volatility structures so the GA
    starts from financially meaningful regions of formula space instead
    of pure noise.
    """
    O, H, L, C, V, A = E.Open, E.High, E.Low, E.Close, E.Volume, E.Amount
    vwap = E.Vwap
    ret1 = E.Returns(C, 1)
    ret5 = E.Returns(C, 5)
    return [
        # Short-term reversal family.
        E.Mul(E.Constant(-1.0), E.Rank(ret5)),
        E.Mul(E.Constant(-1.0), E.Rank(E.Delta(C, 3))),
        E.Mul(E.Constant(-1.0), E.Rank(E.TsMean(ret1, 5))),
        # Volume-price divergence / correlation (alpha101 staples).
        E.Mul(E.Constant(-1.0), E.TsCorr(E.Rank(C), E.Rank(V), 10)),
        E.Mul(E.Constant(-1.0), E.TsCorr(E.Rank(O), E.Rank(V), 10)),
        E.Sub(E.Rank(E.Delta(C, 5)), E.Rank(E.Delta(V, 5))),
        E.TsCorr(C, V, 20),
        # Decayed momentum / smoothed reversal.
        E.Mul(E.Constant(-1.0), E.DecayLinear(E.Delta(C, 3), 10)),
        E.DecayLinear(E.Rank(ret5), 10),
        E.Rank(E.Sub(C, E.TsMean(C, 20))),
        # Intraday positioning.
        E.Div(E.Sub(C, L), E.Sub(H, L)),
        E.Div(E.Sub(C, O), E.Sub(H, L)),
        E.Rank(E.Div(E.Sub(C, vwap), vwap)),
        E.Mul(E.Constant(-1.0), E.Rank(E.Div(E.Sub(H, C), E.Sub(H, L)))),
        # Volatility / range structure.
        E.Mul(E.Constant(-1.0), E.Rank(E.TsStd(ret1, 20))),
        E.Rank(E.TsMean(E.Div(E.Sub(H, L), C), 5)),
        E.TsRank(E.Div(E.Sub(H, L), C), 10),
        # Liquidity / turnover discount.
        E.Mul(E.Constant(-1.0), E.Rank(E.Log(E.TsMean(A, 20)))),
        E.Mul(E.Constant(-1.0), E.Rank(E.Div(V, E.TsMean(V, 20)))),
        E.Div(E.TsStd(A, 20), E.TsMean(A, 20)),
        # Volume shock / crowding.
        E.Div(E.Sub(V, E.TsMean(V, 5)), E.TsStd(V, 5)),
        E.Mul(E.Constant(-1.0), E.Rank(E.TsRank(V, 5))),
        # Gap / overnight behaviour.
        E.Rank(E.Sub(O, E.Delay(C, 1))),
        E.Mul(E.Constant(-1.0), E.Rank(E.Sub(E.TsMax(H, 5), C))),
        # Price level vs anchor (52w-high style on shorter horizon).
        E.Div(C, E.TsMax(H, 60)),
        E.CsZscore(E.Div(C, E.TsMean(C, 60))),
    ]


def _seed_population(rng: random.Random, cfg: SymbolicGAConfig) -> list[E.Expr]:
    """Build the initial population: warm-start templates + mutants + random."""
    population: list[E.Expr] = []
    warm_n = max(0, min(cfg.population, int(round(cfg.population * cfg.warm_start_fraction))))
    templates = _warm_start_templates()
    rng.shuffle(templates)
    for i in range(warm_n):
        base = templates[i % len(templates)]
        # First pass keeps the pristine template, later passes mutate it.
        population.append(base if i < len(templates) else _mutate(base, rng, cfg.max_depth))
    while len(population) < cfg.population:
        population.append(_random_tree(rng, cfg.max_depth, force_internal=True))
    return population


# --------------------------------------------------------------------------- #
# Tree introspection                                                          #
# --------------------------------------------------------------------------- #


def _node_count(node: E.Expr) -> int:
    """Count nodes in a frozen-dataclass tree."""
    return 1 + sum(_node_count(child) for child in _children(node))


def _children(node: E.Expr) -> list[E.Expr]:
    return E.expr_children(node)


def _uses_market_terminal(node: E.Expr) -> bool:
    """Return True when an expression depends on real market input columns."""
    if isinstance(node, E.Column):
        return node.name in set(_MARKET_COLUMNS)
    return any(_uses_market_terminal(child) for child in _children(node))


def _rebuild(node: E.Expr, new_children: list[E.Expr]) -> E.Expr:
    """Reconstruct ``node`` with its Expr-valued fields replaced in order."""
    from dataclasses import fields

    iterator = iter(new_children)
    kwargs = {}
    for f in fields(node):
        value = getattr(node, f.name)
        kwargs[f.name] = next(iterator) if isinstance(value, E.Expr) else value
    return type(node)(**kwargs)


def _replace_subtree(node: E.Expr, target_id: int, replacement: E.Expr, counter: list[int]) -> E.Expr:
    """Return a copy of ``node`` with the ``target_id``-th visited subtree swapped out."""
    counter[0] += 1
    if counter[0] == target_id:
        return replacement
    children = _children(node)
    if not children:
        return node
    new_children = [_replace_subtree(child, target_id, replacement, counter) for child in children]
    return _rebuild(node, new_children)


# --------------------------------------------------------------------------- #
# Genetic operators                                                           #
# --------------------------------------------------------------------------- #


def _crossover(t1: E.Expr, t2: E.Expr, rng: random.Random) -> tuple[E.Expr, E.Expr]:
    n1, n2 = _node_count(t1), _node_count(t2)
    if n1 < 2 or n2 < 2:
        return t1, t2
    a = rng.randrange(2, n1 + 1)
    b = rng.randrange(2, n2 + 1)
    sub1 = _extract_subtree(t1, a, [0])
    sub2 = _extract_subtree(t2, b, [0])
    if sub1 is None or sub2 is None:
        return t1, t2
    c1 = _replace_subtree(t1, a, sub2, [0])
    c2 = _replace_subtree(t2, b, sub1, [0])
    return c1, c2


def _extract_subtree(node: E.Expr, target_id: int, counter: list[int]) -> E.Expr | None:
    counter[0] += 1
    if counter[0] == target_id:
        return node
    for child in _children(node):
        result = _extract_subtree(child, target_id, counter)
        if result is not None:
            return result
    return None


def _mutate(tree: E.Expr, rng: random.Random, max_depth: int) -> E.Expr:
    n = _node_count(tree)
    if n < 2:
        return _random_tree(rng, max_depth)
    target = rng.randrange(2, n + 1)
    replacement = _random_tree(rng, max(1, max_depth - 2))
    return _replace_subtree(tree, target, replacement, [0])


# --------------------------------------------------------------------------- #
# Fitness                                                                     #
# --------------------------------------------------------------------------- #


def _daily_rank_ic(factor_values: pd.Series, labels: pd.Series, trade_date: pd.Series) -> pd.Series:
    """Daily cross-sectional Spearman rank correlation series."""
    df = pd.DataFrame(
        {
            "trade_date": trade_date.values,
            "factor": factor_values.values,
            "label": labels.values,
        }
    ).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    grouped = df.groupby("trade_date")
    rank_factor = grouped["factor"].rank(method="average")
    rank_label = grouped["label"].rank(method="average")
    df = df.assign(rank_factor=rank_factor.values, rank_label=rank_label.values)
    daily = (
        df.groupby("trade_date")[["rank_factor", "rank_label"]]
        .corr()
        .unstack()
        .iloc[:, 1]
        .dropna()
    )
    return daily


def _rank_ic(factor_values: pd.Series, labels: pd.Series, trade_date: pd.Series) -> float:
    """Mean of daily cross-sectional Spearman rank correlations."""
    daily = _daily_rank_ic(factor_values, labels, trade_date)
    if daily.empty:
        return 0.0
    return float(daily.mean())


def _ic_stats(factor_values: pd.Series, labels: pd.Series, trade_date: pd.Series) -> tuple[float, float]:
    """Return (mean rank-IC, daily ICIR)."""
    daily = _daily_rank_ic(factor_values, labels, trade_date)
    if daily.empty:
        return 0.0, 0.0
    mean = float(daily.mean())
    std = float(daily.std(ddof=0))
    icir = mean / std if std > 1e-12 else 0.0
    return mean, icir


def _evaluate_ic(
    tree: E.Expr,
    panel: pd.DataFrame,
    labels: pd.Series,
    trade_date: pd.Series,
) -> tuple[float, float]:
    """Return (rank-IC, finite_ratio) for one tree."""
    try:
        values = tree.evaluate(panel)
    except Exception:
        return 0.0, 0.0
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite_ratio = float(values.notna().mean())
    ic = _rank_ic(values, labels, trade_date)
    return ic, finite_ratio


def _evaluate_fitness(
    tree: E.Expr,
    panel: pd.DataFrame,
    labels: pd.Series,
    trade_date: pd.Series,
    cfg: SymbolicGAConfig,
) -> tuple[float, float, float]:
    """Return (fitness, raw_rank_ic, finite_ratio) for one tree.

    Fitness rewards both the level of the rank-IC and its day-to-day
    stability (daily ICIR), and penalises formula complexity. Absolute
    values are used because a stable negative IC is recoverable by sign
    inversion at selection time.
    """
    if not _uses_market_terminal(tree):
        return -1.0, 0.0, 0.0
    try:
        values = tree.evaluate(panel)
    except Exception:
        return -1.0, 0.0, 0.0
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite_ratio = float(values.notna().mean())
    if finite_ratio < cfg.min_finite_ratio:
        return -1.0, 0.0, finite_ratio
    ic_mean, icir = _ic_stats(values, labels, trade_date)
    score = abs(ic_mean) + cfg.icir_weight * abs(icir) - cfg.complexity_penalty * _node_count(tree)
    return score, ic_mean, finite_ratio


def _chronological_split(
    panel: pd.DataFrame,
    labels: pd.Series,
    trade_date: pd.Series,
    cfg: SymbolicGAConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    dates = pd.Series(pd.to_datetime(trade_date).dropna().unique()).sort_values().to_list()
    if len(dates) < 4 or cfg.validation_fraction <= 0:
        return panel, labels, trade_date, panel, labels, trade_date
    valid_count = max(1, int(round(len(dates) * min(cfg.validation_fraction, 0.5))))
    cutoff_dates = set(dates[-valid_count:])
    mask = pd.to_datetime(trade_date).isin(cutoff_dates)
    train_panel = panel.loc[~mask].reset_index(drop=True)
    valid_panel = panel.loc[mask].reset_index(drop=True)
    train_labels = labels.loc[~mask].reset_index(drop=True)
    valid_labels = labels.loc[mask].reset_index(drop=True)
    train_dates = trade_date.loc[~mask].reset_index(drop=True)
    valid_dates = trade_date.loc[mask].reset_index(drop=True)
    if train_panel.empty or valid_panel.empty:
        return panel, labels, trade_date, panel, labels, trade_date
    return train_panel, train_labels, train_dates, valid_panel, valid_labels, valid_dates


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def _sample_contiguous_dates(dates: list, n_dates: int, rng: random.Random) -> set:
    """Pick a random contiguous block of ``n_dates`` trading dates.

    Contiguity matters: rolling/delay operators computed over randomly
    sampled (gapped) dates silently produce wrong window contents.
    """
    if n_dates >= len(dates):
        return set(dates)
    start = rng.randrange(0, len(dates) - n_dates + 1)
    return set(dates[start : start + n_dates])


def _subsample_panel(
    panel: pd.DataFrame,
    label_col: str,
    cfg: SymbolicGAConfig,
    rng: random.Random,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Sub-sample dates and symbols to keep fitness evaluation tractable."""
    if label_col not in panel.columns:
        raise KeyError(f"label column '{label_col}' missing from panel")
    valid = panel[panel[label_col].notna()].copy()
    valid["trade_date"] = pd.to_datetime(valid["trade_date"])
    dates = sorted(valid["trade_date"].unique())
    if cfg.fitness_sample_dates and len(dates) > cfg.fitness_sample_dates:
        chosen_dates = _sample_contiguous_dates(dates, cfg.fitness_sample_dates, rng)
        valid = valid[valid["trade_date"].isin(chosen_dates)]
    symbols = valid["symbol"].unique()
    if cfg.fitness_sample_symbols and len(symbols) > cfg.fitness_sample_symbols:
        chosen_symbols = rng.sample(list(symbols), cfg.fitness_sample_symbols)
        valid = valid[valid["symbol"].isin(chosen_symbols)]
    valid = valid.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    return valid, valid[label_col], valid["trade_date"]


def synthesize_factors(
    panel: pd.DataFrame,
    labels: pd.DataFrame | None = None,
    config: SymbolicGAConfig | None = None,
) -> SynthesisResult:
    """Run the symbolic regression GA and return top-K discovered factors.

    Parameters
    ----------
    panel
        Wide market-panel frame with at minimum ``symbol``, ``trade_date``,
        ``open``, ``high``, ``low``, ``close``, ``volume``, ``amount``.
    labels
        Optional labels frame keyed on (symbol, trade_date) containing
        ``forward_return_*`` columns. If absent, the function looks for
        ``forward_return_5d`` (or ``cfg.label_column``) directly in ``panel``.
    """
    cfg = config or SymbolicGAConfig()
    rng = random.Random(cfg.seed)
    panel = panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    if labels is not None and not labels.empty:
        labels = labels.copy()
        labels["trade_date"] = pd.to_datetime(labels["trade_date"], errors="coerce")
        # Keep only the join keys, the label and any requested reference
        # factor columns. Carrying the full labels frame into the merge
        # used to suffix overlapping OHLCV columns (close -> close_x/_y),
        # which silently killed every expression evaluation.
        keep = ["symbol", "trade_date"]
        if cfg.label_column in labels.columns:
            keep.append(cfg.label_column)
            labels = labels[labels[cfg.label_column].notna()]
        for ref in cfg.reference_columns:
            if ref in labels.columns and ref not in panel.columns and ref not in keep:
                keep.append(ref)
        labels = labels[[c for c in keep if c in labels.columns]]
        if cfg.fitness_sample_dates and labels["trade_date"].nunique() > cfg.fitness_sample_dates:
            sampled_dates = _sample_contiguous_dates(
                sorted(labels["trade_date"].dropna().unique()), cfg.fitness_sample_dates, rng
            )
            labels = labels[labels["trade_date"].isin(sampled_dates)]
            panel = panel[panel["trade_date"].isin(sampled_dates)]
        if cfg.fitness_sample_symbols and labels["symbol"].nunique() > cfg.fitness_sample_symbols:
            sampled_symbols = set(rng.sample(list(labels["symbol"].dropna().astype(str).unique()), cfg.fitness_sample_symbols))
            labels = labels[labels["symbol"].astype(str).isin(sampled_symbols)]
            panel = panel[panel["symbol"].astype(str).isin(sampled_symbols)]
        merged = panel.merge(labels, on=["symbol", "trade_date"], how="inner")
    else:
        merged = panel
    if cfg.label_column not in merged.columns:
        raise KeyError(
            f"synthesize_factors needs label column '{cfg.label_column}' in panel or labels"
        )
    missing_market = [c for c in _MARKET_COLUMNS if c not in merged.columns]
    if missing_market:
        raise KeyError(
            "merged GA panel lost market columns "
            f"{missing_market} (suffix collision or wrong --market-panel input); "
            f"available: {sorted(merged.columns)[:40]}"
        )
    valid_panel, label_series, date_series = _subsample_panel(merged, cfg.label_column, cfg, rng)
    train_panel, train_labels, train_dates, oos_panel, oos_labels, oos_dates = _chronological_split(
        valid_panel,
        label_series.reset_index(drop=True),
        date_series.reset_index(drop=True),
        cfg,
    )
    logger.info("[ga] fitness sample: %d rows, %d dates, %d symbols",
                len(valid_panel), valid_panel["trade_date"].nunique(), valid_panel["symbol"].nunique())
    logger.info("[ga] split: train=%d rows, validation=%d rows", len(train_panel), len(oos_panel))

    eval_cache: dict[str, tuple[float, float, float]] = {}

    def _cached_fitness(tree: E.Expr) -> tuple[float, float, float]:
        key = repr(tree)
        if key not in eval_cache:
            eval_cache[key] = _evaluate_fitness(tree, train_panel, train_labels, train_dates, cfg)
        return eval_cache[key]

    population: list[E.Expr] = _seed_population(rng, cfg)
    fitness: list[float] = [-1.0] * cfg.population
    raw_ics: list[float] = [0.0] * cfg.population
    finite_ratios: list[float] = [0.0] * cfg.population
    for i, tree in enumerate(population):
        fitness[i], raw_ics[i], finite_ratios[i] = _cached_fitness(tree)

    if max(fitness) <= -1.0:
        # Every tree was rejected; surface the underlying error instead of
        # silently burning generations on a dead population.
        probe = E.Rank(E.Returns(E.Close, 5))
        probe.evaluate(train_panel)  # raises with the real cause if data is broken
        raise RuntimeError(
            "symbolic GA: generation 0 produced no viable tree although the probe "
            "expression evaluates; check label alignment and min_finite_ratio "
            f"(finite ratios seen: {sorted(set(round(f, 2) for f in finite_ratios))[:10]})"
        )

    history_rows: list[dict[str, float]] = []
    for generation in range(cfg.generations):
        best = max(fitness)
        mean = float(np.mean(fitness))
        mean_complexity = float(np.mean([_node_count(t) for t in population]))
        history_rows.append({
            "generation": generation,
            "best_fitness": best,
            "mean_fitness": mean,
            "mean_complexity": mean_complexity,
        })
        logger.info("[ga] gen %d  best=%.4f  mean=%.4f  mean_complexity=%.1f",
                    generation, best, mean, mean_complexity)

        # Elitism + tournament selection + crossover/mutation.
        order = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
        elites = [population[i] for i in order[: cfg.elitism]]
        new_pop = list(elites)
        while len(new_pop) < cfg.population:
            p1 = _tournament_pick(rng, population, fitness, cfg.tournament_size)
            p2 = _tournament_pick(rng, population, fitness, cfg.tournament_size)
            if rng.random() < cfg.crossover_prob:
                c1, c2 = _crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2
            if rng.random() < cfg.mutation_prob:
                c1 = _mutate(c1, rng, cfg.max_depth)
            if rng.random() < cfg.mutation_prob:
                c2 = _mutate(c2, rng, cfg.max_depth)
            new_pop.extend([c1, c2])
        new_pop = new_pop[: cfg.population]
        inject_n = int(round(cfg.population * cfg.random_injection_rate))
        for j in range(inject_n):
            new_pop[-(j + 1)] = _random_tree(rng, cfg.max_depth, force_internal=True)
        population = new_pop
        fitness = [-1.0] * cfg.population
        raw_ics = [0.0] * cfg.population
        finite_ratios = [0.0] * cfg.population
        for i, tree in enumerate(population):
            fitness[i], raw_ics[i], finite_ratios[i] = _cached_fitness(tree)

    # Final leaderboard.
    order = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
    rows: list[dict[str, object]] = []
    chosen: list[E.Expr] = []
    chosen_values: list[pd.Series] = []
    seen_exprs: set[str] = set()
    reference_values: dict[str, pd.Series] = {
        ref: pd.to_numeric(oos_panel[ref], errors="coerce")
        for ref in cfg.reference_columns
        if ref in oos_panel.columns
    }
    for idx in order:
        tree = population[idx]
        if fitness[idx] <= 0:
            break
        expr_key = repr(tree)
        if expr_key in seen_exprs:
            continue
        seen_exprs.add(expr_key)
        train_ic = raw_ics[idx]
        oriented_tree = tree if train_ic >= 0 else E.Mul(E.Constant(-1.0), tree)
        validation_ic, validation_finite = _evaluate_ic(oriented_tree, oos_panel, oos_labels, oos_dates)
        if validation_finite < cfg.min_finite_ratio or validation_ic < cfg.min_validation_rank_ic:
            continue
        try:
            values = pd.to_numeric(oriented_tree.evaluate(oos_panel), errors="coerce")
        except Exception:
            continue
        # Decorrelate against already-chosen survivors.
        if any(
            abs(values.corr(prev, method="spearman")) > cfg.max_correlation
            for prev in chosen_values
            if prev is not None
        ):
            continue
        # Decorrelate against the existing factor library (novelty gate).
        ref_corr = 0.0
        for ref_series in reference_values.values():
            corr = abs(values.corr(ref_series, method="spearman"))
            if np.isfinite(corr):
                ref_corr = max(ref_corr, float(corr))
        if reference_values and ref_corr > cfg.max_reference_correlation:
            continue
        chosen.append(oriented_tree)
        chosen_values.append(values)
        rows.append({
            "name": f"synth_{len(chosen):03d}",
            "expression": repr(oriented_tree),
            "train_rank_ic": float(abs(train_ic)),
            "validation_rank_ic": float(validation_ic),
            "fitness": float(fitness[idx]),
            "complexity": int(_node_count(oriented_tree)),
            "finite_ratio": float(finite_ratios[idx]),
            "max_reference_corr": ref_corr,
        })
        if len(chosen) >= cfg.top_k:
            break

    definitions = [
        E.FactorDefinition(name=row["name"], expr=tree, description="GA-synthesized factor")
        for row, tree in zip(rows, chosen)
    ]
    leaderboard = pd.DataFrame(rows, columns=["name", "expression", "train_rank_ic", "validation_rank_ic", "fitness", "complexity", "finite_ratio", "max_reference_corr"])
    history = pd.DataFrame(history_rows)
    return SynthesisResult(definitions=definitions, leaderboard=leaderboard, history=history)


def _tournament_pick(
    rng: random.Random,
    population: Sequence[E.Expr],
    fitness: Sequence[float],
    tournament_size: int,
) -> E.Expr:
    indices = rng.sample(range(len(population)), min(tournament_size, len(population)))
    best_idx = max(indices, key=lambda i: fitness[i])
    return population[best_idx]


# --------------------------------------------------------------------------- #
# Persistence helpers                                                         #
# --------------------------------------------------------------------------- #


def save_definitions(definitions: Iterable[E.FactorDefinition], path: str | Path) -> Path:
    """Dump discovered factors as JSON (name + repr-of-expression + description)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"name": d.name, "expression": repr(d.expr), "description": d.description}
        for d in definitions
    ]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def save_result(result: SynthesisResult, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    defs_path = save_definitions(result.definitions, out / "synthesized_definitions.json")
    lb_path = out / "synthesized_leaderboard.parquet"
    try:
        result.leaderboard.to_parquet(lb_path, index=False)
    except Exception:
        lb_path = lb_path.with_suffix(".csv")
        result.leaderboard.to_csv(lb_path, index=False)
    hist_path = out / "synthesized_history.parquet"
    try:
        result.history.to_parquet(hist_path, index=False)
    except Exception:
        hist_path = hist_path.with_suffix(".csv")
        result.history.to_csv(hist_path, index=False)
    return {
        "definitions": str(defs_path),
        "leaderboard": str(lb_path),
        "history": str(hist_path),
    }


# --------------------------------------------------------------------------- #
# Load discovered factors back into the live feature pipeline                 #
# --------------------------------------------------------------------------- #


_PARSE_NAMESPACE = {
    "Column": E.Column,
    "OptionalColumn": E.OptionalColumn,
    "Constant": E.Constant,
    "Add": E.Add,
    "Sub": E.Sub,
    "Mul": E.Mul,
    "Div": E.Div,
    "Abs": E.Abs,
    "Sign": E.Sign,
    "Log": E.Log,
    "Delay": E.Delay,
    "Delta": E.Delta,
    "Returns": E.Returns,
    "Rank": E.Rank,
    "TsRank": E.TsRank,
    "TsCorr": E.TsCorr,
    "TsCov": E.TsCov,
    "DecayLinear": E.DecayLinear,
    "CsZscore": E.CsZscore,
    "TsMean": E.TsMean,
    "TsStd": E.TsStd,
    "TsSum": E.TsSum,
    "TsMax": E.TsMax,
    "TsMin": E.TsMin,
    "_RollingReduction": E._RollingReduction,
}


def parse_expression(expr_repr: str) -> E.Expr:
    """Reconstruct an :class:`Expr` from its ``repr()``.

    Safe because the eval namespace is restricted to the DSL classes above —
    no builtins, no arbitrary import paths.
    """
    return eval(expr_repr, {"__builtins__": {}}, _PARSE_NAMESPACE)  # noqa: S307


def load_definitions(path: str | Path) -> list[E.FactorDefinition]:
    """Load top-K factors saved by :func:`save_definitions`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        E.FactorDefinition(name=item["name"], expr=parse_expression(item["expression"]),
                           description=item.get("description", ""))
        for item in payload
    ]


def compute_synthesized_factors(frame: pd.DataFrame, definitions_path: str | Path) -> pd.DataFrame:
    """Evaluate GA-synthesised factors against ``frame`` and return long format.

    Returns an empty DataFrame if the definitions file is missing or empty so
    callers can chain this safely into feature pipelines.
    """
    path = Path(definitions_path)
    if not path.exists():
        return pd.DataFrame(columns=["trade_date", "symbol", "factor_name", "factor_value"])
    definitions = load_definitions(path)
    if not definitions:
        return pd.DataFrame(columns=["trade_date", "symbol", "factor_name", "factor_value"])
    rows: list[pd.DataFrame] = []
    base = frame[["symbol", "trade_date"]].copy()
    for definition in definitions:
        try:
            values = pd.to_numeric(definition.expr.evaluate(frame), errors="coerce")
        except Exception:
            continue
        piece = base.copy()
        piece["factor_name"] = definition.name
        piece["factor_value"] = values.replace([np.inf, -np.inf], np.nan).to_numpy()
        rows.append(piece)
    if not rows:
        return pd.DataFrame(columns=["trade_date", "symbol", "factor_name", "factor_value"])
    return pd.concat(rows, ignore_index=True)


__all__ = [
    "SymbolicGAConfig",
    "SynthesisResult",
    "synthesize_factors",
    "save_definitions",
    "save_result",
    "load_definitions",
    "parse_expression",
    "compute_synthesized_factors",
]
