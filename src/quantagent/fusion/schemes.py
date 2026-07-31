"""Blend-weight derivation schemes.

A *scheme* is a procedure that turns a training segment into a weight vector.
It is not a fixed weight vector: the weights a scheme produces are refitted on
every fold's training segment and then evaluated on that fold's out-of-sample
segment. This is what keeps the search honest — a scheme that only works because
it saw the test data cannot hide behind a hand-picked weight vector.

Schemes fall into two families:

``fitted``
    ``ic_weighted``, ``ic_ir_weighted``, ``inverse_volatility``, ``genetic`` —
    read the training segment.
``control``
    ``equal``, ``random_simplex``, ``single_factor`` — ignore the training
    segment entirely. They exist so the fitted schemes have something to beat,
    and they still consume trials, which correctly deflates the Sharpe ratio.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from quantagent.fusion.evaluation import cross_sectional_scores


class BlendScheme(str, Enum):
    EQUAL = "equal"
    IC_WEIGHTED = "ic_weighted"
    IC_IR_WEIGHTED = "ic_ir_weighted"
    INVERSE_VOLATILITY = "inverse_volatility"
    RANDOM_SIMPLEX = "random_simplex"
    SINGLE_FACTOR = "single_factor"
    GENETIC = "genetic"

    @property
    def is_fitted(self) -> bool:
        return self in {
            BlendScheme.IC_WEIGHTED,
            BlendScheme.IC_IR_WEIGHTED,
            BlendScheme.INVERSE_VOLATILITY,
            BlendScheme.GENETIC,
        }


@dataclass(frozen=True)
class SchemeSpec:
    """One candidate procedure in the search."""

    candidate_id: str
    scheme: BlendScheme
    label: str
    seed: int = 0
    factor_index: int = -1
    top_k: int = 30

    @property
    def is_control(self) -> bool:
        return not self.scheme.is_fitted


def _normalise(vector: np.ndarray) -> np.ndarray:
    """Scale to unit L1 norm, falling back to equal weights when degenerate."""
    values = np.nan_to_num(np.asarray(vector, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.abs(values).sum())
    if total <= 1e-12:
        return np.full(values.shape, 1.0 / max(1, values.size))
    return values / total


def factor_ic_series(
    factor_panel: pd.DataFrame,
    forward_panel: pd.DataFrame,
    factor_names: list[str],
) -> pd.DataFrame:
    """Per-date Spearman rank IC of each factor against the forward return.

    Index is ``trade_date``; columns are ``factor_names``. Dates where a factor
    has fewer than three usable observations produce ``NaN`` for that factor.
    """
    merged = factor_panel[["trade_date", "symbol", *factor_names]].merge(
        forward_panel[["trade_date", "symbol", "forward_return"]],
        on=["trade_date", "symbol"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=factor_names, dtype=float)
    merged["forward_return"] = pd.to_numeric(merged["forward_return"], errors="coerce")
    merged = merged.dropna(subset=["forward_return"])
    if merged.empty:
        return pd.DataFrame(columns=factor_names, dtype=float)

    records: dict[pd.Timestamp, dict[str, float]] = {}
    for trade_date, group in merged.groupby("trade_date", sort=True):
        target = group["forward_return"].rank(pct=True)
        row: dict[str, float] = {}
        for name in factor_names:
            values = pd.to_numeric(group[name], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            usable = values.notna()
            if int(usable.sum()) < 3:
                row[name] = float("nan")
                continue
            correlation = values[usable].rank(pct=True).corr(target[usable])
            row[name] = float(correlation) if pd.notna(correlation) else float("nan")
        records[trade_date] = row
    return pd.DataFrame.from_dict(records, orient="index").reindex(columns=factor_names)


def _genetic_weights(
    factor_panel: pd.DataFrame,
    forward_panel: pd.DataFrame,
    factor_names: list[str],
    spec: SchemeSpec,
    horizon_days: int,
    transaction_cost_bps: float,
) -> np.ndarray:
    """Delegate to the existing multi-objective GA on the training segment."""
    from quantagent.optimization.ga_weight_optimizer import (
        GAConfig,
        WalkForwardConfig,
        optimize_factor_weights_ga,
    )

    dates = pd.DatetimeIndex(sorted(factor_panel["trade_date"].unique()))
    if len(dates) < 8:
        return _normalise(np.ones(len(factor_names)))
    result = optimize_factor_weights_ga(
        factor_panel=factor_panel,
        forward_returns=forward_panel,
        factor_names=factor_names,
        ga_config=GAConfig(
            population_size=16,
            generations=6,
            top_k=spec.top_k,
            random_seed=spec.seed,
            label_horizon_days=horizon_days,
            transaction_cost_bps=transaction_cost_bps,
        ),
        wf_config=WalkForwardConfig(
            n_folds=2,
            embargo_days=horizon_days,
            label_horizon_days=horizon_days,
            min_train_days=max(2, len(dates) // 4),
            min_test_days=2,
        ),
    )
    return _normalise(
        np.array([float(result.best_weights.get(name, 0.0)) for name in factor_names])
    )


def derive_weights(
    spec: SchemeSpec,
    *,
    factor_panel: pd.DataFrame,
    forward_panel: pd.DataFrame,
    factor_names: list[str],
    horizon_days: int = 1,
    transaction_cost_bps: float = 8.0,
) -> np.ndarray:
    """Produce the weight vector for ``spec`` from a training segment.

    Control schemes ignore the panels entirely. Fitted schemes read them, and
    callers are responsible for passing **only** the training segment.
    """
    size = len(factor_names)
    if size == 0:
        raise ValueError("factor_names must not be empty")

    if spec.scheme is BlendScheme.EQUAL:
        return _normalise(np.ones(size))

    if spec.scheme is BlendScheme.SINGLE_FACTOR:
        if not 0 <= spec.factor_index < size:
            raise ValueError(f"factor_index {spec.factor_index} out of range")
        vector = np.zeros(size)
        vector[spec.factor_index] = 1.0
        return vector

    if spec.scheme is BlendScheme.RANDOM_SIMPLEX:
        rng = np.random.default_rng(spec.seed)
        return _normalise(rng.dirichlet(np.ones(size)))

    if spec.scheme is BlendScheme.GENETIC:
        return _genetic_weights(
            factor_panel,
            forward_panel,
            factor_names,
            spec,
            horizon_days,
            transaction_cost_bps,
        )

    ic = factor_ic_series(factor_panel, forward_panel, factor_names)
    if ic.empty:
        return _normalise(np.ones(size))

    mean_ic = ic.mean(axis=0, skipna=True)
    if spec.scheme is BlendScheme.IC_WEIGHTED:
        return _normalise(mean_ic.to_numpy(dtype=float))

    std_ic = ic.std(axis=0, ddof=1, skipna=True).replace(0.0, np.nan)
    if spec.scheme is BlendScheme.IC_IR_WEIGHTED:
        return _normalise((mean_ic / std_ic).to_numpy(dtype=float))

    # INVERSE_VOLATILITY: same sign as the mean IC, magnitude 1/sigma(IC).
    direction = np.sign(mean_ic.to_numpy(dtype=float))
    direction[direction == 0.0] = 1.0
    inverse = 1.0 / std_ic.to_numpy(dtype=float)
    return _normalise(direction * inverse)


def build_scheme_specs(
    factor_names: list[str],
    *,
    top_k: int,
    include_genetic: bool = True,
    random_controls: int = 8,
    single_factor_baselines: int = 6,
    seed: int = 17,
) -> list[SchemeSpec]:
    """Enumerate the candidate procedures for one search.

    The returned list defines ``n_trials``. Callers must not filter it after the
    fact to make the deflated Sharpe ratio look better — the whole point of
    counting controls is that they are counted.
    """
    if not factor_names:
        raise ValueError("factor_names must not be empty")
    if random_controls < 0 or single_factor_baselines < 0:
        raise ValueError("control counts must be >= 0")

    specs: list[SchemeSpec] = [
        SchemeSpec("equal", BlendScheme.EQUAL, "等权基线", top_k=top_k),
        SchemeSpec("ic_weighted", BlendScheme.IC_WEIGHTED, "IC 加权", top_k=top_k),
        SchemeSpec("ic_ir_weighted", BlendScheme.IC_IR_WEIGHTED, "ICIR 加权", top_k=top_k),
        SchemeSpec(
            "inverse_volatility",
            BlendScheme.INVERSE_VOLATILITY,
            "IC 波动率倒数",
            top_k=top_k,
        ),
    ]
    if include_genetic:
        specs.append(
            SchemeSpec("genetic", BlendScheme.GENETIC, "多目标遗传搜索", seed=seed, top_k=top_k)
        )
    for index in range(random_controls):
        specs.append(
            SchemeSpec(
                f"random_{index:02d}",
                BlendScheme.RANDOM_SIMPLEX,
                f"随机对照 #{index + 1}",
                seed=seed + 1000 + index,
                top_k=top_k,
            )
        )
    for index in range(min(single_factor_baselines, len(factor_names))):
        specs.append(
            SchemeSpec(
                f"single_{factor_names[index]}",
                BlendScheme.SINGLE_FACTOR,
                f"单因子 {factor_names[index]}",
                factor_index=index,
                top_k=top_k,
            )
        )
    return specs


__all__ = [
    "BlendScheme",
    "SchemeSpec",
    "build_scheme_specs",
    "derive_weights",
    "factor_ic_series",
]
