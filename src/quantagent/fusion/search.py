"""Fusion search orchestrator.

Protocol, in order, because the order is the guarantee:

1. Enumerate candidate *schemes* (:mod:`quantagent.fusion.schemes`). The size of
   this list is ``n_trials`` and it is fixed before any evaluation runs.
2. Build purged, embargoed, expanding walk-forward folds over the panel dates.
3. For every fold and every scheme: fit on the training segment only, evaluate on
   the out-of-sample segment only.
4. Aggregate each scheme's out-of-sample folds into candidate metrics, and
   compute the search-wide probability of backtest overfitting from the fold
   performance matrix.
5. Score robustness with ``n_trials`` from step 1, take the Pareto frontier over
   the four objectives, and rank the frontier by operator preference.

Step 5 cannot inflate step 1: the trial count is captured before results exist,
so a disappointing search cannot be rescued by dropping its controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from quantagent.fusion.evaluation import SegmentEvaluation, evaluate_blend
from quantagent.fusion.frontier import (
    ObjectivePreference,
    pareto_front,
    rank_by_preference,
)
from quantagent.fusion.robustness import (
    RobustnessInputs,
    RobustnessWeights,
    robustness_score,
    search_pbo,
)
from quantagent.fusion.schemes import SchemeSpec, build_scheme_specs, derive_weights
from quantagent.optimization.ga_weight_optimizer import (
    WalkForwardConfig,
    purged_walk_forward_splits,
)

# (stage label, completed evaluations, total evaluations)
ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class FusionSearchConfig:
    """Everything the operator controls. ``n_trials`` is deliberately absent."""

    factor_names: tuple[str, ...]
    horizon_days: int = 5
    top_k: int = 30
    n_folds: int = 4
    embargo_days: int = 5
    min_train_days: int = 120
    min_test_days: int = 40
    transaction_cost_bps: float = 8.0
    include_genetic: bool = True
    random_controls: int = 8
    single_factor_baselines: int = 6
    seed: int = 17
    benchmark_symbol: str = ""
    preference: ObjectivePreference = field(default_factory=ObjectivePreference)
    robustness_weights: RobustnessWeights = field(default_factory=RobustnessWeights)

    def __post_init__(self) -> None:
        if not self.factor_names:
            raise ValueError("factor_names must not be empty")
        if len(set(self.factor_names)) != len(self.factor_names):
            raise ValueError("factor_names must be unique")
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if self.n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be >= 0")

    @property
    def periods_per_year(self) -> int:
        return max(1, int(round(252 / self.horizon_days)))


@dataclass(frozen=True)
class FusionFoldResult:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    weights: dict[str, float]
    metrics: dict[str, float | int]

    def as_dict(self) -> dict[str, object]:
        return {
            "foldIndex": self.fold_index,
            "trainStart": self.train_start,
            "trainEnd": self.train_end,
            "testStart": self.test_start,
            "testEnd": self.test_end,
            "weights": self.weights,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class FusionCandidate:
    candidate_id: str
    label: str
    scheme: str
    is_control: bool
    weights: dict[str, float]
    metrics: dict[str, float | int]
    robustness_breakdown: dict[str, float | None]
    folds: list[FusionFoldResult]
    nav: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.candidate_id,
            "label": self.label,
            "scheme": self.scheme,
            "isControl": self.is_control,
            "weights": self.weights,
            "metrics": self.metrics,
            "robustnessBreakdown": self.robustness_breakdown,
            "folds": [fold.as_dict() for fold in self.folds],
        }


@dataclass(frozen=True)
class FusionSearchResult:
    config: FusionSearchConfig
    candidates: list[FusionCandidate]
    frontier: list[str]
    ranking: list[dict[str, object]]
    n_trials: int
    pbo: float
    benchmark_mode: str
    fold_windows: list[dict[str, str]]
    generated_at: str

    @property
    def preferred(self) -> FusionCandidate | None:
        if not self.ranking:
            return None
        top = str(self.ranking[0]["id"])
        return next((item for item in self.candidates if item.candidate_id == top), None)

    def summary(self) -> dict[str, object]:
        return {
            "generatedAt": self.generated_at,
            # Declared, not inferred. Every scheme in this package produces a
            # weight VECTOR, and the score is `sum_j w_j * rank(x_j)` — additive
            # across factors by construction, whatever the scheme that produced
            # the weights. The genetic scheme is a nonlinear *optimiser* over an
            # additive model, which is a different thing again; see
            # quantagent.models.interactions.ModelClass for the distinctions and
            # quantagent.research.model_comparison for the arms that do form
            # interaction terms.
            "modelClass": "rank_weighted_additive",
            "representsInteraction": False,
            "nTrials": self.n_trials,
            "pbo": None if not np.isfinite(self.pbo) else round(float(self.pbo), 6),
            "benchmarkMode": self.benchmark_mode,
            "horizonDays": self.config.horizon_days,
            "topK": self.config.top_k,
            "transactionCostBps": self.config.transaction_cost_bps,
            "factorNames": list(self.config.factor_names),
            "foldWindows": self.fold_windows,
            "frontier": self.frontier,
            "preferenceWeights": self.config.preference.normalised(),
            "candidateCount": len(self.candidates),
            "evaluatedCandidateCount": sum(
                1 for item in self.candidates if int(item.metrics.get("observations", 0)) > 0
            ),
        }


def _normalise_panel(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if "trade_date" not in frame.columns or "symbol" not in frame.columns:
        raise KeyError(f"{name} must contain trade_date and symbol columns")
    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["symbol"] = work["symbol"].astype(str)
    duplicated = work.duplicated(["trade_date", "symbol"], keep=False)
    if bool(duplicated.any()):
        sample = work.loc[duplicated, ["trade_date", "symbol"]].head(5).to_dict("records")
        raise ValueError(f"{name} has duplicate trade_date/symbol keys: {sample}")
    return work


def _segment(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (frame["trade_date"] >= start) & (frame["trade_date"] <= end)
    return frame.loc[mask]


def _aggregate_metrics(
    evaluations: list[SegmentEvaluation],
) -> dict[str, float | int]:
    """Pool out-of-sample folds into one candidate-level metric set.

    Scalar rates are averaged across folds weighted by observation count so a
    two-observation fold cannot outvote a hundred-observation fold.
    """
    usable = [item for item in evaluations if not item.is_empty]
    if not usable:
        return {
            "observations": 0,
            "annualReturn": 0.0,
            "excessReturn": 0.0,
            "benchmarkAnnualReturn": 0.0,
            "maxDrawdown": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "averageTurnover": 0.0,
            "costDrag": 0.0,
            "winRate": 0.0,
            "robustness": 0.0,
        }
    weights = np.array([float(item.observations) for item in usable])
    weights = weights / weights.sum()

    def pooled(getter: Callable[[SegmentEvaluation], float]) -> float:
        values = np.array([float(getter(item)) for item in usable])
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.dot(values, weights))

    return {
        "observations": int(sum(item.observations for item in usable)),
        "annualReturn": round(pooled(lambda item: item.annual_return), 6),
        "excessReturn": round(pooled(lambda item: item.excess_return), 6),
        "benchmarkAnnualReturn": round(
            pooled(lambda item: item.benchmark_annual_return), 6
        ),
        # Drawdown pools as the worst fold, not the average: a strategy that
        # lost 30% in one fold did lose 30%.
        "maxDrawdown": round(max(abs(item.max_drawdown) for item in usable), 6),
        "sharpe": round(pooled(lambda item: item.sharpe), 6),
        "calmar": round(pooled(lambda item: item.calmar), 6),
        "averageTurnover": round(pooled(lambda item: item.average_turnover), 6),
        "costDrag": round(pooled(lambda item: item.cost_drag), 6),
        "winRate": round(pooled(lambda item: item.win_rate), 6),
    }


def _median_weights(
    fold_weights: list[np.ndarray],
    factor_names: Sequence[str],
) -> dict[str, float]:
    """Coordinate-wise median across folds, renormalised to unit L1 norm."""
    if not fold_weights:
        return {name: 0.0 for name in factor_names}
    stacked = np.vstack(fold_weights)
    median = np.median(stacked, axis=0)
    total = float(np.abs(median).sum())
    if total <= 1e-12:
        median = np.full(len(factor_names), 1.0 / max(1, len(factor_names)))
    else:
        median = median / total
    return {name: round(float(value), 6) for name, value in zip(factor_names, median)}


def run_fusion_search(
    *,
    factor_panel: pd.DataFrame,
    forward_panel: pd.DataFrame,
    config: FusionSearchConfig,
    benchmark_returns: pd.Series | None = None,
    progress: ProgressCallback | None = None,
) -> FusionSearchResult:
    """Execute the search protocol and return every candidate that was tried."""
    factors = _normalise_panel(factor_panel, "factor_panel")
    forwards = _normalise_panel(forward_panel, "forward_panel")
    if "forward_return" not in forwards.columns:
        raise KeyError("forward_panel must contain a forward_return column")
    missing = [name for name in config.factor_names if name not in factors.columns]
    if missing:
        raise KeyError(f"factor columns missing from factor_panel: {sorted(missing)}")

    factor_names = list(config.factor_names)
    specs = build_scheme_specs(
        factor_names,
        top_k=config.top_k,
        include_genetic=config.include_genetic,
        random_controls=config.random_controls,
        single_factor_baselines=config.single_factor_baselines,
        seed=config.seed,
    )
    n_trials = len(specs)

    wf_config = WalkForwardConfig(
        n_folds=config.n_folds,
        embargo_days=config.embargo_days,
        label_horizon_days=config.horizon_days,
        min_train_days=config.min_train_days,
        min_test_days=config.min_test_days,
    )
    dates = pd.DatetimeIndex(sorted(factors["trade_date"].unique()))
    splits = purged_walk_forward_splits(dates, wf_config)
    if not splits:
        raise ValueError(
            "panel is too short for the requested walk-forward configuration: "
            f"{len(dates)} dates, need at least "
            f"{wf_config.min_train_days + wf_config.effective_gap_days + wf_config.min_test_days}"
        )

    benchmark_mode = (
        f"index:{config.benchmark_symbol}"
        if benchmark_returns is not None and not benchmark_returns.empty
        else "universe_equal_weight"
    )

    fold_windows = [
        {
            "foldIndex": str(index),
            "trainStart": str(train_start.date()),
            "trainEnd": str(train_end.date()),
            "testStart": str(test_start.date()),
            "testEnd": str(test_end.date()),
        }
        for index, (train_start, train_end, test_start, test_end) in enumerate(splits)
    ]

    per_spec_evaluations: dict[str, list[SegmentEvaluation]] = {
        spec.candidate_id: [] for spec in specs
    }
    per_spec_folds: dict[str, list[FusionFoldResult]] = {
        spec.candidate_id: [] for spec in specs
    }
    per_spec_weights: dict[str, list[np.ndarray]] = {
        spec.candidate_id: [] for spec in specs
    }

    total_steps = max(1, len(splits) * len(specs))
    completed = 0
    for fold_index, (train_start, train_end, test_start, test_end) in enumerate(splits):
        train_factors = _segment(factors, train_start, train_end)
        train_forwards = _segment(forwards, train_start, train_end)
        test_factors = _segment(factors, test_start, test_end)
        test_forwards = _segment(forwards, test_start, test_end)

        for spec in specs:
            weights = derive_weights(
                spec,
                factor_panel=train_factors,
                forward_panel=train_forwards,
                factor_names=factor_names,
                horizon_days=config.horizon_days,
                transaction_cost_bps=config.transaction_cost_bps,
            )
            evaluation = evaluate_blend(
                factor_panel=test_factors,
                forward_panel=test_forwards,
                factor_names=factor_names,
                weights=weights,
                top_k=spec.top_k,
                horizon_days=config.horizon_days,
                transaction_cost_bps=config.transaction_cost_bps,
                benchmark_returns=benchmark_returns,
            )
            per_spec_evaluations[spec.candidate_id].append(evaluation)
            per_spec_weights[spec.candidate_id].append(np.asarray(weights, dtype=float))
            per_spec_folds[spec.candidate_id].append(
                FusionFoldResult(
                    fold_index=fold_index,
                    train_start=str(train_start.date()),
                    train_end=str(train_end.date()),
                    test_start=str(test_start.date()),
                    test_end=str(test_end.date()),
                    weights={
                        name: round(float(value), 6)
                        for name, value in zip(factor_names, weights)
                    },
                    metrics=evaluation.as_dict(),
                )
            )
            completed += 1
            if progress is not None:
                progress(
                    f"fold {fold_index + 1}/{len(splits)} · {spec.candidate_id}",
                    completed,
                    total_steps,
                )

    # CSCV needs many time slices, not one row per fold: with four folds there
    # are not enough partitions to form a single symmetric IS/OOS split. Build
    # the matrix from the pooled out-of-sample period returns instead, which is
    # the granularity the method was designed for.
    pooled_by_candidate = {
        spec.candidate_id: pd.concat(
            [
                item.period_returns
                for item in per_spec_evaluations[spec.candidate_id]
                if not item.is_empty
            ]
            or [pd.Series(dtype=float)]
        ).sort_index()
        for spec in specs
    }
    performance_matrix = pd.DataFrame(pooled_by_candidate)
    performance_matrix = performance_matrix[
        performance_matrix.notna().any(axis=1)
    ].fillna(0.0)
    pbo = search_pbo(performance_matrix)

    candidate_sharpes = np.array(
        [
            float(np.nan_to_num(np.mean([
                item.sharpe for item in per_spec_evaluations[spec.candidate_id]
                if not item.is_empty
            ] or [0.0]), nan=0.0))
            for spec in specs
        ],
        dtype=float,
    )

    candidates: list[FusionCandidate] = []
    for spec in specs:
        evaluations = per_spec_evaluations[spec.candidate_id]
        metrics = _aggregate_metrics(evaluations)
        usable = [item for item in evaluations if not item.is_empty]
        pooled_returns = (
            pd.concat([item.period_returns for item in usable]).sort_index()
            if usable
            else pd.Series(dtype=float)
        )
        score, breakdown = robustness_score(
            RobustnessInputs(
                fold_excess_returns=[item.excess_return for item in usable],
                fold_sharpes=[item.sharpe for item in usable],
                period_returns=pooled_returns,
                periods_per_year=config.periods_per_year,
                candidate_sharpes=candidate_sharpes,
                pbo=pbo,
            ),
            config.robustness_weights,
        )
        metrics["robustness"] = round(score, 6)
        nav = (
            pd.concat([item.nav for item in usable]).sort_index()
            if usable
            else pd.Series(dtype=float)
        )
        candidates.append(
            FusionCandidate(
                candidate_id=spec.candidate_id,
                label=spec.label,
                scheme=spec.scheme.value,
                is_control=spec.is_control,
                weights=_median_weights(per_spec_weights[spec.candidate_id], factor_names),
                metrics=metrics,
                robustness_breakdown=breakdown,
                folds=per_spec_folds[spec.candidate_id],
                nav=nav,
            )
        )

    serialisable = [item.as_dict() for item in candidates]
    frontier = pareto_front(serialisable)
    ranking = rank_by_preference(
        [item for item in serialisable if str(item["id"]) in frontier],
        config.preference,
    )

    return FusionSearchResult(
        config=config,
        candidates=candidates,
        frontier=frontier,
        ranking=ranking,
        n_trials=n_trials,
        pbo=pbo,
        benchmark_mode=benchmark_mode,
        fold_windows=fold_windows,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def save_fusion_artifacts(
    result: FusionSearchResult,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Persist the search so every number in the UI has a file behind it."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    candidates_payload = [item.as_dict() for item in result.candidates]
    paths = {
        "summary": target / "fusion_summary.json",
        "candidates": target / "fusion_candidates.json",
        "ranking": target / "fusion_ranking.json",
        "manifest": target / "manifest.json",
    }
    paths["candidates"].write_text(
        json.dumps(candidates_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths["summary"].write_text(
        json.dumps(result.summary(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths["ranking"].write_text(
        json.dumps(result.ranking, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    nav_frames = {
        item.candidate_id: item.nav
        for item in result.candidates
        if not item.nav.empty
    }
    if nav_frames:
        nav_path = target / "fusion_nav.csv"
        pd.DataFrame(nav_frames).sort_index().to_csv(nav_path, index_label="trade_date")
        paths["nav"] = nav_path

    content_hash = hashlib.sha256(
        json.dumps(candidates_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    paths["manifest"].write_text(
        json.dumps(
            {
                "artifact": "factor_fusion_search",
                "schemaVersion": 1,
                "generatedAt": result.generated_at,
                "contentHash": content_hash,
                "nTrials": result.n_trials,
                "benchmarkMode": result.benchmark_mode,
                "factorNames": list(result.config.factor_names),
                "horizonDays": result.config.horizon_days,
                "topK": result.config.top_k,
                "folds": result.fold_windows,
                "files": {key: value.name for key, value in paths.items() if key != "manifest"},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "FusionCandidate",
    "FusionFoldResult",
    "FusionSearchConfig",
    "FusionSearchResult",
    "ProgressCallback",
    "run_fusion_search",
    "save_fusion_artifacts",
]
