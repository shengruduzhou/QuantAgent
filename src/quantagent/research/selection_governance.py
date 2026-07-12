"""Nested purged selection, PBO/DSR gates and cumulative trial accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quantagent.quant_math.performance import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
    spa_test,
)
from quantagent.quant_math.purged_cv import PurgedKFoldConfig, purged_kfold_split


@dataclass(frozen=True)
class NestedSelectionConfig:
    outer_splits: int = 5
    inner_splits: int = 4
    embargo_pct: float = 0.01
    periods_per_year: int = 252
    pbo_partitions: int = 8
    max_pbo: float = 0.25
    min_dsr_probability: float = 0.95
    max_spa_pvalue: float = 0.05
    max_losing_outer_fold_rate: float = 0.40


@dataclass(frozen=True)
class TrialRecord:
    family: str
    candidate_id: str
    parameters: Mapping[str, Any]
    dataset_hash: str
    train_window: tuple[str, str]
    search_window: tuple[str, str]
    metric: str
    git_hash: str = "unknown"
    status: str = "registered"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "family": self.family,
            "candidate_id": self.candidate_id,
            "parameters": dict(self.parameters),
            "dataset_hash": self.dataset_hash,
            "train_window": list(self.train_window),
            "search_window": list(self.search_window),
            "metric": self.metric,
            "git_hash": self.git_hash,
            "status": self.status,
            "created_at": self.created_at,
        }
        payload["trial_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return payload


class TrialRegistry:
    """Append-only JSONL registry used to count all attempted configurations."""

    def __init__(self, path: str | Path = "runtime/state/experiment_trials.jsonl") -> None:
        self.path = Path(path)

    def append(self, record: TrialRecord) -> dict[str, Any]:
        payload = record.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        # O_APPEND makes each small record a single append operation.  This is
        # intentionally simpler and more robust than rewriting a shared JSON file.
        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        return payload

    def read(self, family: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if family is None or row.get("family") == family:
                rows.append(row)
        return rows

    def count(self, family: str | None = None) -> int:
        return len(self.read(family=family))


@dataclass
class OuterFoldSelection:
    fold_index: int
    selected_candidate: str
    inner_scores: dict[str, float]
    outer_sharpe: float
    outer_mean_return: float
    test_start: str
    test_end: str


@dataclass
class SelectionGovernanceReport:
    selected_candidate: str
    outer_folds: list[OuterFoldSelection]
    pbo: float
    dsr_probability: float
    spa_pvalue: float
    cumulative_trials: int
    losing_outer_fold_rate: float
    accepted: bool
    rejection_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate": self.selected_candidate,
            "outer_folds": [fold.__dict__ for fold in self.outer_folds],
            "pbo": self.pbo,
            "dsr_probability": self.dsr_probability,
            "spa_pvalue": self.spa_pvalue,
            "cumulative_trials": self.cumulative_trials,
            "losing_outer_fold_rate": self.losing_outer_fold_rate,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
        }


def _safe_sharpe(series: pd.Series, periods_per_year: int) -> float:
    value = sharpe_ratio(series.dropna(), periods_per_year=periods_per_year)
    return float(value) if np.isfinite(value) else -1e9


def _subset_label_times(
    times: pd.Series,
    label_end_times: pd.Series,
    indices: np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    subset_times = times.iloc[indices].reset_index(drop=True)
    subset_end = label_end_times.iloc[indices].reset_index(drop=True)
    subset_times.index = subset_end.index
    return subset_times, subset_end


def nested_purged_select(
    candidate_returns: pd.DataFrame,
    *,
    label_end_times: pd.Series,
    benchmark_returns: pd.Series | None = None,
    config: NestedSelectionConfig | None = None,
    cumulative_trials: int | None = None,
) -> SelectionGovernanceReport:
    """Select a candidate using inner folds and evaluate only on outer folds.

    ``candidate_returns`` must have a monotonic DatetimeIndex and one column per
    candidate.  The final candidate is chosen by the median inner-fold score
    across outer folds; outer performance never participates in final selection.
    """
    cfg = config or NestedSelectionConfig()
    if candidate_returns is None or candidate_returns.empty:
        raise ValueError("candidate_returns is empty")
    if candidate_returns.shape[1] < 2:
        raise ValueError("nested selection requires at least two candidates")
    returns = candidate_returns.sort_index().astype(float)
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index, errors="raise")
    times = pd.Series(returns.index, index=pd.RangeIndex(len(returns)))
    label_end = pd.Series(pd.to_datetime(label_end_times, errors="coerce").to_numpy())
    if len(label_end) != len(returns) or label_end.isna().any():
        raise ValueError("label_end_times must align one-to-one with candidate_returns")

    outer_cfg = PurgedKFoldConfig(n_splits=cfg.outer_splits, embargo_pct=cfg.embargo_pct)
    outer_results: list[OuterFoldSelection] = []
    candidate_inner_history: dict[str, list[float]] = {str(c): [] for c in returns.columns}

    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(
        purged_kfold_split(times, label_end, outer_cfg)
    ):
        outer_train = returns.iloc[outer_train_idx]
        outer_test = returns.iloc[outer_test_idx]
        inner_times, inner_end = _subset_label_times(times, label_end, outer_train_idx)
        inner_cfg = PurgedKFoldConfig(n_splits=cfg.inner_splits, embargo_pct=cfg.embargo_pct)
        scores: dict[str, list[float]] = {str(c): [] for c in returns.columns}
        for inner_train_idx, inner_test_idx in purged_kfold_split(inner_times, inner_end, inner_cfg):
            # Candidate ranking is evaluated on the inner validation partition.
            # The inner train partition exists to purge labels and may be used by
            # model-training callers before they materialise candidate returns.
            del inner_train_idx
            validation = outer_train.iloc[inner_test_idx]
            for candidate in returns.columns:
                scores[str(candidate)].append(
                    _safe_sharpe(validation[candidate], cfg.periods_per_year)
                )
        aggregate = {
            candidate: float(np.median(values)) if values else -1e9
            for candidate, values in scores.items()
        }
        selected = max(aggregate, key=aggregate.get)
        for candidate, value in aggregate.items():
            candidate_inner_history[candidate].append(value)
        outer_series = outer_test[selected].dropna()
        outer_results.append(
            OuterFoldSelection(
                fold_index=fold_idx,
                selected_candidate=selected,
                inner_scores=aggregate,
                outer_sharpe=_safe_sharpe(outer_series, cfg.periods_per_year),
                outer_mean_return=float(outer_series.mean()) if not outer_series.empty else float("nan"),
                test_start=str(outer_test.index.min()),
                test_end=str(outer_test.index.max()),
            )
        )

    if not outer_results:
        raise ValueError("no valid outer folds")
    final_scores = {
        candidate: float(np.median(values)) if values else -1e9
        for candidate, values in candidate_inner_history.items()
    }
    selected_candidate = max(final_scores, key=final_scores.get)

    # PBO consumes time-slice performance.  Use daily/cadence returns directly;
    # the function partitions the time axis symmetrically.
    pbo = probability_of_backtest_overfitting(
        returns,
        n_partitions=min(cfg.pbo_partitions, max(4, (len(returns) // 2) * 2)),
    )
    candidate_sharpes = np.asarray(
        [_safe_sharpe(returns[col], cfg.periods_per_year) for col in returns.columns],
        dtype=float,
    )
    if cumulative_trials is not None and cumulative_trials > len(candidate_sharpes):
        # DSR must reflect the full family search count, including rejected runs.
        padding = np.full(cumulative_trials - len(candidate_sharpes), np.nanmedian(candidate_sharpes))
        candidate_sharpes = np.concatenate([candidate_sharpes, padding])
    dsr = deflated_sharpe_ratio(
        returns[selected_candidate],
        candidate_sharpes,
        periods_per_year=cfg.periods_per_year,
    )
    bench = (
        benchmark_returns.reindex(returns.index).fillna(0.0)
        if benchmark_returns is not None
        else pd.Series(0.0, index=returns.index)
    )
    spa = spa_test(returns, bench, n_bootstrap=500, rng_seed=0)
    spa_pvalue = float(spa.get("p_consistent", float("nan")))
    losing_rate = float(
        np.mean([fold.outer_mean_return <= 0 for fold in outer_results])
    )

    reasons: list[str] = []
    if not np.isfinite(pbo) or pbo > cfg.max_pbo:
        reasons.append(f"pbo={pbo:.4f} exceeds {cfg.max_pbo:.4f}")
    if not np.isfinite(dsr) or dsr < cfg.min_dsr_probability:
        reasons.append(f"dsr={dsr:.4f} below {cfg.min_dsr_probability:.4f}")
    if not np.isfinite(spa_pvalue) or spa_pvalue > cfg.max_spa_pvalue:
        reasons.append(f"spa_pvalue={spa_pvalue:.4f} exceeds {cfg.max_spa_pvalue:.4f}")
    if losing_rate > cfg.max_losing_outer_fold_rate:
        reasons.append(
            f"losing_outer_fold_rate={losing_rate:.4f} exceeds "
            f"{cfg.max_losing_outer_fold_rate:.4f}"
        )

    return SelectionGovernanceReport(
        selected_candidate=selected_candidate,
        outer_folds=outer_results,
        pbo=float(pbo),
        dsr_probability=float(dsr),
        spa_pvalue=spa_pvalue,
        cumulative_trials=int(cumulative_trials or len(candidate_sharpes)),
        losing_outer_fold_rate=losing_rate,
        accepted=not reasons,
        rejection_reasons=reasons,
    )
