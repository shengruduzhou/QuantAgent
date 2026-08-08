"""Corrected statistical layer for the factor-fusion search.

The legacy search correctly separates train/test folds and correctly avoids
compounding overlapping forward labels inside each cohort.  Its statistical
post-processing, however, used the horizon-return observations as if they were
an independent lower-frequency series and concatenated fold NAVs that each
restart near 1.0.  Both choices contaminate PBO/DSR evidence.

This module keeps the fitted candidates and scalar cohort metrics from the
legacy search, but reconstructs the *statistical* OOS record from within-fold
NAV changes only, stitches folds without reset jumps, and evaluates PBO/DSR on
the resulting business-daily staggered portfolio return series.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from quantagent.fusion.frontier import pareto_front, rank_by_preference
from quantagent.fusion.robustness import fold_consistency, regime_consistency, search_pbo
from quantagent.fusion.search import (
    FusionCandidate,
    FusionFoldResult,
    FusionSearchConfig,
    FusionSearchResult,
    ProgressCallback,
    save_fusion_artifacts,
)
from quantagent.fusion.search import run_fusion_search as _legacy_run_fusion_search
from quantagent.quant_math.performance import deflated_sharpe_ratio, sharpe_ratio


def _within_fold_returns(
    nav: pd.Series,
    fold_windows: list[dict[str, str]],
) -> pd.Series:
    """Remove artificial returns created by concatenating fold-local NAVs.

    Every fold NAV starts from fresh capital.  Therefore the first NAV point of
    a fold has no economic return relative to the last point of the previous
    fold and must never enter DSR, SPA, PBO, Sharpe, or a stitched NAV.
    """
    clean = pd.Series(nav, dtype=float).dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float)
    clean.index = pd.to_datetime(clean.index)
    pieces: list[pd.Series] = []
    for window in fold_windows:
        start = pd.Timestamp(window["testStart"])
        end = pd.Timestamp(window["testEnd"])
        local = clean.loc[(clean.index >= start) & (clean.index <= end)]
        if len(local) < 2:
            continue
        returns = local.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
        if not returns.empty:
            pieces.append(returns)
    if not pieces:
        return pd.Series(dtype=float)
    out = pd.concat(pieces).sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).mean()
    return out.astype(float)


def _stitched_nav(returns: pd.Series) -> pd.Series:
    clean = pd.Series(returns, dtype=float).dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float)
    return (1.0 + clean).cumprod()


def _corrected_robustness(
    candidate: FusionCandidate,
    *,
    daily_returns: pd.Series,
    measured_annual_sharpes: np.ndarray,
    n_trials: int,
    pbo: float,
    config: FusionSearchConfig,
) -> tuple[float, dict[str, float | None]]:
    fold_excess = [float(fold.metrics.get("excessReturn", 0.0)) for fold in candidate.folds]
    fold_sharpes = [float(fold.metrics.get("sharpe", 0.0)) for fold in candidate.folds]
    consistency = fold_consistency(fold_excess)
    regime = regime_consistency(fold_sharpes)

    dsr = float("nan")
    if len(daily_returns) >= 4 and measured_annual_sharpes.size >= 2:
        dsr = deflated_sharpe_ratio(
            daily_returns,
            measured_annual_sharpes / np.sqrt(252.0),
            periods_per_year=252,
            n_trials=max(int(n_trials), int(measured_annual_sharpes.size)),
        )
    dsr_term = float(dsr) if np.isfinite(dsr) else 0.0
    overfitting_term = 0.5 if not np.isfinite(pbo) else float(np.clip(1.0 - pbo, 0.0, 1.0))
    weights = config.robustness_weights
    score = (
        weights.fold_consistency * consistency
        + weights.overfitting * overfitting_term
        + weights.deflated_sharpe * dsr_term
        + weights.regime_consistency * regime
    )
    breakdown: dict[str, float | None] = {
        "foldConsistency": round(float(consistency), 6),
        "overfittingResistance": round(overfitting_term, 6),
        "deflatedSharpeProbability": None if not np.isfinite(dsr) else round(float(dsr), 6),
        "regimeConsistency": round(float(regime), 6),
        "pbo": None if not np.isfinite(pbo) else round(float(pbo), 6),
        "statisticalPeriodsPerYear": 252.0,
    }
    return float(np.clip(score, 0.0, 1.0)), breakdown


def run_fusion_search(*args: Any, **kwargs: Any) -> FusionSearchResult:
    """Run the existing governed search and correct its OOS statistical record."""
    raw = _legacy_run_fusion_search(*args, **kwargs)

    returns_by_candidate = {
        candidate.candidate_id: _within_fold_returns(candidate.nav, raw.fold_windows)
        for candidate in raw.candidates
    }
    performance_matrix = pd.DataFrame(returns_by_candidate).sort_index()
    complete = performance_matrix.dropna(how="any")
    pbo = search_pbo(complete) if not complete.empty else float("nan")

    annual_sharpes: list[float] = []
    for series in returns_by_candidate.values():
        if len(series) < 2:
            continue
        value = sharpe_ratio(series, periods_per_year=252)
        if np.isfinite(value):
            annual_sharpes.append(float(value))
    measured = np.asarray(annual_sharpes, dtype=float)

    corrected: list[FusionCandidate] = []
    for candidate in raw.candidates:
        returns = returns_by_candidate[candidate.candidate_id]
        robustness, breakdown = _corrected_robustness(
            candidate,
            daily_returns=returns,
            measured_annual_sharpes=measured,
            n_trials=raw.n_trials,
            pbo=pbo,
            config=raw.config,
        )
        metrics = dict(candidate.metrics)
        metrics["robustness"] = round(robustness, 6)
        corrected.append(
            replace(
                candidate,
                metrics=metrics,
                robustness_breakdown=breakdown,
                nav=_stitched_nav(returns),
            )
        )

    serialisable = [candidate.as_dict() for candidate in corrected]
    frontier = pareto_front(serialisable)
    ranking = rank_by_preference(
        [item for item in serialisable if str(item["id"]) in frontier],
        raw.config.preference,
    )
    return replace(
        raw,
        candidates=corrected,
        frontier=frontier,
        ranking=ranking,
        pbo=pbo,
    )


__all__ = [
    "FusionCandidate",
    "FusionFoldResult",
    "FusionSearchConfig",
    "FusionSearchResult",
    "ProgressCallback",
    "run_fusion_search",
    "save_fusion_artifacts",
]
