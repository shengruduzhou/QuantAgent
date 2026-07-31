"""Out-of-sample robustness scoring for fusion candidates.

Robustness is the fourth declared objective and the one most easily faked, so it
is built from four independent pieces of evidence rather than a single number:

``fold_consistency``
    Did the blend work in *every* out-of-sample fold, or did one lucky fold
    carry it? Combines the fraction of folds with a positive excess return and a
    dispersion penalty on the fold-to-fold spread.
``1 - PBO``
    Combinatorially-symmetric cross-validation probability that the in-sample
    winner lands below the out-of-sample median. Needs the whole candidate
    matrix, so it is a property of the *search*, applied to each candidate's
    selection risk.
``deflated Sharpe probability``
    Probabilistic Sharpe ratio with the threshold inflated by the number of
    trials actually run. ``n_trials`` comes from the search, never from input.
``regime_consistency``
    Worst-segment performance floor: the minimum fold Sharpe, squashed to
    ``[0, 1]``. A blend that earns its return in one regime and gives it back in
    another scores near zero here even with a strong average.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.quant_math.performance import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)


@dataclass(frozen=True)
class RobustnessWeights:
    """Blend weights for the four robustness components. Must sum to 1."""

    fold_consistency: float = 0.35
    overfitting: float = 0.25
    deflated_sharpe: float = 0.25
    regime_consistency: float = 0.15

    def __post_init__(self) -> None:
        total = (
            self.fold_consistency
            + self.overfitting
            + self.deflated_sharpe
            + self.regime_consistency
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"robustness weights must sum to 1, got {total:.6f}")


@dataclass(frozen=True)
class RobustnessInputs:
    """Per-candidate evidence assembled by the search."""

    fold_excess_returns: list[float]
    fold_sharpes: list[float]
    period_returns: pd.Series
    periods_per_year: int
    candidate_sharpes: np.ndarray
    pbo: float


def fold_consistency(fold_excess_returns: list[float]) -> float:
    """Fraction of positive folds, penalised by fold-to-fold dispersion.

    Returns a value in ``[0, 1]``. A single fold cannot demonstrate consistency,
    so it scores at most 0.5.
    """
    values = np.asarray(
        [value for value in fold_excess_returns if np.isfinite(value)], dtype=float
    )
    if values.size == 0:
        return 0.0
    positive_fraction = float((values > 0).mean())
    if values.size == 1:
        return 0.5 * positive_fraction
    mean = float(values.mean())
    dispersion = float(values.std(ddof=1))
    if dispersion <= 1e-12:
        stability = 1.0
    else:
        # Coefficient-of-variation style penalty, bounded so a near-zero mean
        # cannot drive the score negative.
        stability = float(np.clip(abs(mean) / (abs(mean) + dispersion), 0.0, 1.0))
    return float(np.clip(0.6 * positive_fraction + 0.4 * stability, 0.0, 1.0))


def regime_consistency(fold_sharpes: list[float]) -> float:
    """Squash the worst fold Sharpe onto ``[0, 1]`` with 0 mapping to 0.5."""
    values = [value for value in fold_sharpes if np.isfinite(value)]
    if not values:
        return 0.0
    worst = float(min(values))
    return float(1.0 / (1.0 + np.exp(-worst)))


def deflated_sharpe_probability(
    period_returns: pd.Series,
    candidate_sharpes: np.ndarray,
    periods_per_year: int,
) -> float:
    """Deflated Sharpe probability, or 0.0 when the sample cannot support it.

    Units matter here and the mistake is silent. ``deflated_sharpe_ratio``
    derives the selection-bias threshold as ``sqrt(Var[SR])`` scaled by the
    expected maximum of N draws, then hands it to the probabilistic Sharpe
    ratio, which internally divides by ``sqrt(periods_per_year)``. The threshold
    must therefore be built from **per-period** Sharpe ratios. Feeding it
    annualised ones inflates the bar by ``sqrt(periods_per_year)`` and drives
    the result to ~0 for every candidate, which looks like a strict test but is
    really a dead term that carries no information.

    Callers hold annualised Sharpes because that is what the rest of the
    platform reports, so the conversion happens here, once.
    """
    periods = max(1, int(periods_per_year))
    sharpes = np.asarray(candidate_sharpes, dtype=float)
    if period_returns.dropna().size < 4 or sharpes.size < 2:
        return 0.0
    per_period_sharpes = sharpes / np.sqrt(periods)
    value = deflated_sharpe_ratio(
        period_returns,
        per_period_sharpes,
        periods_per_year=periods,
    )
    return float(value) if np.isfinite(value) else 0.0


def search_pbo(performance_matrix: pd.DataFrame, n_partitions: int = 16) -> float:
    """PBO for the whole search, or ``NaN`` when the matrix is too small.

    ``performance_matrix`` is ``(time slices, candidates)``. We degrade to
    ``NaN`` rather than raising: a search with too few slices genuinely has no
    PBO estimate, and reporting 0.0 would understate overfitting risk.
    """
    if performance_matrix.empty or performance_matrix.shape[1] < 2:
        return float("nan")
    partitions = int(n_partitions)
    if partitions % 2 != 0:
        partitions -= 1
    while partitions >= 4 and performance_matrix.shape[0] < partitions:
        partitions -= 2
    if partitions < 4:
        return float("nan")
    try:
        return float(
            probability_of_backtest_overfitting(
                performance_matrix, n_partitions=partitions
            )
        )
    except ValueError:
        return float("nan")


def robustness_score(
    inputs: RobustnessInputs,
    weights: RobustnessWeights | None = None,
) -> tuple[float, dict[str, float]]:
    """Return ``(score, component_breakdown)`` with score in ``[0, 1]``."""
    blend = weights or RobustnessWeights()
    consistency = fold_consistency(inputs.fold_excess_returns)
    regime = regime_consistency(inputs.fold_sharpes)
    dsr = deflated_sharpe_probability(
        inputs.period_returns, inputs.candidate_sharpes, inputs.periods_per_year
    )
    # A missing PBO estimate is treated as no evidence (0.5), not as a pass.
    overfitting_term = 0.5 if not np.isfinite(inputs.pbo) else float(1.0 - inputs.pbo)

    score = (
        blend.fold_consistency * consistency
        + blend.overfitting * overfitting_term
        + blend.deflated_sharpe * dsr
        + blend.regime_consistency * regime
    )
    breakdown = {
        "foldConsistency": round(consistency, 6),
        "overfittingResistance": round(overfitting_term, 6),
        "deflatedSharpeProbability": round(dsr, 6),
        "regimeConsistency": round(regime, 6),
        "pbo": None if not np.isfinite(inputs.pbo) else round(float(inputs.pbo), 6),
    }
    return float(np.clip(score, 0.0, 1.0)), breakdown


__all__ = [
    "RobustnessInputs",
    "RobustnessWeights",
    "deflated_sharpe_probability",
    "fold_consistency",
    "regime_consistency",
    "robustness_score",
    "search_pbo",
]
