"""Deflated-Sharpe inputs: neither degenerate candidates nor unmeasured trials
may decide the gate.

Two ways the sample handed to ``deflated_sharpe_ratio`` used to drive the
result instead of the evidence:

* ``_safe_sharpe`` substitutes a worst-possible sentinel when a candidate's
  Sharpe does not exist, which is right for the ``argmax`` that picks a fold
  winner and wrong for the deflated-Sharpe family: the selection-bias benchmark
  is derived from ``var(candidate_sharpes)``, so one sentinel sets the
  benchmark by itself and the gate rejects every champion on a numerical
  artifact.
* Trials that were run but not measured used to be imputed into that same
  sample as family-median draws. A constant pad shrinks the variance like
  ``1/n_trials`` while the order-statistic term grows only like
  ``sqrt(log n_trials)``, so declaring more data mining made the gate *easier*
  to pass -- backwards for a multiple-testing correction, and a false-accept
  route in the production training gate. Dispersion and trial count are now
  separate arguments; see ``test_dsr_is_monotonically_non_increasing_*``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.quant_math.performance import deflated_sharpe_ratio, sharpe_ratio
from quantagent.research.selection_governance import (
    NestedSelectionConfig,
    _family_sharpes,
    evaluate_frozen_candidate,
    nested_purged_select,
)

PERIODS_PER_YEAR = 252
OBSERVED_DAYS = 160


def _family(*, champion_drift: float, with_degenerate: bool, seed: int = 11) -> pd.DataFrame:
    """A candidate family, optionally holding a book that never trades."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-02", periods=OBSERVED_DAYS, freq="B")
    columns = {
        "champion": rng.normal(champion_drift, 0.010, len(index)),
        "challenger_a": rng.normal(0.0002, 0.011, len(index)),
        "challenger_b": rng.normal(0.0001, 0.012, len(index)),
    }
    if with_degenerate:
        # Zero variance and zero mean: Sharpe is 0/0, i.e. undefined.
        columns["never_trades"] = np.zeros(len(index))
    return pd.DataFrame(columns, index=index)


def test_constant_return_candidate_leaves_a_live_dsr_for_a_strong_champion() -> None:
    """The regression: one never-trading column used to zero the whole gate."""
    returns = _family(champion_drift=0.0016, with_degenerate=True)
    assert not np.isfinite(sharpe_ratio(returns["never_trades"], periods_per_year=PERIODS_PER_YEAR))

    report = evaluate_frozen_candidate(
        returns,
        selected_candidate="champion",
        minimum_observed_days=OBSERVED_DAYS,
    )

    assert report.dsr_probability > 0.5, (
        "a champion at ~3.4 annualised Sharpe must retain a live deflated Sharpe "
        "when a co-tried candidate merely never traded"
    )
    assert np.isfinite(report.dsr_probability)


def test_dsr_is_barely_moved_by_adding_a_candidate_that_never_trades() -> None:
    """Evidence, not the placeholder, has to drive the benchmark."""
    without = evaluate_frozen_candidate(
        _family(champion_drift=0.0016, with_degenerate=False),
        selected_candidate="champion",
        minimum_observed_days=OBSERVED_DAYS,
    )
    with_degenerate = evaluate_frozen_candidate(
        _family(champion_drift=0.0016, with_degenerate=True),
        selected_candidate="champion",
        minimum_observed_days=OBSERVED_DAYS,
    )

    assert without.dsr_probability > 0.5  # guards against comparing 0.0 to 0.0
    assert with_degenerate.dsr_probability == pytest.approx(
        without.dsr_probability, abs=0.05
    )
    # The degenerate candidate was still tried, so it still costs a trial and
    # cannot make the gate easier to pass.
    assert with_degenerate.cumulative_trials == 4
    assert without.cumulative_trials == 3
    assert with_degenerate.dsr_probability <= without.dsr_probability


def test_gate_still_rejects_a_champion_without_edge() -> None:
    """The repair must not switch the gate on for everything."""
    returns = _family(champion_drift=0.0001, with_degenerate=True)

    report = evaluate_frozen_candidate(
        returns,
        selected_candidate="champion",
        minimum_observed_days=OBSERVED_DAYS,
    )

    assert report.dsr_probability < 0.95
    assert not report.accepted
    assert any(reason.startswith("dsr=") for reason in report.rejection_reasons)


def test_family_of_only_degenerate_candidates_rejects_rather_than_accepts() -> None:
    """With nothing measurable, the benchmark is unestimable — reject, not pass."""
    index = pd.date_range("2024-01-02", periods=OBSERVED_DAYS, freq="B")
    returns = pd.DataFrame(
        {"flat_a": np.zeros(len(index)), "flat_b": np.zeros(len(index))},
        index=index,
    )

    report = evaluate_frozen_candidate(
        returns,
        selected_candidate="flat_a",
        minimum_observed_days=OBSERVED_DAYS,
    )

    assert not np.isfinite(report.dsr_probability)
    assert not report.accepted


def test_nested_selection_dsr_survives_a_degenerate_candidate() -> None:
    returns = _family(champion_drift=0.0016, with_degenerate=True)
    label_end_times = pd.Series(returns.index + pd.Timedelta(days=1))

    report = nested_purged_select(
        returns,
        label_end_times=label_end_times,
        cumulative_trials=len(returns.columns),
    )

    assert report.dsr_probability > 0.5
    # The sentinel is retained where it belongs: a book that never trades can
    # never win an inner-fold argmax.
    assert report.selected_candidate != "never_trades"
    assert all(fold.selected_candidate != "never_trades" for fold in report.outer_folds)


def test_family_sharpes_drops_undefined_and_returns_per_period_units() -> None:
    returns = _family(champion_drift=0.0016, with_degenerate=True)

    sample = _family_sharpes(returns, PERIODS_PER_YEAR)

    assert np.isfinite(sample).all()
    # Only the three measurable candidates. The fourth was tried and is
    # declared through n_trials; it is never imputed into the sample.
    assert sample.size == 3
    # Per-period, not annualised: the largest entry is the champion's Sharpe
    # divided by sqrt(252), which is what deflated_sharpe_ratio re-annualises.
    champion_annualised = sharpe_ratio(returns["champion"], periods_per_year=PERIODS_PER_YEAR)
    assert sample.max() == pytest.approx(champion_annualised / np.sqrt(PERIODS_PER_YEAR))


def test_family_sharpes_reports_none_when_every_candidate_is_degenerate() -> None:
    index = pd.date_range("2024-01-02", periods=OBSERVED_DAYS, freq="B")
    flat = pd.DataFrame(
        {"flat_a": np.zeros(len(index)), "flat_b": np.zeros(len(index))}, index=index
    )

    empty = _family_sharpes(flat, PERIODS_PER_YEAR)

    assert empty.size == 0
    assert not np.isfinite(deflated_sharpe_ratio(flat["flat_a"], empty, n_trials=2))


# --------------------------------------------------------------------------- #
# Trial accounting: more declared mining may only ever tighten the gate        #
# --------------------------------------------------------------------------- #

DECLARED_TRIALS = [3, 4, 10, 50, 500, 5000]


def test_dsr_is_monotonically_non_increasing_in_declared_trials() -> None:
    """The core property of a multiple-testing correction, at the metric."""
    returns = _family(champion_drift=0.0016, with_degenerate=False)
    dispersion = _family_sharpes(returns, PERIODS_PER_YEAR)

    scores = [
        deflated_sharpe_ratio(
            returns["champion"],
            dispersion,
            periods_per_year=PERIODS_PER_YEAR,
            n_trials=n,
        )
        for n in DECLARED_TRIALS
    ]

    assert all(np.isfinite(score) for score in scores)
    for lower, higher, score_lower, score_higher in zip(
        DECLARED_TRIALS, DECLARED_TRIALS[1:], scores, scores[1:]
    ):
        assert score_higher <= score_lower, (
            f"declaring {higher} trials instead of {lower} raised DSR "
            f"{score_lower:.6f} -> {score_higher:.6f}; a wider search must "
            "never look safer"
        )
    # Not merely flat: the correction has to actually bite.
    assert scores[-1] < scores[0] - 0.10


def test_frozen_gate_dsr_is_monotonically_non_increasing_in_cumulative_trials() -> None:
    """Same property end-to-end, through the gate the trainer calls."""
    returns = _family(champion_drift=0.0016, with_degenerate=False)

    scores = [
        evaluate_frozen_candidate(
            returns,
            selected_candidate="champion",
            cumulative_trials=n,
            minimum_observed_days=OBSERVED_DAYS,
        ).dsr_probability
        for n in DECLARED_TRIALS
    ]

    assert all(np.isfinite(score) for score in scores)
    assert scores == sorted(scores, reverse=True), (
        f"DSR must not rise with declared trials, got {dict(zip(DECLARED_TRIALS, scores))}"
    )


def test_nested_selection_dsr_is_monotonically_non_increasing_in_cumulative_trials() -> None:
    returns = _family(champion_drift=0.0016, with_degenerate=False)
    label_end_times = pd.Series(returns.index + pd.Timedelta(days=1))

    scores = [
        nested_purged_select(
            returns,
            label_end_times=label_end_times,
            cumulative_trials=n,
        ).dsr_probability
        for n in DECLARED_TRIALS
    ]

    assert all(np.isfinite(score) for score in scores)
    assert scores == sorted(scores, reverse=True), (
        f"DSR must not rise with declared trials, got {dict(zip(DECLARED_TRIALS, scores))}"
    )


def test_heavily_mined_champion_is_rejected_where_a_lightly_mined_one_passes() -> None:
    """The false accept this guards: the trainer raises on rejection, so a gate
    that loosened under mining would have promoted an overfit champion.

    Keep this synthetic case comfortably on opposite sides of the fixed 0.95
    DSR policy under the corrected Pearson-kurtosis PSR denominator.
    """
    returns = _family(champion_drift=0.00162, with_degenerate=False)

    lightly_mined = evaluate_frozen_candidate(
        returns,
        selected_candidate="champion",
        cumulative_trials=3,
        minimum_observed_days=OBSERVED_DAYS,
    )
    heavily_mined = evaluate_frozen_candidate(
        returns,
        selected_candidate="champion",
        cumulative_trials=500,
        minimum_observed_days=OBSERVED_DAYS,
    )

    assert lightly_mined.dsr_probability >= NestedSelectionConfig().min_dsr_probability
    assert heavily_mined.dsr_probability < NestedSelectionConfig().min_dsr_probability
    assert not heavily_mined.accepted
    assert any(reason.startswith("dsr=") for reason in heavily_mined.rejection_reasons)


def test_n_trials_defaults_to_the_measured_sample_size() -> None:
    """Declaring exactly what was measured must change nothing."""
    returns = _family(champion_drift=0.0016, with_degenerate=False)
    dispersion = _family_sharpes(returns, PERIODS_PER_YEAR)

    inferred = deflated_sharpe_ratio(
        returns["champion"], dispersion, periods_per_year=PERIODS_PER_YEAR
    )
    declared = deflated_sharpe_ratio(
        returns["champion"],
        dispersion,
        periods_per_year=PERIODS_PER_YEAR,
        n_trials=dispersion.size,
    )

    assert inferred == pytest.approx(declared)


def test_declaring_fewer_trials_than_were_measured_is_rejected() -> None:
    """An incoherent declaration is a caller bug, not a weaker correction."""
    returns = _family(champion_drift=0.0016, with_degenerate=False)
    dispersion = _family_sharpes(returns, PERIODS_PER_YEAR)

    with pytest.raises(ValueError, match="n_trials cannot be smaller"):
        deflated_sharpe_ratio(
            returns["champion"],
            dispersion,
            periods_per_year=PERIODS_PER_YEAR,
            n_trials=dispersion.size - 1,
        )