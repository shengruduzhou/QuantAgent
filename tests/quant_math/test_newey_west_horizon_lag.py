"""Guard the horizon-aware HAC truncation lag.

An h-day forward-return label makes consecutive per-date IC observations share
h-1 days of return.  If the Newey-West truncation lag is chosen from the sample
size alone -- which is what the automatic bandwidth does -- the resulting t-stat
stays inflated on long horizons, and pure-noise factors clear a |t|>1.96 screen
far more often than the nominal 5%.

These tests pin the lag *floor* rather than the exact t-stat, so they stay valid
if the estimator's internals are tuned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.factors.evaluation import infer_horizon_days
from quantagent.quant_math.performance import newey_west_auto_lag, newey_west_t_stat


def _overlapping_null(n: int, horizon: int, seed: int) -> pd.Series:
    """A series with a TRUE zero mean whose only autocorrelation is label overlap."""
    rng = np.random.default_rng(seed)
    daily = rng.standard_normal(n + horizon)
    rolled = pd.Series(daily).rolling(horizon).mean().dropna()
    return rolled.iloc[:n].reset_index(drop=True)


class TestAutoLag:
    def test_auto_bandwidth_is_sized_from_n_only(self):
        # The documented Newey-West (1994) rule: 4 * (n/100)^(2/9).
        assert newey_west_auto_lag(250) == 4
        assert newey_west_auto_lag(1000) == 6
        assert newey_west_auto_lag(2500) == 8

    def test_auto_bandwidth_is_far_short_of_a_20d_label(self):
        # This is the whole reason horizon_days exists: an h=20 label needs 19.
        assert newey_west_auto_lag(2500) < 20 - 1


class TestHorizonFloor:
    @pytest.mark.parametrize("horizon", [5, 10, 20, 60])
    def test_horizon_raises_the_lag_floor(self, horizon):
        """A longer horizon must not produce a LARGER t-stat than a short one.

        Same data, same true (zero) mean: widening the truncation lag can only
        admit more autocovariance, so |t| must be non-increasing in the lag.
        """
        series = _overlapping_null(n=600, horizon=horizon, seed=7)
        auto = abs(newey_west_t_stat(series))
        floored = abs(newey_west_t_stat(series, horizon_days=horizon))
        assert floored <= auto + 1e-9, (
            f"horizon={horizon}: floored |t|={floored} exceeded auto |t|={auto}, "
            "which means the horizon floor did not widen the lag"
        )

    def test_short_horizon_keeps_the_longer_automatic_lag(self):
        """max(auto, h-1) -- never shrink a correctly-large automatic bandwidth."""
        series = _overlapping_null(n=2500, horizon=5, seed=3)
        # auto lag at n=2500 is 8, which already exceeds h-1 = 4.
        assert newey_west_t_stat(series, horizon_days=5) == pytest.approx(
            newey_west_t_stat(series)
        )

    def test_explicit_max_lag_still_wins(self):
        series = _overlapping_null(n=400, horizon=20, seed=5)
        assert newey_west_t_stat(series, 2, horizon_days=20) == pytest.approx(
            newey_west_t_stat(series, 2)
        )

    def test_rejects_nonsense_horizon(self):
        series = _overlapping_null(n=200, horizon=5, seed=1)
        with pytest.raises(ValueError, match="horizon_days"):
            newey_west_t_stat(series, horizon_days=0)

    def test_lag_cannot_exceed_available_observations(self):
        series = pd.Series([0.1, -0.2, 0.3, 0.05])
        assert np.isfinite(newey_west_t_stat(series, horizon_days=250))


class TestFalsePositiveRate:
    def test_horizon_floor_materially_reduces_false_positives_at_h20(self):
        """The measured payoff, at reduced trial count to stay fast.

        Under a true null the nominal rejection rate at |t|>1.96 is 5%.  With the
        automatic lag an h=20 screen rejects ~22-38%; with the floor, ~12-17%.
        We assert only the *direction and materiality* of the improvement, since
        the absolute rate is sample dependent.
        """
        trials = 400
        n, horizon = 500, 20
        auto_hits = floored_hits = 0
        for seed in range(trials):
            series = _overlapping_null(n=n, horizon=horizon, seed=1000 + seed)
            auto_hits += abs(newey_west_t_stat(series)) > 1.96
            floored_hits += abs(newey_west_t_stat(series, horizon_days=horizon)) > 1.96

        auto_rate = auto_hits / trials
        floored_rate = floored_hits / trials
        assert auto_rate > 0.15, f"expected an inflated baseline rate, got {auto_rate:.1%}"
        assert floored_rate < auto_rate * 0.75, (
            f"horizon floor did not materially help: {auto_rate:.1%} -> {floored_rate:.1%}"
        )

    def test_the_floor_does_not_claim_a_correctly_sized_test(self):
        """Honesty guard: the floor improves size, it does not restore 5%.

        Bartlett-kernel HAC is undersized in finite samples when lag/n is large.
        If a future change makes this assertion fail because the rate genuinely
        reached ~5%, verify it with a fresh null study before relaxing the
        docstring -- do not simply delete this test.
        """
        trials = 400
        n, horizon = 500, 20
        hits = sum(
            abs(newey_west_t_stat(_overlapping_null(n=n, horizon=horizon, seed=2000 + s),
                                  horizon_days=horizon)) > 1.96
            for s in range(trials)
        )
        assert hits / trials > 0.07, (
            "the horizon floor appears to have produced a correctly-sized test; "
            "confirm with a full null study rather than assuming it"
        )


class TestHorizonInference:
    @pytest.mark.parametrize(
        "column,expected",
        [
            ("forward_return_1d", 1),
            ("forward_return_5d", 5),
            ("forward_return_20d", 20),
            ("forward_return_120d", 120),
            ("future_20d_return", 20),
            ("exec_return_10d", 10),
            ("excess_forward_return_5d", 5),
        ],
    )
    def test_infers_repo_label_conventions(self, column, expected):
        assert infer_horizon_days(column) == expected

    @pytest.mark.parametrize("column", ["ret", "label", "some_random_col", "alpha001"])
    def test_returns_none_rather_than_defaulting_to_one(self, column):
        """None, never 1.

        Defaulting an unrecognised column to a 1-day horizon would silently
        reinstate the under-corrected t-stat -- a missing measurement replaced
        by a plausible default, which is precisely the failure this guards.
        """
        assert infer_horizon_days(column) is None
