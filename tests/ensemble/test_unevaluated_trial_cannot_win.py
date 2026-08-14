"""A trial that never traded must not win the strict search.

When the decision chain rejects every candidate the target book is empty and no
backtest runs. `_empty_metrics` used to report that as
`annualized_return=0.0, max_drawdown=0.0, excess_return_ann=0.0` -- zeros that
are not neutral placeholders but false claims which happen to score well:

* holding no position while the benchmark returns +18% has an annual excess of
  **-0.18**, not 0.0;
* `max_drawdown=0.0` reads as a flawless risk profile rather than an absent one.

With the repo's default weights the do-nothing trial scored 0.0 while a real
+12%/yr policy against a +18% benchmark scored -0.026, so the argmax preferred
trading nothing. That is not an edge case here: these models are documented as
underperforming their own universe, so a negative score is the normal case.
"""

from __future__ import annotations

import math

import pytest

from quantagent.ensemble.strict_factor_search import _empty_metrics
from quantagent.ensemble.strict_policy_search import (
    StrictPolicySearchConfig,
    _score_metrics,
)

REAL_POLICY = {
    "annualized_return": 0.12,
    "excess_return_ann": -0.06,
    "max_drawdown": 0.15,
    "turnover": 0.30,
    "total_cost": 5_000.0,
}


def _cfg() -> StrictPolicySearchConfig:
    return StrictPolicySearchConfig()


class TestEmptyMetricsAreNotZero:
    @pytest.mark.parametrize(
        "key",
        ["annualized_return", "max_drawdown", "excess_return_ann", "sharpe", "calmar"],
    )
    def test_unmeasured_performance_is_nan(self, key):
        assert math.isnan(_empty_metrics()[key]), (
            f"{key} must be NaN when no backtest ran; 0.0 is a false claim"
        )

    def test_the_trial_is_flagged_as_not_evaluated(self):
        assert _empty_metrics()["evaluated"] == 0.0


class TestScoring:
    def test_unevaluated_trial_scores_negative_infinity(self):
        assert _score_metrics(_empty_metrics(), _cfg()) == float("-inf")

    def test_a_losing_but_real_policy_beats_doing_nothing(self):
        do_nothing = _score_metrics(_empty_metrics(), _cfg())
        real = _score_metrics(REAL_POLICY, _cfg())
        assert real > do_nothing
        assert real < 0.0, "the real policy is genuinely negative-scoring here"

    def test_argmax_is_not_captured_when_the_empty_trial_comes_first(self):
        """NaN would not fix this: `x > nan` is False, so a NaN first trial
        would become the incumbent and never be displaced."""
        best = None
        for score in (_score_metrics(_empty_metrics(), _cfg()),
                      _score_metrics(REAL_POLICY, _cfg())):
            if best is None or score > best:
                best = score
        assert best == pytest.approx(_score_metrics(REAL_POLICY, _cfg()))

    def test_a_real_trial_still_ranks_normally_against_another_real_trial(self):
        better = dict(REAL_POLICY, annualized_return=0.20)
        assert _score_metrics(better, _cfg()) > _score_metrics(REAL_POLICY, _cfg())

    def test_all_trials_empty_yields_no_positive_winner(self):
        scores = [_score_metrics(_empty_metrics(), _cfg()) for _ in range(3)]
        assert all(s == float("-inf") for s in scores), (
            "if every candidate was rejected the search must not surface one as a win"
        )
