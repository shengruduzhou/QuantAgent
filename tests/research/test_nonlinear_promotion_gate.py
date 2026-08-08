from __future__ import annotations

import pandas as pd

from quantagent.research.model_comparison import (
    ArmResult,
    ComparisonConfig,
    ComparisonReport,
)
from quantagent.research.nonlinear_promotion import evaluate_nonlinear_promotion


def _report(*, champion: str, points: int) -> ComparisonReport:
    index = pd.bdate_range("2026-01-05", periods=points)
    baseline = ArmResult(
        name="linear_baseline",
        model_class="rank_weighted_additive",
        feature_summary={},
        status="measured",
        daily_returns=pd.Series(0.001, index=index),
    )
    challenger = ArmResult(
        name="gbm",
        model_class="nonlinear_learner",
        feature_summary={},
        status="measured",
        daily_returns=pd.Series(0.002, index=index),
    )
    return ComparisonReport(
        config=ComparisonConfig(horizon_days=5, max_pbo=0.25),
        arms=[baseline, challenger],
        incremental=[],
        champion=champion,
        verdict="production_accepted",
        verdict_reasons=("raw comparison passed",),
        n_trials=2,
        # Deliberately perfect raw evidence: promotion must ignore it and use
        # the non-overlapping cohort record instead.
        pbo=0.0,
        dsr_probability=1.0,
        fold_windows=[],
        generated_at="2026-08-08T00:00:00+00:00",
    )


def test_nonlinear_promotion_does_not_trust_raw_overlapping_dsr() -> None:
    promotion = evaluate_nonlinear_promotion(_report(champion="gbm", points=20))
    assert not promotion.accepted
    assert promotion.final_verdict == "hypothesis_rejected"
    assert any("cohort" in reason.lower() for reason in promotion.rejection_reasons)


def test_linear_control_can_never_be_promoted_as_nonlinear_champion() -> None:
    promotion = evaluate_nonlinear_promotion(_report(champion="linear_baseline", points=80))
    assert not promotion.accepted
    assert any("nonlinear champion" in reason for reason in promotion.rejection_reasons)
