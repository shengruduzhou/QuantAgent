from __future__ import annotations

from types import SimpleNamespace

from quantagent.research.governed_model_comparison import GovernedModelComparison


def _comparison() -> SimpleNamespace:
    return SimpleNamespace(as_dict=lambda: {})


def _promotion(*, accepted: bool = True) -> SimpleNamespace:
    return SimpleNamespace(accepted=accepted, to_dict=lambda: {"accepted": accepted})


def _holdout(*, accepted: bool = True) -> SimpleNamespace:
    return SimpleNamespace(accepted=accepted, to_dict=lambda: {"accepted": accepted})


def test_stage4_economic_truth_blocker_overrides_positive_statistical_gates() -> None:
    report = GovernedModelComparison(
        comparison=_comparison(),
        promotion=_promotion(),
        holdout=_holdout(),
        governance={"stage4Governed": True, "economicBacktestCertified": False},
    )

    assert report.production_eligible is False
    assert report.to_dict()["productionEligible"] is False


def test_legacy_or_missing_governance_fails_closed() -> None:
    report = GovernedModelComparison(
        comparison=_comparison(),
        promotion=_promotion(),
    )

    assert report.production_eligible is False
    assert report.to_dict()["productionEligible"] is False


def test_statistical_promotion_without_one_shot_holdout_fails_closed() -> None:
    report = GovernedModelComparison(
        comparison=_comparison(),
        promotion=_promotion(),
        governance={"stage4Governed": True, "economicBacktestCertified": True},
    )

    assert report.production_eligible is False


def test_non_stage4_result_cannot_become_production_eligible_even_if_economic_flag_is_true() -> None:
    report = GovernedModelComparison(
        comparison=_comparison(),
        promotion=_promotion(),
        holdout=_holdout(),
        governance={"stage4Governed": False, "economicBacktestCertified": True},
    )

    assert report.production_eligible is False


def test_production_eligibility_requires_all_authoritative_gates() -> None:
    report = GovernedModelComparison(
        comparison=_comparison(),
        promotion=_promotion(),
        holdout=_holdout(),
        governance={"stage4Governed": True, "economicBacktestCertified": True},
    )

    assert report.production_eligible is True
    assert report.to_dict()["productionEligible"] is True
