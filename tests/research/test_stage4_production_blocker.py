from __future__ import annotations

from types import SimpleNamespace

from quantagent.research.governed_model_comparison import GovernedModelComparison


def test_stage4_economic_truth_blocker_overrides_positive_statistical_gates() -> None:
    report = GovernedModelComparison(
        comparison=SimpleNamespace(as_dict=lambda: {}),
        promotion=SimpleNamespace(accepted=True, to_dict=lambda: {"accepted": True}),
        holdout=SimpleNamespace(accepted=True, to_dict=lambda: {"accepted": True}),
        governance={"economicBacktestCertified": False},
    )

    assert report.production_eligible is False
    assert report.to_dict()["productionEligible"] is False


def test_legacy_governed_result_remains_backward_compatible_without_blocker() -> None:
    report = GovernedModelComparison(
        comparison=SimpleNamespace(as_dict=lambda: {}),
        promotion=SimpleNamespace(accepted=True, to_dict=lambda: {"accepted": True}),
    )

    assert report.production_eligible is True
