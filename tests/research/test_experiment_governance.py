from __future__ import annotations

import json

import numpy as np
import pytest

from quantagent.research.experiment_governance import (
    ExperimentEvent,
    ExperimentLedger,
    ExperimentSpec,
    FinalHoldoutLedger,
    FinalHoldoutSpec,
    with_cumulative_trial_count,
)
from quantagent.research.governed_model_comparison import (
    _qualify_final_holdout,
    _validate_stage4_contract,
)
from quantagent.research.model_comparison import (
    ArmResult,
    ComparisonConfig,
    ComparisonReport,
    LINEAR_BASELINE,
)


def _experiment(**overrides) -> ExperimentSpec:
    values = dict(
        family="nonlinear_factor_v1",
        candidate_id="gbm_plus_pair",
        parameters={"depth": 5, "features": ["value", "momentum"]},
        dataset_hash="data-sha-001",
        train_window=("2018-01-01", "2023-12-31"),
        search_window=("2024-01-01", "2025-12-31"),
        metric="rank_ic_then_net_return",
        git_hash="abc123",
        declared_trial_count=12,
        recipe_hash="recipe-sha",
        split_id="wf-v4",
    )
    values.update(overrides)
    return ExperimentSpec(**values)


def _holdout(**overrides) -> FinalHoldoutSpec:
    values = dict(
        family="nonlinear_factor_v1",
        dataset_hash="data-sha-001",
        holdout_window=("2026-01-01", "2026-04-30"),
        label_contract_hash="label-sha",
    )
    values.update(overrides)
    return FinalHoldoutSpec(**values)


def _stage4_config(**overrides) -> ComparisonConfig:
    values = dict(
        label_column="forward_executable_return_5d",
        horizon_days=5,
        n_folds=4,
        holdout_folds=1,
    )
    values.update(overrides)
    return ComparisonConfig(**values)


def test_fingerprint_is_stable_across_event_time_and_status():
    spec = _experiment()
    first = ExperimentEvent(spec, status="registered", created_at="2026-01-01T00:00:00+00:00")
    second = ExperimentEvent(spec, status="completed", created_at="2026-02-01T00:00:00+00:00")

    assert first.to_dict()["fingerprint"] == second.to_dict()["fingerprint"]
    assert first.to_dict()["event_hash"] != second.to_dict()["event_hash"]


def test_fingerprint_is_mapping_order_independent():
    left = _experiment(parameters={"a": 1, "b": {"x": 2, "y": 3}})
    right = _experiment(parameters={"b": {"y": 3, "x": 2}, "a": 1})

    assert left.fingerprint == right.fingerprint


def test_search_breadth_is_part_of_experiment_identity():
    assert _experiment(declared_trial_count=12).fingerprint != _experiment(
        declared_trial_count=13
    ).fingerprint


def test_fingerprint_rejects_unstable_object_repr():
    class BadParameter:
        pass

    with pytest.raises(TypeError, match="non-deterministic fingerprint"):
        _experiment(parameters={"bad": BadParameter()})


def test_fingerprint_rejects_non_finite_parameters():
    with pytest.raises(ValueError, match="non-finite"):
        _experiment(parameters={"alpha": float("nan")})


def test_ledger_counts_attempts_and_declared_search_multiplicity(tmp_path):
    ledger = ExperimentLedger(tmp_path / "trials.jsonl")
    same = _experiment(declared_trial_count=12)
    other = _experiment(
        candidate_id="pair_only",
        parameters={"pairs": 8},
        declared_trial_count=7,
    )

    ledger.append(ExperimentEvent(same, created_at="2026-01-01T00:00:00+00:00"))
    ledger.append(ExperimentEvent(same, created_at="2026-01-02T00:00:00+00:00"))
    ledger.append(ExperimentEvent(other, created_at="2026-01-03T00:00:00+00:00"))

    assert ledger.attempt_count("nonlinear_factor_v1") == 3
    assert ledger.multiple_testing_trial_count("nonlinear_factor_v1") == 31
    assert ledger.unique_fingerprint_count("nonlinear_factor_v1") == 2
    ledger.verify()


def test_ledger_integrity_check_rejects_mutation(tmp_path):
    path = tmp_path / "trials.jsonl"
    ledger = ExperimentLedger(path)
    ledger.append(ExperimentEvent(_experiment()))
    row = json.loads(path.read_text(encoding="utf-8"))
    row["candidate_id"] = "edited-after-the-fact"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity failure"):
        ledger.verify()


def test_final_holdout_is_one_shot_even_for_same_candidate(tmp_path):
    ledger = FinalHoldoutLedger(tmp_path / "seals")
    spec = _holdout()
    candidate = _experiment().fingerprint

    first = ledger.consume(spec, candidate_fingerprint=candidate, git_hash="abc123")
    assert first["candidate_fingerprint"] == candidate

    with pytest.raises(RuntimeError, match="already been consumed"):
        ledger.consume(spec, candidate_fingerprint=candidate, git_hash="abc123")


def test_final_holdout_cannot_be_reused_by_a_different_candidate(tmp_path):
    ledger = FinalHoldoutLedger(tmp_path / "seals")
    spec = _holdout()
    ledger.consume(spec, candidate_fingerprint=_experiment().fingerprint, git_hash="abc123")

    challenger = _experiment(candidate_id="nn", parameters={"layers": 3}).fingerprint
    with pytest.raises(RuntimeError, match="already been consumed"):
        ledger.consume(spec, candidate_fingerprint=challenger, git_hash="def456")


def test_stage4_contract_rejects_search_overlapping_final_holdout():
    experiment = _experiment(search_window=("2024-01-01", "2026-01-15"))
    with pytest.raises(ValueError, match="strictly before"):
        _validate_stage4_contract(experiment, _holdout(), _stage4_config())


def test_stage4_contract_rejects_dataset_mismatch():
    with pytest.raises(ValueError, match="dataset_hash"):
        _validate_stage4_contract(
            _experiment(),
            _holdout(dataset_hash="other-data"),
            _stage4_config(),
        )


def test_stage4_contract_rejects_multiple_expanding_holdout_folds():
    with pytest.raises(ValueError, match="exactly one contiguous"):
        _validate_stage4_contract(
            _experiment(),
            _holdout(),
            ComparisonConfig(
                label_column="forward_executable_return_5d",
                horizon_days=5,
                n_folds=5,
                holdout_folds=2,
            ),
        )


def test_stage4_contract_rejects_train_search_overlap():
    with pytest.raises(ValueError, match="train_window"):
        _validate_stage4_contract(
            _experiment(train_window=("2018-01-01", "2024-02-01")),
            _holdout(),
            _stage4_config(),
        )


def test_cumulative_trial_count_never_reduces_existing_count():
    config = _stage4_config()
    report = ComparisonReport(
        config=config,
        arms=[],
        incremental=[],
        champion="",
        verdict="data_invalid",
        verdict_reasons=(),
        n_trials=6,
        pbo=np.nan,
        dsr_probability=np.nan,
        fold_windows=[],
        generated_at="2026-01-01T00:00:00+00:00",
    )

    assert with_cumulative_trial_count(report, 3).n_trials == 6
    assert with_cumulative_trial_count(report, 19).n_trials == 19


def _comparison_for_holdout(
    *,
    winner_ic=0.04,
    base_ic=0.02,
    winner_net=0.18,
    base_net=0.10,
):
    config = _stage4_config()
    baseline = ArmResult(
        name=LINEAR_BASELINE,
        model_class="rank_weighted_additive",
        feature_summary={},
        status="measured",
        holdout_metrics={"rank_ic_mean": base_ic, "net_annual_return": base_net},
    )
    winner = ArmResult(
        name="gbm",
        model_class="nonlinear_learner",
        feature_summary={},
        status="measured",
        holdout_metrics={"rank_ic_mean": winner_ic, "net_annual_return": winner_net},
    )
    return ComparisonReport(
        config=config,
        arms=[baseline, winner],
        incremental=[],
        champion="gbm",
        verdict="production_accepted",
        verdict_reasons=(),
        n_trials=20,
        pbo=0.1,
        dsr_probability=0.99,
        fold_windows=[
            {
                "foldId": "3",
                "role": "holdout",
                "trainStart": "2018-01-01",
                "trainEnd": "2025-12-31",
                "validStart": "2026-01-01",
                "validEnd": "2026-04-30",
            }
        ],
        generated_at="2026-05-01T00:00:00+00:00",
    )


def test_final_holdout_accepts_only_frozen_champion_that_generalises():
    result = _qualify_final_holdout(_comparison_for_holdout(), expected=_holdout())

    assert result.accepted
    assert result.ic_delta == pytest.approx(0.02)
    assert result.net_return_delta == pytest.approx(0.08)


def test_final_holdout_vetoes_economic_collapse():
    result = _qualify_final_holdout(
        _comparison_for_holdout(winner_net=0.05, base_net=0.10),
        expected=_holdout(),
    )

    assert not result.accepted
    assert any("after-cost" in reason for reason in result.reasons)


def test_final_holdout_vetoes_prediction_sign_failure():
    result = _qualify_final_holdout(
        _comparison_for_holdout(winner_ic=-0.01, base_ic=-0.02),
        expected=_holdout(),
    )

    assert not result.accepted
    assert any("not positive" in reason for reason in result.reasons)


def test_final_holdout_window_must_match_preregistered_window():
    result = _qualify_final_holdout(
        _comparison_for_holdout(),
        expected=_holdout(holdout_window=("2026-01-01", "2026-05-31")),
    )

    assert not result.accepted
    assert any("does not match" in reason for reason in result.reasons)
