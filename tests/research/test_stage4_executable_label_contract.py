from __future__ import annotations

import pytest

from quantagent.research.experiment_governance import ExperimentSpec, FinalHoldoutSpec
from quantagent.research.governed_model_comparison import _validate_stage4_contract
from quantagent.research.model_comparison import ComparisonConfig


def _experiment() -> ExperimentSpec:
    return ExperimentSpec(
        family="nonlinear_factor_v1",
        candidate_id="gbm_plus_pair",
        parameters={"features": ["value", "momentum"]},
        dataset_hash="data-sha-001",
        train_window=("2018-01-01", "2023-12-31"),
        search_window=("2024-01-01", "2025-12-31"),
        metric="rank_ic_then_net_return",
        git_hash="abc123",
        declared_trial_count=12,
        recipe_hash="recipe-sha",
        split_id="wf-v4",
    )


def _holdout(*, label_contract_hash: str = "label-contract-sha") -> FinalHoldoutSpec:
    return FinalHoldoutSpec(
        family="nonlinear_factor_v1",
        dataset_hash="data-sha-001",
        holdout_window=("2026-01-01", "2026-04-30"),
        label_contract_hash=label_contract_hash,
    )


def _config(
    *,
    label_column: str = "forward_executable_return_5d",
    horizon_days: int = 5,
) -> ComparisonConfig:
    return ComparisonConfig(
        label_column=label_column,
        horizon_days=horizon_days,
        n_folds=4,
        holdout_folds=1,
    )


def test_stage4_accepts_horizon_matched_executable_label_contract() -> None:
    _validate_stage4_contract(_experiment(), _holdout(), _config())


def test_stage4_rejects_legacy_forward_return_label() -> None:
    with pytest.raises(ValueError, match="canonical executable next-session label"):
        _validate_stage4_contract(
            _experiment(),
            _holdout(),
            _config(label_column="forward_return_5d"),
        )


def test_stage4_rejects_executable_label_for_the_wrong_horizon() -> None:
    with pytest.raises(ValueError, match="forward_executable_return_10d"):
        _validate_stage4_contract(
            _experiment(),
            _holdout(),
            _config(label_column="forward_executable_return_5d", horizon_days=10),
        )


def test_stage4_requires_holdout_label_contract_hash() -> None:
    with pytest.raises(ValueError, match="label_contract_hash"):
        _validate_stage4_contract(
            _experiment(),
            _holdout(label_contract_hash=""),
            _config(),
        )
