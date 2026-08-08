from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from quantagent.training.ft_transformer_trainer import (
    OBJECTIVE_SEMANTICS_VERSION,
    FTTransformerTrainer,
    validation_objective_from_predictions,
)


def test_rank_weight_zero_is_exact_pointwise_objective_and_needs_no_dates() -> None:
    pred = np.array([[0.2], [0.0], [-0.1]], dtype=float)
    target = np.array([[0.1], [0.0], [0.1]], dtype=float)
    parts = validation_objective_from_predictions(
        pred,
        target,
        pd.Series([pd.NaT, pd.NaT, pd.NaT]),
        huber_delta=1.0,
        rank_loss_weight=0.0,
        rank_loss_temperature=0.5,
    )
    expected = 0.5 * np.mean((pred - target) ** 2)
    assert parts["huber"] == pytest.approx(expected)
    assert parts["rank"] == 0.0
    assert parts["composite"] == pytest.approx(expected)
    assert parts["rank_groups"] == 0.0


def test_listwise_validation_is_separated_by_trade_date() -> None:
    pred = np.array(
        [
            [0.8], [0.1], [-0.2],
            [-0.4], [0.2], [0.9],
        ],
        dtype=float,
    )
    target = np.array(
        [
            [0.05], [0.01], [-0.03],
            [-0.04], [0.00], [0.06],
        ],
        dtype=float,
    )
    dates = pd.Series(
        ["2026-01-05"] * 3 + ["2026-01-06"] * 3,
        dtype="datetime64[ns]",
    )
    together = validation_objective_from_predictions(
        pred,
        target,
        dates,
        huber_delta=1.0,
        rank_loss_weight=1.0,
        rank_loss_temperature=0.5,
    )
    d1 = validation_objective_from_predictions(
        pred[:3], target[:3], dates.iloc[:3],
        huber_delta=1.0, rank_loss_weight=1.0, rank_loss_temperature=0.5,
    )
    d2 = validation_objective_from_predictions(
        pred[3:], target[3:], dates.iloc[3:],
        huber_delta=1.0, rank_loss_weight=1.0, rank_loss_temperature=0.5,
    )
    assert together["rank_groups"] == 2.0
    assert together["rank"] == pytest.approx((d1["rank"] + d2["rank"]) / 2.0)


def test_rank_validation_fails_closed_on_missing_or_invalid_dates() -> None:
    pred = np.array([[0.2], [0.1]], dtype=float)
    target = np.array([[0.1], [0.0]], dtype=float)
    with pytest.raises(ValueError, match="trade_date"):
        validation_objective_from_predictions(
            pred,
            target,
            pd.Series(["2026-01-05", "not-a-date"]),
            huber_delta=1.0,
            rank_loss_weight=0.5,
            rank_loss_temperature=0.5,
        )


def test_rank_validation_requires_a_real_cross_section() -> None:
    pred = np.array([[0.2], [0.1]], dtype=float)
    target = np.array([[0.1], [0.0]], dtype=float)
    with pytest.raises(ValueError, match=">=2 names"):
        validation_objective_from_predictions(
            pred,
            target,
            pd.Series(["2026-01-05", "2026-01-06"]),
            huber_delta=1.0,
            rank_loss_weight=0.5,
            rank_loss_temperature=0.5,
        )


def test_composite_objective_can_choose_differently_from_huber_only() -> None:
    # Candidate A is close pointwise but carries no useful ordering. Candidate B
    # has worse pointwise calibration but correctly concentrates score on the
    # best target. With a strong configured rank term, governance must permit
    # the composite criterion to prefer B rather than silently selecting on A's
    # Huber advantage.
    target = np.array([[0.01], [0.0], [-0.01]], dtype=float)
    dates = pd.Series(["2026-01-05"] * 3)
    a = validation_objective_from_predictions(
        np.zeros_like(target), target, dates,
        huber_delta=1.0, rank_loss_weight=10.0, rank_loss_temperature=0.05,
    )
    b = validation_objective_from_predictions(
        np.array([[0.10], [0.0], [-0.10]]), target, dates,
        huber_delta=1.0, rank_loss_weight=10.0, rank_loss_temperature=0.05,
    )
    assert a["huber"] < b["huber"]
    assert b["rank"] < a["rank"]
    assert b["composite"] < a["composite"]


def test_fit_path_wires_composite_validation_to_checkpoint_selection() -> None:
    source = inspect.getsource(FTTransformerTrainer._fit_torch)
    assert "val_dates" in source
    assert "_validation_objective(" in source
    assert 'val_parts["composite"]' in source
    assert 'entry["val_huber_loss"]' in source
    assert 'entry["val_rank_loss"]' in source
    assert 'entry["val_composite_loss"]' in source
    assert "improved = val_loss < best_val" in source


def test_objective_semantics_version_is_explicit_and_stable() -> None:
    assert OBJECTIVE_SEMANTICS_VERSION == "ft_transformer_objective_v2_per_date_listwise_validation"
