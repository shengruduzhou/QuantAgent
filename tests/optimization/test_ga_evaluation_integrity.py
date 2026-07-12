"""Regression tests for GA evaluation integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.optimization.ga_weight_optimizer import (
    GAConfig,
    WalkForwardConfig,
    optimize_factor_weights_ga,
    purged_walk_forward_splits,
)


def _constant_horizon_panel(
    *,
    n_dates: int = 500,
    n_symbols: int = 12,
    forward_return: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    symbols = [f"600{i:03d}.SH" for i in range(n_symbols)]
    factor_rows: list[dict] = []
    label_rows: list[dict] = []
    for date_index, trade_date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            factor_rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "factor": float(symbol_index + date_index * 1e-6),
                }
            )
            label_rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "forward_return": forward_return,
                }
            )
    return pd.DataFrame(factor_rows), pd.DataFrame(label_rows)


def test_walk_forward_gap_covers_label_horizon() -> None:
    dates = pd.bdate_range("2022-01-03", periods=400).tolist()
    config = WalkForwardConfig(
        n_folds=2,
        embargo_days=5,
        label_horizon_days=60,
        min_train_days=100,
        min_test_days=80,
    )
    splits = purged_walk_forward_splits(dates, config)
    assert splits
    ordered = pd.DatetimeIndex(dates)
    for _, train_end, test_start, _ in splits:
        train_index = int(ordered.get_loc(train_end))
        test_index = int(ordered.get_loc(test_start))
        assert test_index - train_index - 1 >= 60


def test_multiday_forward_returns_are_not_compounded_as_daily_returns() -> None:
    factors, labels = _constant_horizon_panel()
    result = optimize_factor_weights_ga(
        factor_panel=factors,
        forward_returns=labels,
        factor_names=["factor"],
        ga_config=GAConfig(
            population_size=4,
            generations=1,
            elitism=1,
            top_k=5,
            random_seed=3,
            label_horizon_days=20,
            transaction_cost_bps=0.0,
            min_cohort_observations=2,
        ),
        wf_config=WalkForwardConfig(
            n_folds=2,
            embargo_days=5,
            label_horizon_days=20,
            min_train_days=160,
            min_test_days=100,
        ),
    )
    # Correct annualisation is roughly (1.01) ** (252/20) - 1 ~= 13.4%.
    # Treating every overlapping 20-day label as a daily return would exceed 1000%.
    assert result.fold_results
    assert all(0.05 < fold.components["net_return"] < 0.30 for fold in result.fold_results)


def test_optimizer_fails_closed_when_label_coverage_is_too_low() -> None:
    factors, labels = _constant_horizon_panel(n_dates=220)
    labels.loc[labels.groupby("trade_date").cumcount() % 3 != 0, "forward_return"] = np.nan
    with pytest.raises(ValueError, match="integrity gates"):
        optimize_factor_weights_ga(
            factor_panel=factors,
            forward_returns=labels,
            factor_names=["factor"],
            ga_config=GAConfig(
                population_size=4,
                generations=1,
                elitism=1,
                top_k=6,
                min_label_coverage=0.80,
            ),
            wf_config=WalkForwardConfig(
                n_folds=2,
                embargo_days=2,
                label_horizon_days=1,
                min_train_days=80,
                min_test_days=40,
            ),
        )


def test_duplicate_keys_fail_before_merge_multiplication() -> None:
    factors, labels = _constant_horizon_panel(n_dates=220)
    duplicated = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate trade_date/symbol"):
        optimize_factor_weights_ga(
            factor_panel=factors,
            forward_returns=duplicated,
            factor_names=["factor"],
            ga_config=GAConfig(population_size=4, generations=1, elitism=1),
            wf_config=WalkForwardConfig(
                n_folds=2,
                embargo_days=2,
                min_train_days=80,
                min_test_days=40,
            ),
        )
