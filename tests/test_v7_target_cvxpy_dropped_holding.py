from __future__ import annotations

import pandas as pd
import pytest

from quantagent.portfolio.v7_target_weights import (
    V7TargetWeightsConfig,
    build_v7_target_weights,
)


def test_explicit_cvxpy_keeps_partial_exit_feasible_for_dropped_holding() -> None:
    pytest.importorskip("cvxpy")
    trade_date = pd.Timestamp("2026-08-11")
    predictions = pd.DataFrame(
        [
            {"trade_date": trade_date, "symbol": "B", "prediction": 0.20},
            {"trade_date": trade_date, "symbol": "C", "prediction": 0.10},
        ]
    )
    market = pd.DataFrame(
        [
            {"trade_date": trade_date, "symbol": "B", "close": 10.0, "amount": 100_000_000.0},
            {"trade_date": trade_date, "symbol": "C", "close": 20.0, "amount": 100_000_000.0},
        ]
    )
    initial = pd.Series({"A": 0.60})
    result = build_v7_target_weights(
        predictions,
        market,
        config=V7TargetWeightsConfig(
            optimizer_backend="cvxpy",
            selection_mode="top_k",
            top_k=1,
            top_k_ratio=None,
            max_weight_per_name=1.0,
            max_sector_weight=1.0,
            max_turnover=0.40,
            liquidity_participation=1.0,
            min_selection_pressure=1.0,
            weighting="equal",
        ),
        initial_weights=initial,
    )
    assert not result.target_weights.empty
    row = result.target_weights.drop(columns=["trade_date"]).iloc[-1]
    actual = row.reindex(row.index.union(initial.index)).fillna(0.0)
    start = initial.reindex(actual.index).fillna(0.0)
    turnover = float((actual - start).abs().sum())
    assert turnover <= 0.4000001
    assert float(actual["A"]) > 0.0
    assert float(actual.get("B", 0.0)) > 0.0
