from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.research.model_comparison import ComparisonConfig, _topk_daily_returns


def test_topk_cost_scales_with_actual_replacement_turnover() -> None:
    # Horizon 1: each day's book replaces yesterday's, so day-over-day
    # replacement is exactly the traded turnover.
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    rows: list[dict[str, object]] = []
    rankings = (
        {"A": 3.0, "B": 2.0, "C": 1.0},
        {"A": 3.0, "B": 2.0, "C": 1.0},
        {"A": 3.0, "B": 1.0, "C": 2.0},
    )
    for date, scores in zip(dates, rankings, strict=True):
        for symbol, prediction in scores.items():
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "prediction": prediction,
                    "forward_return_1d": 0.05,
                }
            )

    config = ComparisonConfig(
        label_column="forward_return_1d",
        horizon_days=1,
        top_k=2,
        cost_bps=10.0,
        n_folds=3,
        holdout_folds=1,
        min_symbols_per_date=3,
    )

    net, turnover, _ = _topk_daily_returns(
        pd.DataFrame(rows),
        "forward_return_1d",
        config,
    )

    # Day 1 builds the book (100%); day 2 keeps A/B (0%); day 3 replaces B by C (50%).
    expected_turnover = np.asarray([1.0, 0.0, 0.5])
    expected_gross = 0.05
    expected_cost = expected_turnover * (10.0 / 10_000.0)

    np.testing.assert_allclose(turnover.to_numpy(), expected_turnover, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        net.to_numpy(),
        expected_gross - expected_cost,
        rtol=0.0,
        atol=1e-12,
    )
    assert net.iloc[1] == expected_gross  # no trade => no modeled transaction cost


def test_h_day_tranche_trades_against_the_tranche_it_replaces() -> None:
    """Round-29 R3-F10: with H=2, the day-3 tranche replaces the day-1 tranche."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    rankings = (
        {"A": 3.0, "B": 2.0, "C": 1.0},  # tranche 1: A, B
        {"A": 3.0, "C": 2.0, "B": 1.0},  # tranche 2: A, C (built from cash)
        {"A": 3.0, "B": 2.0, "C": 1.0},  # replaces tranche 1 (A, B): nothing new
    )
    rows = [
        {"trade_date": date, "symbol": symbol, "prediction": prediction, "forward_return_2d": 0.02}
        for date, scores in zip(dates, rankings, strict=True)
        for symbol, prediction in scores.items()
    ]
    config = ComparisonConfig(
        label_column="forward_return_2d", horizon_days=2, top_k=2, cost_bps=10.0,
        n_folds=3, holdout_folds=1, min_symbols_per_date=3,
    )
    _, turnover, _ = _topk_daily_returns(pd.DataFrame(rows), "forward_return_2d", config)
    # Day-over-day churn would have been [1.0, 0.5, 0.5].
    np.testing.assert_allclose(turnover.to_numpy(), [1.0, 1.0, 0.0])
