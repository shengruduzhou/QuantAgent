from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.research.model_comparison import ComparisonConfig, _topk_daily_returns


def test_topk_cost_scales_with_actual_replacement_turnover() -> None:
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
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
                    "forward_return_5d": 0.05,
                }
            )

    config = ComparisonConfig(
        label_column="forward_return_5d",
        horizon_days=5,
        top_k=2,
        cost_bps=10.0,
        n_folds=3,
        holdout_folds=1,
        min_symbols_per_date=3,
    )

    net, turnover, _ = _topk_daily_returns(
        pd.DataFrame(rows),
        "forward_return_5d",
        config,
    )

    # 10 bps round research cost spread over a 5-session label = 2 bps per
    # daily-equivalent observation, multiplied by the one-way replacement rate.
    # Day 1 builds the book (100%); day 2 keeps A/B (0%); day 3 replaces B by C (50%).
    expected_turnover = np.asarray([1.0, 0.0, 0.5])
    expected_gross = 0.05 / 5.0
    expected_cost = expected_turnover * (10.0 / 10_000.0) / 5.0

    np.testing.assert_allclose(turnover.to_numpy(), expected_turnover, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        net.to_numpy(),
        expected_gross - expected_cost,
        rtol=0.0,
        atol=1e-12,
    )
    assert net.iloc[1] == expected_gross  # no trade => no modeled transaction cost
