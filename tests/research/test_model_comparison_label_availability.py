from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.research.model_comparison import (
    ComparisonConfig,
    _prepare_comparison_panel,
    _topk_daily_returns,
)


def _config() -> ComparisonConfig:
    return ComparisonConfig(
        label_column="forward_executable_return_5d",
        horizon_days=5,
        top_k=2,
        cost_bps=0.0,
        n_folds=3,
        holdout_folds=1,
        min_symbols_per_date=3,
    )


def test_inference_panel_preserves_rows_with_unknown_future_outcome() -> None:
    panel = pd.DataFrame(
        [
            {
                "trade_date": "2026-01-05",
                "symbol": "A",
                "factor_a": 1.0,
                "factor_b": 2.0,
                "forward_executable_return_5d": np.nan,
            },
            {
                "trade_date": "2026-01-05",
                "symbol": "B",
                "factor_a": 2.0,
                "factor_b": 1.0,
                "forward_executable_return_5d": 0.02,
            },
        ]
    )

    work = _prepare_comparison_panel(panel, ("factor_a", "factor_b"), _config())

    assert list(work["symbol"]) == ["A", "B"]
    assert pd.isna(work.loc[work["symbol"] == "A", "forward_executable_return_5d"]).all()


def test_future_label_missing_does_not_backfill_a_lower_ranked_name() -> None:
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    rows: list[dict[str, object]] = []

    # Day 1 target is A/B from prediction. A's future outcome is unavailable.
    # C has a perfectly valid future outcome but must NOT replace A merely
    # because C's label happens to exist.
    day1 = {
        "A": (3.0, np.nan),
        "B": (2.0, 0.05),
        "C": (1.0, 0.04),
    }
    # Day 2 target becomes A/C. Because the day-1 intended target A/B is carried
    # despite its censored outcome, this is a 50% replacement (B -> C), not a
    # fresh 100% build and not a C/B portfolio inferred from day-1 labels.
    day2 = {
        "A": (3.0, 0.03),
        "B": (1.0, 0.02),
        "C": (2.0, 0.04),
    }

    for date, snapshot in zip(dates, (day1, day2), strict=True):
        for symbol, (prediction, outcome) in snapshot.items():
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "prediction": prediction,
                    "forward_executable_return_5d": outcome,
                }
            )

    net, turnover, _ = _topk_daily_returns(
        pd.DataFrame(rows),
        "forward_executable_return_5d",
        _config(),
    )

    # Day 1 has no economic observation because one selected name has no future
    # outcome. Crucially, C is not substituted to manufacture an observable PnL.
    assert list(net.index) == [dates[1]]
    assert list(turnover.index) == [dates[1]]
    assert turnover.iloc[0] == 0.5
    assert net.iloc[0] == (0.03 + 0.04) / 2.0 / 5.0
