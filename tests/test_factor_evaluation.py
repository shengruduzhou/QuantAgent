import numpy as np
import pandas as pd

from quantagent.factors.evaluation import (
    factor_decay_curve,
    forward_return_labels,
    information_coefficient,
    quantile_group_backtest,
)


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=30)
    rows = []
    for j, symbol in enumerate(["A", "B", "C", "D", "E"]):
        for i, date in enumerate(dates):
            close = 10 + i * (0.1 + j * 0.02)
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "close": close,
                    "amount": 1e7 + j * 1e6,
                    "factor": float(j),
                    "constant": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_forward_return_labels_use_future_only_as_label():
    frame = _panel()
    labels = forward_return_labels(frame, horizons=(1,))
    first = labels[(labels["symbol"] == "A")].iloc[0]
    second_close = labels[(labels["symbol"] == "A")].iloc[1]["close"]
    assert first["forward_return_1d"] == second_close / first["close"] - 1.0
    assert np.isnan(labels[labels["symbol"] == "A"].iloc[-1]["forward_return_1d"])


def test_constant_factor_returns_nan_ic_safely():
    labels = forward_return_labels(_panel(), horizons=(1,))
    result = information_coefficient(labels, "constant", "forward_return_1d")
    assert result.ic_by_date.dropna().empty
    assert np.isnan(result.summary.mean_ic)


def test_group_backtest_and_decay_return_outputs():
    labels = forward_return_labels(_panel(), horizons=(1, 3))
    groups = quantile_group_backtest(labels, "factor", "forward_return_1d", quantiles=3)
    decay = factor_decay_curve(labels, "factor", horizons=(1, 3))
    assert not groups.group_returns.empty
    assert set(decay.rank_ic.index) == {1, 3}

