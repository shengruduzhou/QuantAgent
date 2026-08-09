from __future__ import annotations

import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import METRIC_SEMANTICS_VERSION, _compute_metrics, run_strict_backtest_v8


def test_compute_metrics_includes_first_post_trade_loss_from_initial_capital() -> None:
    nav = pd.Series(
        [990_000.0],
        index=[pd.Timestamp("2026-01-05")],
        name="nav",
    )
    metrics = _compute_metrics(nav, pd.DataFrame(), initial_nav=1_000_000.0)
    assert metrics.total_return == pytest.approx(-0.01)
    assert metrics.max_drawdown == pytest.approx(0.01)


def test_compute_metrics_default_helper_semantics_remain_backward_compatible() -> None:
    nav = pd.Series(
        [1_000_000.0, 1_100_000.0],
        index=[pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")],
    )
    metrics = _compute_metrics(nav, pd.DataFrame())
    assert metrics.total_return == pytest.approx(0.10)


def test_run_strict_backtest_first_execution_day_cost_is_not_normalized_away() -> None:
    signal_date = pd.Timestamp("2026-01-05")
    execution_date = pd.Timestamp("2026-01-06")
    targets = pd.DataFrame({"600000.SH": [0.50]}, index=[signal_date])
    market = pd.DataFrame(
        [
            {
                "trade_date": signal_date,
                "symbol": "600000.SH",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 10_000_000.0,
                "amount": 100_000_000.0,
                "is_suspended": False,
                "is_st": False,
                "is_limit_up": False,
                "is_limit_down": False,
            },
            {
                "trade_date": execution_date,
                "symbol": "600000.SH",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 10_000_000.0,
                "amount": 100_000_000.0,
                "is_suspended": False,
                "is_st": False,
                "is_limit_up": False,
                "is_limit_down": False,
            },
        ]
    )
    result = run_strict_backtest_v8(
        targets,
        market,
        config=AShareExecutionSimulationConfig(
            initial_cash=1_000_000.0,
            slippage_bps=8.0,
        ),
    )
    assert result.metrics.total_return < 0.0
    assert result.daily_pnl.loc[0, "daily_return"] < 0.0
    assert result.daily_pnl.loc[0, "trade_date"] == execution_date
    assert result.config["metric_semantics_version"] == METRIC_SEMANTICS_VERSION
    payload = result.metrics.to_dict()
    assert payload["metric_semantics_version"] == METRIC_SEMANTICS_VERSION
    assert payload["nav_baseline"] == "configured_initial_cash"


def test_initial_nav_must_be_positive_and_finite() -> None:
    nav = pd.Series([1.0], index=[pd.Timestamp("2026-01-05")])
    with pytest.raises(ValueError, match="initial_nav"):
        _compute_metrics(nav, pd.DataFrame(), initial_nav=0.0)
