"""StrictBacktestV8 tests — full output bundle (spec section 9)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quantagent.backtest.strict_v8 import (
    StrictBacktestArtifactSet,
    StrictBacktestMetrics,
    run_strict_backtest_v8,
)
from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
)


def _market_panel(n_days: int = 8, prices: dict[str, float] | None = None) -> pd.DataFrame:
    prices = prices or {"600000.SH": 10.0, "000001.SZ": 12.0}
    dates = pd.bdate_range("2024-03-01", periods=n_days)
    rows = []
    for d in dates:
        for sym, price in prices.items():
            rows.append({
                "trade_date": d, "symbol": sym, "close": price,
                "volume": 1_000_000.0, "amount": 10_000_000.0,
                "is_suspended": False, "is_st": False,
                "is_limit_up": False, "is_limit_down": False,
            })
    return pd.DataFrame(rows)


def _target_weights(n_days: int = 8, weight: float = 0.02) -> pd.DataFrame:
    return pd.DataFrame(
        {"600000.SH": [weight] * n_days, "000001.SZ": [weight] * n_days},
        index=pd.bdate_range("2024-03-01", periods=n_days),
    )


def test_returns_artifact_set_with_all_attributes():
    result = run_strict_backtest_v8(
        _target_weights(), _market_panel(),
        config=AShareExecutionSimulationConfig(slippage_bps=0),
    )
    assert isinstance(result, StrictBacktestArtifactSet)
    assert isinstance(result.metrics, StrictBacktestMetrics)
    assert isinstance(result.nav, pd.Series)
    assert isinstance(result.daily_pnl, pd.DataFrame)
    assert isinstance(result.selected_stocks, pd.DataFrame)
    assert isinstance(result.trades, pd.DataFrame)
    assert isinstance(result.failed_orders, pd.DataFrame)
    assert isinstance(result.risk_events, list)


def test_metrics_has_all_nine_spec_fields():
    result = run_strict_backtest_v8(
        _target_weights(), _market_panel(),
        config=AShareExecutionSimulationConfig(slippage_bps=0),
    )
    d = result.metrics.to_dict()
    for k in (
        "total_return",
        "annualized_return",
        "max_drawdown",
        "sharpe",
        "calmar",
        "volatility",
        "turnover",
        "win_rate",
        "avg_profit_per_trade",
    ):
        assert k in d, f"metric {k} missing"


def test_write_emits_every_spec_file(tmp_path):
    sector_map = pd.DataFrame([
        {"symbol": "600000.SH", "sector_level_1": "Bank"},
        {"symbol": "000001.SZ", "sector_level_1": "Bank"},
    ])
    result = run_strict_backtest_v8(
        _target_weights(), _market_panel(),
        sector_map=sector_map,
        factor_weights={"factor_a": 0.6, "factor_b": 0.4},
        config=AShareExecutionSimulationConfig(slippage_bps=0),
    )
    paths = result.write(tmp_path)
    expected = (
        "metrics", "nav", "pnl",
        "selected_stocks", "trades", "failed_orders",
        "risk_events", "profit_by_stock", "profit_by_sector",
        "factor_weights",
    )
    for k in expected:
        assert paths[k].exists(), f"{k} not written"
    # factor_weights JSON content sanity
    fw = json.loads(paths["factor_weights"].read_text(encoding="utf-8"))
    assert fw == {"factor_a": 0.6, "factor_b": 0.4}


def test_profit_by_sector_aggregates_when_sector_map_provided():
    sector_map = pd.DataFrame([
        {"symbol": "600000.SH", "sector_level_1": "Bank"},
        {"symbol": "000001.SZ", "sector_level_1": "Bank"},
    ])
    result = run_strict_backtest_v8(
        _target_weights(), _market_panel(),
        sector_map=sector_map,
        config=AShareExecutionSimulationConfig(slippage_bps=0),
    )
    if not result.profit_by_sector.empty:
        assert "Bank" in set(result.profit_by_sector["sector_level_1"])


def test_profit_by_stock_top_lists_symbols():
    result = run_strict_backtest_v8(
        _target_weights(), _market_panel(),
        config=AShareExecutionSimulationConfig(slippage_bps=0),
    )
    if not result.profit_by_stock.empty:
        for col in ("symbol", "n_fills", "gross_value", "pnl_proxy"):
            assert col in result.profit_by_stock.columns


def test_empty_inputs_produce_empty_but_well_shaped_bundle(tmp_path):
    empty_panel = pd.DataFrame(columns=["trade_date", "symbol", "close", "volume",
                                         "amount", "is_suspended", "is_st",
                                         "is_limit_up", "is_limit_down"])
    empty_targets = pd.DataFrame()
    result = run_strict_backtest_v8(empty_targets, empty_panel)
    assert result.metrics.n_trades == 0
    paths = result.write(tmp_path)
    # Files exist even when empty
    assert paths["metrics"].exists()
    assert paths["trades"].exists()
