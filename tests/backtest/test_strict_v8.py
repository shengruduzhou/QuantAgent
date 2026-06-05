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


def test_sparse_nav_annualization_uses_elapsed_calendar_time():
    from quantagent.backtest.strict_v8 import _compute_metrics

    nav = pd.Series(
        [1_000_000.0, 2_000_000.0],
        index=[pd.Timestamp("2023-01-03"), pd.Timestamp("2024-01-03")],
        name="nav",
    )

    metrics = _compute_metrics(nav, pd.DataFrame())

    assert 0.95 < metrics.annualized_return < 1.05


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
        for col in ("symbol", "n_trades", "n_fills", "gross_pnl", "cost", "net_pnl", "pnl_proxy"):
            assert col in result.profit_by_stock.columns


def test_profit_by_stock_uses_realized_trade_net_pnl():
    from quantagent.backtest.strict_v8 import _profit_by_stock

    realized = pd.DataFrame([
        {"symbol": "A", "quantity": 100, "gross_pnl": 50.0, "cost": 3.0, "net_pnl": 47.0},
        {"symbol": "A", "quantity": 100, "gross_pnl": -10.0, "cost": 3.0, "net_pnl": -13.0},
        {"symbol": "B", "quantity": 100, "gross_pnl": 20.0, "cost": 2.0, "net_pnl": 18.0},
    ])
    audit = pd.DataFrame([
        {"trade_date": pd.Timestamp("2024-03-01"), "symbol": "A", "side": "buy",
         "filled_quantity": 100, "avg_price": 10.0},
        {"trade_date": pd.Timestamp("2024-03-02"), "symbol": "A", "side": "sell",
         "filled_quantity": 100, "avg_price": 10.5},
        {"trade_date": pd.Timestamp("2024-03-01"), "symbol": "B", "side": "buy",
         "filled_quantity": 100, "avg_price": 20.0},
    ])
    out = _profit_by_stock(realized, audit)
    row_a = out[out["symbol"] == "A"].iloc[0]
    assert row_a["net_pnl"] == 34.0
    assert row_a["pnl_proxy"] == 34.0
    assert row_a["win_rate"] == 0.5


def test_realized_round_trip_pnl_matches_known_trade():
    """A clean buy-low / sell-high round trip yields the expected net PnL."""
    from quantagent.backtest.strict_v8 import _realized_round_trip_pnl
    from quantagent.execution.cost_model import AShareCostModel

    audit = pd.DataFrame([
        {"trade_date": pd.Timestamp("2024-03-01"), "symbol": "600000.SH",
         "side": "buy", "filled_quantity": 1000, "avg_price": 10.0},
        {"trade_date": pd.Timestamp("2024-03-05"), "symbol": "600000.SH",
         "side": "sell", "filled_quantity": 1000, "avg_price": 11.0},
    ])
    rt = _realized_round_trip_pnl(audit)
    assert len(rt) == 1
    row = rt.iloc[0]
    assert row["quantity"] == 1000
    # gross = 1000 * (11 - 10) = 1000
    assert abs(row["gross_pnl"] - 1000.0) < 1e-6
    # cost = buy fees + sell fees from the engine's own model (>0, includes stamp)
    cm = AShareCostModel()
    from quantagent.execution.broker_base import OrderSide
    expected_cost = (cm.calculate(OrderSide("buy"), 1000, 10.0)["total"]
                     + cm.calculate(OrderSide("sell"), 1000, 11.0)["total"])
    assert abs(row["cost"] - expected_cost) < 1e-6
    assert abs(row["net_pnl"] - (1000.0 - expected_cost)) < 1e-6


def test_realized_pnl_win_rate_partial_fifo():
    """FIFO matching across multiple buy lots and a losing trade."""
    from quantagent.backtest.strict_v8 import _realized_round_trip_pnl
    audit = pd.DataFrame([
        {"trade_date": pd.Timestamp("2024-03-01"), "symbol": "X", "side": "buy",
         "filled_quantity": 500, "avg_price": 10.0},
        {"trade_date": pd.Timestamp("2024-03-02"), "symbol": "X", "side": "buy",
         "filled_quantity": 500, "avg_price": 20.0},
        {"trade_date": pd.Timestamp("2024-03-03"), "symbol": "X", "side": "sell",
         "filled_quantity": 1000, "avg_price": 15.0},
    ])
    rt = _realized_round_trip_pnl(audit)
    # two closed trades: 500@10→15 (win) and 500@20→15 (loss)
    assert len(rt) == 2
    assert (rt["gross_pnl"] > 0).sum() == 1
    assert (rt["gross_pnl"] < 0).sum() == 1


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
