"""Round-trip trade statistics apply corporate actions (round-29 R10-F04).

On a raw execution panel the NAV is corporate-action correct, but trade
statistics matched fills only: a 10-for-10 bonus issue was booked as a -50%
losing trade and the bonus shares as an unmatched sell.
"""
from __future__ import annotations

import pandas as pd
import pytest

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import _realized_round_trip_pnl, run_strict_backtest_v8
from quantagent.data.ashare import execution_panel as ep
from tests.data.test_execution_panel import SESSIONS, _bars, _factors


def test_bonus_issue_round_trip_is_not_a_trading_loss():
    bars = _bars("300059.SZ", [20.0, 20.0, 10.0, 10.0, 10.0, 10.0])
    factors = _factors("300059.SZ", [("2010-01-01", 1.0), (SESSIONS[2], 2.0)])
    ca = pd.DataFrame([{"symbol": "300059.SZ", "ex_date": SESSIONS[2], "cash_dividend_per_share": 0.0,
                        "stock_dividend_ratio": 0.0, "rights_ratio": 1.0}])
    panel, _ = ep.build_execution_panel(bars, session_gaps=None, factors=factors, corporate_actions=ca)
    weights = pd.DataFrame({"300059.SZ": [0.5, 0.0]}, index=SESSIONS[[0, 4]])
    result = run_strict_backtest_v8(weights, panel, config=AShareExecutionSimulationConfig(slippage_bps=0))
    metrics = result.metrics
    assert metrics.total_return == pytest.approx(0.0, abs=2e-3)
    # Only fees were lost; the halved price is the bonus issue, not a trade loss.
    assert metrics.gross_loss < 0.01 * 1_000_000 * 0.5
    assert metrics.n_trades >= 1


def _fills(rows):
    return pd.DataFrame(rows, columns=["trade_date", "symbol", "side", "filled_quantity", "avg_price", "total_cost"])


def test_dividend_and_cash_in_lieu_reduce_the_lot_cost_basis():
    fills = _fills([
        (pd.Timestamp("2024-03-04"), "600000.SH", "buy", 1000, 10.0, 0.0),
        (pd.Timestamp("2024-03-08"), "600000.SH", "sell", 1300, 7.5, 0.0),
    ])
    # 10 send 3 plus 0.5 CNY cash per share: 1000 -> 1300 shares, 500 CNY credit.
    ca = pd.DataFrame([{"trade_date": pd.Timestamp("2024-03-06"), "symbol": "600000.SH", "shares_before": 1000,
                        "cash_per_share": 0.5, "share_ratio": 0.3, "bonus_shares": 300, "cash_credit": 500.0,
                        "close": 7.3, "basis": "ex_date"}])
    trades = _realized_round_trip_pnl(fills, corporate_actions=ca)
    assert trades["quantity"].sum() == pytest.approx(1300)
    # cost 10,000 - credit 500 = 9,500 for 1,300 shares; sold for 9,750 -> +250
    assert trades["gross_pnl"].sum() == pytest.approx(250.0)


def test_delisting_write_off_closes_lots_at_zero_and_sale_tax_is_charged():
    fills = _fills([
        (pd.Timestamp("2024-03-04"), "000001.SZ", "buy", 100, 10.0, 0.0),
        (pd.Timestamp("2024-03-05"), "000001.SZ", "sell", 100, 11.0, 0.0),
        (pd.Timestamp("2024-03-04"), "000002.SZ", "buy", 200, 5.0, 0.0),
    ])
    ca = pd.DataFrame([
        {"trade_date": pd.Timestamp("2024-03-05"), "symbol": "000001.SZ", "shares_before": 100, "cash_per_share": 0.0,
         "share_ratio": 0.0, "bonus_shares": 0, "cash_credit": -12.0, "close": 11.0, "basis": "dividend_tax"},
        {"trade_date": pd.Timestamp("2024-03-07"), "symbol": "000002.SZ", "shares_before": 200, "cash_per_share": 0.0,
         "share_ratio": -1.0, "bonus_shares": -200, "cash_credit": 0.0, "close": 0.8, "basis": "delisting_writeoff",
         "written_off_value": 160.0},
    ])
    trades = _realized_round_trip_pnl(fills, corporate_actions=ca).set_index("symbol")
    assert trades.loc["000001.SZ", "cost"] == pytest.approx(12.0)
    assert trades.loc["000002.SZ", "sell_price"] == 0.0
    assert trades.loc["000002.SZ", "gross_pnl"] == pytest.approx(-1000.0)
