from __future__ import annotations

import pandas as pd
import pytest

from quantagent.execution.broker_base import OrderSide
from quantagent.execution.intraday_fill import CostConfig, FillMode, IntradayFillSimulator, trade_cost_breakdown
from quantagent.execution.intraday_ledger import IntradayLedger


def test_intraday_ledger_never_sells_today_buy():
    ledger = IntradayLedger("000001.SZ", "2026-06-01", carried_shares=0, target_shares=0, cash=100_000)
    ledger.open_buy_low(pair_id="p1", quantity=100, price=10.0)

    assert ledger.today_bought == 100
    assert ledger.sellable_shares == 0
    assert ledger.current_position == 100
    with pytest.raises(ValueError, match="T\\+1"):
        ledger.close_buy_pair_sell_after_buy(quantity=100, price=10.2)


def test_intraday_ledger_round_trip_and_restore_event():
    ledger = IntradayLedger("000001.SZ", "2026-06-01", carried_shares=1000, target_shares=1000, cash=10_000)
    ledger.open_sell_high(pair_id="s1", quantity=200, price=10.5, cost=2.0)
    assert ledger.sellable_shares == 800
    assert ledger.current_position == 800

    event = ledger.close_sell_pair_buyback(quantity=200, price=10.0, cost=2.0)
    assert event.action == "BUY_BACK"
    assert ledger.today_bought == 200
    assert ledger.current_position == 1000
    assert ledger.realized_net_pnl > 0

    ledger.open_sell_high(pair_id="s2", quantity=100, price=10.6)
    restore = ledger.mark_eod_restore(price=10.4)
    assert restore is not None
    assert restore.event_type == "EOD_RESTORE"
    assert not ledger.open_sell_pairs


def _bars():
    return pd.DataFrame(
        [
            {"trade_time": "2026-06-01 09:30:00", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 20_000, "limit_up": 11.0, "limit_down": 9.0},
            {"trade_time": "2026-06-01 09:31:00", "open": 10.1, "high": 10.2, "low": 10.0, "close": 10.15, "volume": 10_000, "limit_up": 11.0, "limit_down": 9.0},
        ]
    )


def test_conservative_fill_uses_next_bar_and_capacity():
    sim = IntradayFillSimulator(cost_config=CostConfig(slippage_bps=5, spread_bps=5))
    fill = sim.simulate(_bars(), signal_index=0, side=OrderSide.BUY, quantity=1000, mode=FillMode.CONSERVATIVE)

    assert fill.status == "partial"
    assert fill.filled_qty == 500
    assert fill.fill_price > 10.1
    assert fill.reason == "capacity_partial"
    assert fill.costs["total"] > 0


def test_conservative_fill_rejects_near_limit_buy():
    bars = _bars()
    bars.loc[1, "open"] = 10.99
    bars.loc[1, "close"] = 10.99
    sim = IntradayFillSimulator()
    fill = sim.simulate(bars, signal_index=0, side="buy", quantity=100, mode="conservative")
    assert fill.status == "rejected"
    assert fill.reason == "near_price_limit"


def test_trade_cost_breakdown_reports_bps_components():
    cost = trade_cost_breakdown("sell", 1000, 10.0, CostConfig(slippage_bps=4, spread_bps=3))
    assert cost["commission_bps"] > 0
    assert cost["stamp_tax_bps"] > 0
    assert cost["transfer_fee_bps"] > 0
    assert cost["slippage_bps"] == 4
    assert cost["spread_cost_bps"] == 3
    assert cost["net_pnl_bps"] < 0
