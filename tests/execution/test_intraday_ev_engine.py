from __future__ import annotations

from quantagent.execution.intraday_ev_engine import EVDecisionConfig, IntradayModelSignals, decide_ev
from quantagent.execution.intraday_fill import CostConfig
from quantagent.execution.intraday_ledger import IntradayLedger


def _state(**overrides):
    base = {
        "close": 10.0,
        "last": 10.0,
        "rolling_volatility_20m": 0.002,
        "estimated_spread_bps": 4.0,
        "volume_capacity_ratio": 0.0,
        "one_way_trend_probability": 0.1,
        "mean_reversion_probability": 0.7,
        "rolling_return_20m": 0.0,
        "limit_up_distance": 0.08,
        "limit_down_distance": 0.08,
        "near_limit_risk": False,
        "minutes_to_close": 120,
    }
    base.update(overrides)
    return base


def _cfg():
    return EVDecisionConfig(
        cost=CostConfig(commission_rate=0.0001, min_commission=0.0, stamp_tax_sell=0.0002, transfer_fee=0.0, slippage_bps=2.0, spread_bps=2.0),
        absolute_min_edge_bps=5.0,
    )


def test_ev_decision_defaults_to_no_trade_when_edge_missing():
    ledger = IntradayLedger("000001.SZ", "2026-06-01", carried_shares=1000, target_shares=1000, cash=100_000)
    decision = decide_ev(_state(), ledger, IntradayModelSignals(), _cfg())
    assert decision.action == "NO_TRADE"


def test_ev_decision_allows_sell_high_only_with_sellable_inventory():
    signals = IntradayModelSignals(
        p_sell_high_success=0.9,
        expected_sell_high_gain_bps=160.0,
        p_fail_new_high=0.05,
        expected_chase_loss_bps=30.0,
    )
    cfg = _cfg()
    ledger = IntradayLedger("000001.SZ", "2026-06-01", carried_shares=1000, target_shares=1000, cash=100_000)
    decision = decide_ev(_state(), ledger, signals, cfg)
    assert decision.action == "SELL_HIGH"
    assert decision.quantity <= ledger.sellable_shares

    no_sellable = IntradayLedger("000001.SZ", "2026-06-01", carried_shares=0, target_shares=0, cash=100_000)
    blocked = decide_ev(_state(), no_sellable, signals, cfg)
    assert blocked.action == "NO_TRADE"


def test_ev_decision_blocks_reverse_t_in_one_way_uptrend():
    signals = IntradayModelSignals(
        p_sell_high_success=0.95,
        expected_sell_high_gain_bps=300.0,
        p_fail_new_high=0.01,
        expected_chase_loss_bps=20.0,
    )
    ledger = IntradayLedger("000001.SZ", "2026-06-01", carried_shares=1000, target_shares=1000, cash=100_000)
    decision = decide_ev(_state(one_way_trend_probability=0.9, rolling_return_20m=0.03), ledger, signals, _cfg())
    assert decision.action == "NO_TRADE"


def test_ev_decision_closes_open_sell_pair_before_new_pairs():
    ledger = IntradayLedger("000001.SZ", "2026-06-01", carried_shares=1000, target_shares=1000, cash=0)
    ledger.open_sell_high(pair_id="s1", quantity=200, price=10.5)
    signals = IntradayModelSignals(p_buyback_now=0.9, expected_buyback_edge_bps=120.0, wait_extra_edge_bps=5.0)

    decision = decide_ev(_state(close=9.9, last=9.9), ledger, signals, _cfg())
    assert decision.action == "BUY_BACK"
    assert decision.quantity == 200


def test_ev_decision_blocks_new_pair_near_close():
    signals = IntradayModelSignals(p_buy_low_success=0.9, expected_buy_low_gain_bps=200.0)
    ledger = IntradayLedger("000001.SZ", "2026-06-01", carried_shares=1000, target_shares=1000, cash=100_000)
    decision = decide_ev(_state(minutes_to_close=10), ledger, signals, _cfg())
    assert decision.action == "NO_TRADE"
    assert "no_new_pair_near_close" in decision.risk_flags
