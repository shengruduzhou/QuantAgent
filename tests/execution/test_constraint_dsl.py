"""ExecutionConstraintDSL tests (spec section 8)."""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.execution.constraints import (
    AuctionPhase,
    ExecutionConstraintEvaluator,
    ExecutionConstraintSet,
    OrderIntentRecord,
    classify_auction_phase,
)


def _intent(
    intent_id: str,
    symbol: str = "600519.SH",
    side: str = "buy",
    quantity: int = 100,
    price: float = 100.0,
    timestamp: pd.Timestamp | None = None,
    parent: str | None = None,
    nav: float | None = 1_000_000.0,
    daily_volume: float | None = None,
    order_value: float = 0.0,
) -> OrderIntentRecord:
    ts = timestamp if timestamp is not None else pd.Timestamp("2024-03-01 10:30:00")
    return OrderIntentRecord(
        intent_id=intent_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        timestamp=ts,
        parent_intent_id=parent,
        portfolio_nav=nav,
        daily_volume_hint=daily_volume,
        order_value=order_value,
    )


# ---------------------------------------------------------------------------
# Auction phase
# ---------------------------------------------------------------------------

def test_classify_auction_phase_each_band():
    assert classify_auction_phase(pd.Timestamp("2024-03-01 09:20:00")) == AuctionPhase.PRE_AUCTION_OPEN
    assert classify_auction_phase(pd.Timestamp("2024-03-01 09:27:00")) == AuctionPhase.AUCTION_OPEN
    assert classify_auction_phase(pd.Timestamp("2024-03-01 10:30:00")) == AuctionPhase.CONTINUOUS
    assert classify_auction_phase(pd.Timestamp("2024-03-01 14:58:00")) == AuctionPhase.PRE_AUCTION_CLOSE
    assert classify_auction_phase(pd.Timestamp("2024-03-01 15:00:00")) == AuctionPhase.AUCTION_CLOSE
    assert classify_auction_phase(pd.Timestamp("2024-03-01 08:00:00")) == AuctionPhase.CLOSED


# ---------------------------------------------------------------------------
# Posture + global limits
# ---------------------------------------------------------------------------

def test_empty_intent_stream_passes():
    rep = ExecutionConstraintEvaluator().evaluate([])
    assert rep.passed
    assert rep.n_intents == 0


def test_default_set_passes_a_sane_intent_stream():
    intents = [
        _intent(f"i{i}", quantity=200, price=50.0, timestamp=pd.Timestamp(f"2024-03-01 10:{i:02d}:00"))
        for i in range(5)
    ]
    rep = ExecutionConstraintEvaluator().evaluate(intents)
    assert rep.passed
    assert rep.n_violations == 0


def test_live_trading_with_dry_run_required_blocks():
    constraints = ExecutionConstraintSet(
        qmt_dry_run_required_by_default=True, live_trading_enabled=True
    )
    rep = ExecutionConstraintEvaluator(constraints).evaluate([_intent("a")])
    assert not rep.passed
    names = {v.constraint for v in rep.violations}
    assert "qmt_dry_run_required_by_default" in names


def test_max_orders_per_day_block():
    constraints = ExecutionConstraintSet(max_orders_per_day=3)
    intents = [_intent(f"i{i}") for i in range(5)]
    rep = ExecutionConstraintEvaluator(constraints).evaluate(intents)
    assert not rep.passed
    assert any(v.constraint == "max_orders_per_day" for v in rep.violations)


def test_max_orders_per_second_block():
    constraints = ExecutionConstraintSet(max_orders_per_second=2)
    ts = pd.Timestamp("2024-03-01 10:30:00")
    intents = [_intent(f"i{i}", timestamp=ts) for i in range(5)]
    rep = ExecutionConstraintEvaluator(constraints).evaluate(intents)
    assert not rep.passed
    assert any(v.constraint == "max_orders_per_second" for v in rep.violations)


def test_max_single_order_value_block():
    constraints = ExecutionConstraintSet(max_single_order_value=1_000.0)
    rep = ExecutionConstraintEvaluator(constraints).evaluate(
        [_intent("big", quantity=1000, price=10.0)]
    )
    assert not rep.passed
    assert any(v.constraint == "max_single_order_value" for v in rep.violations)


def test_max_cancel_ratio_block():
    constraints = ExecutionConstraintSet(max_cancel_ratio=0.20)
    intents = [_intent("i1"), _intent("i2")]
    cancels = [_intent(f"c{i}", side="cancel", parent="i1") for i in range(3)]
    rep = ExecutionConstraintEvaluator(constraints).evaluate(intents + cancels)
    assert not rep.passed
    assert any(v.constraint == "max_cancel_ratio" for v in rep.violations)


def test_max_daily_turnover_block():
    constraints = ExecutionConstraintSet(max_daily_turnover=0.10)
    nav = 1_000_000.0
    # Two orders, each 60k value → 0.12 turnover > 0.10
    intents = [
        _intent("i1", quantity=600, price=100.0, nav=nav),
        _intent("i2", quantity=600, price=100.0, nav=nav,
                timestamp=pd.Timestamp("2024-03-01 10:31:00")),
    ]
    rep = ExecutionConstraintEvaluator(constraints).evaluate(intents)
    assert not rep.passed
    assert any(v.constraint == "max_daily_turnover" for v in rep.violations)


# ---------------------------------------------------------------------------
# Participation rate
# ---------------------------------------------------------------------------

def test_participation_rate_block():
    constraints = ExecutionConstraintSet(max_single_stock_participation_rate=0.05)
    intents = [
        _intent("i1", quantity=600, daily_volume=10_000),
        _intent("i2", quantity=200, daily_volume=10_000,
                timestamp=pd.Timestamp("2024-03-01 10:31:00")),
    ]
    rep = ExecutionConstraintEvaluator(constraints).evaluate(intents)
    assert not rep.passed
    assert any(v.constraint == "max_single_stock_participation_rate" for v in rep.violations)


def test_participation_rate_skipped_without_volume_hint():
    constraints = ExecutionConstraintSet(max_single_stock_participation_rate=0.01)
    intents = [_intent("i1", quantity=100_000, daily_volume=None)]
    rep = ExecutionConstraintEvaluator(constraints).evaluate(intents)
    # No volume hint → skipped, not flagged
    names = {v.constraint for v in rep.violations}
    assert "max_single_stock_participation_rate" not in names


# ---------------------------------------------------------------------------
# Auction-mode + resting time
# ---------------------------------------------------------------------------

def test_auction_mode_max_orders_per_symbol_block():
    constraints = ExecutionConstraintSet(auction_mode_max_orders_per_symbol=1)
    base = pd.Timestamp("2024-03-01 09:18:00")
    intents = [
        _intent(f"a{i}", timestamp=base + pd.Timedelta(seconds=i * 10))
        for i in range(3)
    ]
    rep = ExecutionConstraintEvaluator(constraints).evaluate(intents)
    assert not rep.passed
    assert any(v.constraint == "auction_mode_max_orders_per_symbol" for v in rep.violations)


def test_min_order_resting_time_block():
    constraints = ExecutionConstraintSet(min_order_resting_time_seconds=5.0,
                                          auction_mode_min_resting_time_seconds=None)
    submit_ts = pd.Timestamp("2024-03-01 10:30:00")
    cancel_ts = submit_ts + pd.Timedelta(seconds=1)
    submit = _intent("s1", timestamp=submit_ts)
    cancel = _intent("c1", side="cancel", parent="s1", timestamp=cancel_ts)
    rep = ExecutionConstraintEvaluator(constraints).evaluate([submit, cancel])
    assert not rep.passed
    assert any(v.constraint == "min_order_resting_time_seconds" for v in rep.violations)


def test_auction_resting_time_is_tighter():
    constraints = ExecutionConstraintSet(
        min_order_resting_time_seconds=1.0,
        auction_mode_min_resting_time_seconds=30.0,
    )
    submit_ts = pd.Timestamp("2024-03-01 09:18:00")  # auction phase
    cancel_ts = submit_ts + pd.Timedelta(seconds=5)
    submit = _intent("s1", timestamp=submit_ts)
    cancel = _intent("c1", side="cancel", parent="s1", timestamp=cancel_ts)
    rep = ExecutionConstraintEvaluator(constraints).evaluate([submit, cancel])
    assert not rep.passed
    assert any(v.constraint == "min_order_resting_time_seconds" for v in rep.violations)


# ---------------------------------------------------------------------------
# Spoof / layer / pull-push heuristics
# ---------------------------------------------------------------------------

def test_spoofing_heuristic_block():
    constraints = ExecutionConstraintSet(no_spoofing=True, spoof_max_repeated_cancels=3,
                                          # disable resting-time so cancel block is from spoofing
                                          min_order_resting_time_seconds=None,
                                          auction_mode_min_resting_time_seconds=None,
                                          max_cancel_ratio=None)
    base = pd.Timestamp("2024-03-01 10:30:00")
    submits = [_intent(f"s{i}", timestamp=base + pd.Timedelta(seconds=i)) for i in range(3)]
    cancels = [
        _intent(f"c{i}", side="cancel", parent=f"s{i}",
                timestamp=base + pd.Timedelta(seconds=i + 1))
        for i in range(3)
    ]
    rep = ExecutionConstraintEvaluator(constraints).evaluate(submits + cancels)
    assert any(v.constraint == "no_spoofing" for v in rep.violations)


def test_layering_heuristic_block():
    constraints = ExecutionConstraintSet(no_layering=True, layering_max_concurrent_levels=3,
                                          max_orders_per_second=None)
    ts = pd.Timestamp("2024-03-01 10:30:00")
    intents = [
        _intent(f"L{i}", price=100.0 + i, timestamp=ts) for i in range(5)
    ]
    rep = ExecutionConstraintEvaluator(constraints).evaluate(intents)
    assert any(v.constraint == "no_layering" for v in rep.violations)


def test_pull_push_size_jump_block():
    constraints = ExecutionConstraintSet(no_pull_push=True, pull_push_min_size_jump=3.0,
                                          max_orders_per_second=None,
                                          max_single_stock_participation_rate=None)
    base = pd.Timestamp("2024-03-01 10:30:00")
    intents = [
        _intent("p1", quantity=100, timestamp=base),
        _intent("p2", quantity=500, timestamp=base + pd.Timedelta(seconds=10)),
    ]
    rep = ExecutionConstraintEvaluator(constraints).evaluate(intents)
    assert any(v.constraint == "no_pull_push" for v in rep.violations)


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------

def test_report_to_dict_serialisable_under_violations():
    import json
    constraints = ExecutionConstraintSet(max_orders_per_day=1)
    rep = ExecutionConstraintEvaluator(constraints).evaluate([_intent("i1"), _intent("i2")])
    payload = rep.to_dict()
    json.dumps(payload, default=str)
    assert payload["passed"] is False
    assert payload["by_constraint"]["max_orders_per_day"] == 1


def test_constraint_set_as_dict_has_all_fields():
    d = ExecutionConstraintSet().as_dict()
    for k in (
        "max_orders_per_second",
        "max_orders_per_day",
        "max_cancel_ratio",
        "min_order_resting_time_seconds",
        "max_single_stock_participation_rate",
        "max_single_order_value",
        "max_daily_turnover",
        "auction_mode_max_orders_per_symbol",
        "no_spoofing",
        "no_layering",
        "no_pull_push",
        "qmt_dry_run_required_by_default",
        "live_trading_enabled",
    ):
        assert k in d
