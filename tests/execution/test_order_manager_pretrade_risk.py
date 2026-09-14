from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace

import pytest

from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import OrderEventType as CanonicalEventType
from quantagent.execution.broker_base import (
    Order,
    OrderSide,
    OrderState,
    OrderStatus,
    OrderType,
)
from quantagent.execution.constraints import (
    ExecutionConstraintEvaluator,
    ExecutionConstraintSet,
)
from quantagent.execution.order_manager import OrderManager


class LiveRiskBroker:
    """Minimal broker double that advertises the QMT live risk contract."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            require_risk_approval=True,
            live_trading_enabled=True,
            dry_run=False,
        )
        self.submitted: list[Order] = []

    def submit(self, order: Order) -> OrderState:
        self.submitted.append(order)
        return OrderState(order.client_order_id, "broker-1", OrderStatus.SUBMITTED, 0, 0.0)

    def cancel(self, client_order_id: str) -> OrderState:
        return OrderState(client_order_id, None, OrderStatus.CANCELLED, 0, 0.0)

    def query_order(self, client_order_id: str) -> OrderState:
        return OrderState(client_order_id, None, OrderStatus.PENDING, 0, 0.0)

    def query_positions(self):
        return []

    def query_account_value(self) -> float:
        return 100_000.0

    def on_trade(self, callback) -> None:
        return None


def _manager(tmp_path, *, max_order_value: float = 1_000.0) -> tuple[OrderManager, LiveRiskBroker]:
    broker = LiveRiskBroker()
    constraints = ExecutionConstraintSet(
        max_orders_per_second=None,
        max_orders_per_day=None,
        max_cancel_ratio=None,
        min_order_resting_time_seconds=None,
        max_single_stock_participation_rate=None,
        max_single_order_value=max_order_value,
        max_daily_turnover=None,
        auction_mode_max_orders_per_symbol=None,
        no_spoofing=False,
        no_layering=False,
        no_pull_push=False,
        qmt_dry_run_required_by_default=False,
        live_trading_enabled=True,
    )
    manager = OrderManager(
        broker=broker,
        lineage=Lineage(run_id="run-live-risk-test"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        idempotency_path=str(tmp_path / "idem.jsonl"),
        constraint_evaluator=ExecutionConstraintEvaluator(constraints),
    )
    return manager, broker


def _limit_order(client_order_id: str, quantity: int = 100, price: float = 10.0) -> Order:
    return Order(
        client_order_id=client_order_id,
        symbol="600000.SH",
        side=OrderSide.BUY,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        price=price,
        signal_id=f"sig-{client_order_id}",
        strategy_version="production-candidate",
        timestamp="2026-08-08T02:30:00+00:00",
    )


def test_blocking_constraint_is_canonical_risk_rejected_before_broker_submit(tmp_path) -> None:
    manager, broker = _manager(tmp_path, max_order_value=1_000.0)

    states = manager.submit_orders([_limit_order("too-large", quantity=200, price=10.0)])

    assert states[0].status == OrderStatus.REJECTED
    assert broker.submitted == []
    canonical = manager.book.orders()[0]
    events = manager.book.history_of(canonical.order_id)
    assert [event.event_type for event in events] == [
        CanonicalEventType.CREATED,
        CanonicalEventType.RISK_REJECTED,
    ]
    decision = events[-1].risk_decision
    assert decision is not None
    assert decision.approved is False
    assert decision.rule == "execution_constraint_dsl"
    assert "max_single_order_value" in decision.reason


def test_live_submit_gets_approved_only_after_constraint_dsl_passes(tmp_path) -> None:
    manager, broker = _manager(tmp_path, max_order_value=1_000.0)

    states = manager.submit_orders([_limit_order("valid", quantity=100, price=10.0)])

    assert states[0].status == OrderStatus.SUBMITTED
    assert len(broker.submitted) == 1
    assert broker.submitted[0].risk_check_result == "approved"
    canonical = manager.book.orders()[0]
    events = manager.book.history_of(canonical.order_id)
    event_types = [event.event_type for event in events]
    assert event_types[:3] == [
        CanonicalEventType.CREATED,
        CanonicalEventType.RISK_APPROVED,
        CanonicalEventType.SUBMITTED,
    ]
    decision = events[1].risk_decision
    assert decision is not None and decision.approved is True
    assert decision.rule == "execution_constraint_dsl"


def test_unbounded_market_order_fails_closed_on_live_path(tmp_path) -> None:
    manager, broker = _manager(tmp_path, max_order_value=1_000_000.0)
    order = Order(
        client_order_id="market",
        symbol="600000.SH",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        price=None,
        signal_id="sig-market",
        strategy_version="production-candidate",
        timestamp="2026-08-08T02:30:00+00:00",
    )

    state = manager.submit_orders([order])[0]

    assert state.status == OrderStatus.REJECTED
    assert broker.submitted == []
    canonical = manager.book.orders()[0]
    events = manager.book.history_of(canonical.order_id)
    assert events[-1].event_type == CanonicalEventType.RISK_REJECTED
    assert "bounded positive" in (events[-1].risk_decision.reason if events[-1].risk_decision else "")


@pytest.mark.parametrize("price", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_invalid_limit_never_reaches_broker(tmp_path, price) -> None:
    manager, broker = _manager(tmp_path)
    state = manager.submit_orders([_limit_order("invalid-price", price=price)])[0]
    assert state.status == OrderStatus.REJECTED
    assert broker.submitted == []
    canonical = manager.book.orders()[0]
    assert manager.book.history_of(canonical.order_id)[-1].event_type == CanonicalEventType.RISK_REJECTED


@pytest.mark.parametrize("limit", ["max_orders_per_day", "max_daily_turnover", "max_single_stock_participation_rate"])
def test_intraday_consumed_limits_survive_restart_and_reset(tmp_path, limit) -> None:
    def configured():
        manager, broker = _manager(tmp_path)
        value = {"max_orders_per_day": 1, "max_daily_turnover": 0.01,
                 "max_single_stock_participation_rate": 0.1}[limit]
        manager.constraint_evaluator = ExecutionConstraintEvaluator(
            replace(manager.constraint_evaluator.constraints, **{limit: value})
        )
        manager.daily_volume_hints = {"600000.SH": 1000.0}
        return manager, broker

    first, broker = configured()
    assert first.submit_orders([_limit_order("first")])[0].status == OrderStatus.SUBMITTED
    # Cancellation does not refund the consumed submit/turnover budget.
    original = first.book.orders()[0]
    first.book.apply(original.order_id, CanonicalEventType.CANCELLED)
    first.canonical.append(first.book.history_of(original.order_id)[-1], trade_date=original.trade_date)
    restored, restored_broker = configured()
    assert restored.submit_orders([_limit_order("after-restart")])[0].status == OrderStatus.REJECTED
    restored.reset_daily_counters()
    assert restored.submit_orders([_limit_order("after-reset")])[0].status == OrderStatus.REJECTED
    assert restored_broker.submitted == []
    tomorrow = replace(_limit_order("next-session"), timestamp="2026-08-09T02:30:00+00:00")
    assert restored.submit_orders([tomorrow])[0].status == OrderStatus.SUBMITTED


def test_legacy_submitted_order_without_risk_snapshot_blocks_recovery(tmp_path) -> None:
    first, broker = _manager(tmp_path)
    broker.config.dry_run = True
    assert first.submit_orders([_limit_order("legacy")])[0].status == OrderStatus.SUBMITTED
    restored, live_broker = _manager(tmp_path)
    state = restored.submit_orders([_limit_order("new")])[0]
    assert state.status == OrderStatus.REJECTED
    assert "recovery-required" in state.last_message
    assert live_broker.submitted == []


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("missing", ["nav", "volume", "both"])
def test_current_measurements_cannot_be_replaced_by_history(tmp_path, restart, missing) -> None:
    manager, broker = _manager(tmp_path)
    constraints = replace(manager.constraint_evaluator.constraints,
                          max_daily_turnover=2.0, max_single_stock_participation_rate=0.1)
    manager.constraint_evaluator = ExecutionConstraintEvaluator(constraints)
    broker.query_daily_volume = lambda symbol: 1_000_000.0
    assert manager.submit_orders([_limit_order("measured")])[0].status == OrderStatus.SUBMITTED
    if restart:
        manager, broker = _manager(tmp_path)
        manager.constraint_evaluator = ExecutionConstraintEvaluator(constraints)
        broker.query_daily_volume = lambda symbol: 1_000_000.0
    if missing in {"nav", "both"}:
        broker.query_account_value = lambda: None
    if missing in {"volume", "both"}:
        broker.query_daily_volume = lambda symbol: None
    before = len(broker.submitted)
    state = manager.submit_orders([_limit_order("unmeasured")])[0]
    assert state.status == OrderStatus.REJECTED
    assert "unmeasured" in state.last_message
    assert len(broker.submitted) == before


def test_restart_counts_same_shanghai_session_across_timezones(tmp_path) -> None:
    manager, broker = _manager(tmp_path)
    constraints = replace(manager.constraint_evaluator.constraints, max_orders_per_day=1)
    manager.constraint_evaluator = ExecutionConstraintEvaluator(constraints)
    first = replace(_limit_order("new-york-clock"), timestamp="2026-08-17T22:30:00-04:00")
    assert manager.submit_orders([first])[0].status == OrderStatus.SUBMITTED
    assert manager.book.orders()[0].trade_date == "2026-08-18"
    restored, broker = _manager(tmp_path)
    restored.constraint_evaluator = ExecutionConstraintEvaluator(constraints)
    second = replace(_limit_order("utc-clock"), timestamp="2026-08-18T02:30:00Z")
    assert restored.submit_orders([second])[0].status == OrderStatus.REJECTED
    assert broker.submitted == []
