from __future__ import annotations

from types import SimpleNamespace

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
