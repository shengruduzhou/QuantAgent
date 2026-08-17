"""Production pre-trade must fail closed on constraints it could not measure.

Round 21 / R2 (risk) finding.  Two portfolio-level limits were structurally
unreachable from the production order path:

* ``max_single_stock_participation_rate`` — the evaluator skipped any symbol
  without a day-volume hint (``if not dvol: continue``) and ``OrderManager``
  hardcoded ``daily_volume_hint=None``, so the limit could never fire.
* ``max_daily_turnover`` — the evaluator skipped the check when no intent
  carried a NAV (``if navs:``) and ``OrderManager`` swallowed every exception
  from ``query_account_value`` into ``nav = None``.

In both cases the report still read ``passed=True``: a measurement that was
never taken was indistinguishable from a limit that was honoured.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from quantagent.domain.lineage import Lineage
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
    OrderIntentRecord,
)
from quantagent.execution.order_manager import OrderManager


class _LiveBroker:
    """Broker double advertising the production risk contract."""

    def __init__(self, *, nav: float | None = 100_000.0, day_volume: float | None = None) -> None:
        self.config = SimpleNamespace(
            require_risk_approval=True,
            live_trading_enabled=True,
            dry_run=False,
        )
        self._nav = nav
        self._day_volume = day_volume
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
        if self._nav is None:
            raise RuntimeError("account value unavailable")
        return self._nav

    def query_daily_volume(self, symbol: str) -> float:
        if self._day_volume is None:
            raise RuntimeError("day volume unavailable")
        return self._day_volume

    def on_trade(self, callback) -> None:
        return None


def _constraints(**overrides) -> ExecutionConstraintSet:
    base = dict(
        max_orders_per_second=None,
        max_orders_per_day=None,
        max_cancel_ratio=None,
        min_order_resting_time_seconds=None,
        max_single_stock_participation_rate=None,
        max_single_order_value=None,
        max_daily_turnover=None,
        auction_mode_max_orders_per_symbol=None,
        auction_mode_min_resting_time_seconds=None,
        no_spoofing=False,
        no_layering=False,
        no_pull_push=False,
        qmt_dry_run_required_by_default=False,
        live_trading_enabled=True,
    )
    base.update(overrides)
    return ExecutionConstraintSet(**base)


def _manager(tmp_path, broker: _LiveBroker, constraints: ExecutionConstraintSet) -> OrderManager:
    return OrderManager(
        broker=broker,
        lineage=Lineage(run_id="run-unmeasured"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        idempotency_path=str(tmp_path / "idem.jsonl"),
        constraint_evaluator=ExecutionConstraintEvaluator(constraints),
    )


def _order(client_order_id: str = "ord-1", quantity: int = 100) -> Order:
    return Order(
        client_order_id=client_order_id,
        symbol="600000.SH",
        side=OrderSide.BUY,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        price=10.0,
        signal_id=f"sig-{client_order_id}",
        strategy_version="production-candidate",
        timestamp="2026-08-18T02:30:00+00:00",
    )


def _intent(**overrides) -> OrderIntentRecord:
    base = dict(
        intent_id="i-1",
        symbol="600000.SH",
        side="buy",
        quantity=1_000,
        price=10.0,
        timestamp=pd.Timestamp("2026-08-18T02:30:00Z"),
        order_value=10_000.0,
    )
    base.update(overrides)
    return OrderIntentRecord(**base)


# ---------------------------------------------------------------------------
# Evaluator level
# ---------------------------------------------------------------------------

def test_participation_without_day_volume_is_unmeasured_not_passed() -> None:
    evaluator = ExecutionConstraintEvaluator(
        _constraints(max_single_stock_participation_rate=0.10)
    )
    report = evaluator.evaluate([_intent(daily_volume_hint=None)])

    assert report.passed is True, "no confirmed breach, so passed stays true"
    assert report.fully_measured is False
    assert report.unmeasured_constraints == ["max_single_stock_participation_rate"]
    assert report.to_dict()["unmeasured"][0]["missing_input"] == "daily_volume_hint"


def test_turnover_without_nav_is_unmeasured_not_passed() -> None:
    evaluator = ExecutionConstraintEvaluator(_constraints(max_daily_turnover=2.0))
    report = evaluator.evaluate([_intent(portfolio_nav=None)])

    assert report.passed is True
    assert report.fully_measured is False
    assert report.unmeasured_constraints == ["max_daily_turnover"]
    assert report.to_dict()["unmeasured"][0]["missing_input"] == "portfolio_nav"


def test_measured_inputs_leave_no_unmeasured_record() -> None:
    evaluator = ExecutionConstraintEvaluator(
        _constraints(max_single_stock_participation_rate=0.10, max_daily_turnover=2.0)
    )
    report = evaluator.evaluate(
        [_intent(daily_volume_hint=1_000_000.0, portfolio_nav=1_000_000.0)]
    )

    assert report.passed is True
    assert report.fully_measured is True
    assert report.unmeasured == []


def test_measured_participation_breach_still_blocks() -> None:
    evaluator = ExecutionConstraintEvaluator(
        _constraints(max_single_stock_participation_rate=0.10)
    )
    report = evaluator.evaluate([_intent(quantity=5_000, daily_volume_hint=10_000.0)])

    assert report.passed is False
    assert report.fully_measured is True
    assert "max_single_stock_participation_rate" in report.by_constraint


# ---------------------------------------------------------------------------
# OrderManager level — the production path
# ---------------------------------------------------------------------------

def test_production_submit_refuses_when_nav_is_unavailable(tmp_path) -> None:
    broker = _LiveBroker(nav=None, day_volume=1_000_000.0)
    manager = _manager(tmp_path, broker, _constraints(max_daily_turnover=2.0))

    state = manager.submit_orders([_order()])[0]

    assert state.status is OrderStatus.REJECTED
    assert broker.submitted == [], "order must not reach the broker"


def test_production_submit_refuses_when_day_volume_is_unavailable(tmp_path) -> None:
    broker = _LiveBroker(nav=100_000.0, day_volume=None)
    manager = _manager(tmp_path, broker, _constraints(max_single_stock_participation_rate=0.10))

    state = manager.submit_orders([_order()])[0]

    assert state.status is OrderStatus.REJECTED
    assert broker.submitted == []


def test_production_submit_proceeds_when_every_constraint_is_measured(tmp_path) -> None:
    broker = _LiveBroker(nav=100_000.0, day_volume=1_000_000.0)
    manager = _manager(
        tmp_path,
        broker,
        _constraints(max_single_stock_participation_rate=0.10, max_daily_turnover=2.0),
    )

    state = manager.submit_orders([_order()])[0]

    assert state.status is not OrderStatus.REJECTED
    assert len(broker.submitted) == 1


def test_injected_day_volume_hint_satisfies_the_participation_input(tmp_path) -> None:
    broker = _LiveBroker(nav=100_000.0, day_volume=None)
    manager = _manager(tmp_path, broker, _constraints(max_single_stock_participation_rate=0.10))
    manager.daily_volume_hints["600000.SH"] = 1_000_000.0

    state = manager.submit_orders([_order()])[0]

    assert state.status is not OrderStatus.REJECTED
    assert len(broker.submitted) == 1
