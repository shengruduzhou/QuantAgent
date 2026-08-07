"""The canonical order entities: immutability, legal transitions, full lineage.

These pin the properties the phase depends on. Without them, event-level
reconciliation between the fast and streaming engines is not merely hard, it is
undefined — there is no shared notion of "the same order".
"""

from __future__ import annotations

import pytest

from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import (
    Fill,
    IllegalTransition,
    Order,
    OrderBook,
    OrderEventType,
    OrderIntent,
    OrderStatus,
    PositionLot,
    RiskDecision,
    Side,
    Signal,
    sellable_quantity,
    total_quantity,
)

RUN = Lineage(
    research_id="res_1", strategy_id="str_1", strategy_version_id="sv_1", run_id="run_1"
)


def _intent(quantity: int = 1_000, side: Side = Side.BUY) -> OrderIntent:
    signal = Signal.create(symbol="600000.SH", trade_date="2026-08-03", score=0.9, lineage=RUN)
    return OrderIntent.create(
        symbol="600000.SH", side=side, quantity=quantity,
        trade_date="2026-08-03", lineage=signal.lineage, reference_price=10.0,
    )


def _fill(order: Order, quantity: int, price: float = 10.0, execution: str = "ex_1") -> Fill:
    return Fill(
        execution_id=execution, order_id=order.order_id, symbol=order.symbol, side=order.side,
        quantity=quantity, price=price, reference_price=10.0,
        commission=quantity * price * 2.5 / 10_000,
        lineage=order.lineage.derive(execution_id=execution),
    )


# -- lineage -----------------------------------------------------------------
def test_lineage_survives_from_signal_to_fill():
    intent = _intent()
    book = OrderBook()
    order = book.open(intent)
    book.apply(order.order_id, OrderEventType.RISK_APPROVED)
    book.apply(order.order_id, OrderEventType.SUBMITTED)
    book.apply(order.order_id, OrderEventType.ACCEPTED)
    filled = book.apply(order.order_id, OrderEventType.FILL, fill=_fill(order, 1_000))

    chain = filled.fills[0].lineage
    assert chain.research_id == "res_1"
    assert chain.strategy_version_id == "sv_1"
    assert chain.run_id == "run_1"
    assert chain.signal_id == intent.lineage.signal_id
    assert chain.order_intent_id == intent.order_intent_id
    assert chain.order_id == order.order_id
    assert chain.execution_id == "ex_1"


def test_identifiers_are_reproducible_across_engines():
    """Two engines building the same intent must agree on its identity."""
    assert _intent().order_intent_id == _intent().order_intent_id
    assert _intent(quantity=500).order_intent_id != _intent(quantity=1_000).order_intent_id


# -- immutability ------------------------------------------------------------
def test_applying_an_event_returns_a_new_snapshot():
    book = OrderBook()
    order = book.open(_intent())
    approved = book.apply(order.order_id, OrderEventType.RISK_APPROVED)

    assert order.status is OrderStatus.PENDING_RISK, "the earlier snapshot must not mutate"
    assert approved.status is OrderStatus.APPROVED
    with pytest.raises(AttributeError):
        order.status = OrderStatus.FILLED  # type: ignore[misc]


def test_history_is_append_only_and_complete():
    book = OrderBook()
    order = book.open(_intent())
    for event in (OrderEventType.RISK_APPROVED, OrderEventType.SUBMITTED, OrderEventType.ACCEPTED):
        book.apply(order.order_id, event)
    book.apply(order.order_id, OrderEventType.FILL, fill=_fill(order, 1_000))

    history = [event.event_type for event in book.history_of(order.order_id)]

    assert history == [
        OrderEventType.CREATED, OrderEventType.RISK_APPROVED,
        OrderEventType.SUBMITTED, OrderEventType.ACCEPTED, OrderEventType.FILL,
    ]
    assert [e.sequence for e in book.events()] == list(range(5))


# -- transitions -------------------------------------------------------------
def test_a_fill_after_a_terminal_cancel_is_rejected():
    """The bug that silently inflates a simulated book."""
    book = OrderBook()
    order = book.open(_intent())
    book.apply(order.order_id, OrderEventType.RISK_APPROVED)
    book.apply(order.order_id, OrderEventType.SUBMITTED)
    book.apply(order.order_id, OrderEventType.ACCEPTED)
    book.apply(order.order_id, OrderEventType.CANCEL_REQUESTED)
    book.apply(order.order_id, OrderEventType.CANCELLED)

    with pytest.raises(IllegalTransition):
        book.apply(order.order_id, OrderEventType.FILL, fill=_fill(order, 1_000))


def test_a_fill_racing_an_in_flight_cancel_is_allowed():
    """The venue may legitimately trade before it processes the cancel."""
    book = OrderBook()
    order = book.open(_intent())
    for event in (OrderEventType.RISK_APPROVED, OrderEventType.SUBMITTED, OrderEventType.ACCEPTED):
        book.apply(order.order_id, event)
    book.apply(order.order_id, OrderEventType.CANCEL_REQUESTED)

    filled = book.apply(order.order_id, OrderEventType.FILL, fill=_fill(order, 1_000))

    assert filled.status is OrderStatus.FILLED


def test_a_rejected_event_is_recorded_before_the_state_changes():
    book = OrderBook()
    order = book.open(_intent())
    decision = RiskDecision.create(
        approved=False, rule="max_position_weight", threshold=0.05, measured=0.12,
        reason="single-name limit exceeded", lineage=order.lineage,
    )

    rejected = book.apply(
        order.order_id, OrderEventType.RISK_REJECTED,
        reason=decision.reason, risk_decision=decision,
    )

    assert rejected.status is OrderStatus.REJECTED
    assert rejected.is_terminal
    event = book.history_of(order.order_id)[-1]
    assert event.risk_decision.rule == "max_position_weight"
    assert event.risk_decision.measured == 0.12


def test_an_illegal_transition_does_not_record_an_event():
    """A refused event must leave no trace, or history lies about what happened."""
    book = OrderBook()
    order = book.open(_intent())
    before = len(book.events())

    with pytest.raises(IllegalTransition):
        book.apply(order.order_id, OrderEventType.FILL, fill=_fill(order, 1_000))

    assert len(book.events()) == before
    assert book.state_of(order.order_id).status is OrderStatus.PENDING_RISK


# -- partial fills -----------------------------------------------------------
def test_partial_fills_accumulate_and_close_the_order_exactly():
    book = OrderBook()
    order = book.open(_intent(quantity=1_000))
    for event in (OrderEventType.RISK_APPROVED, OrderEventType.SUBMITTED, OrderEventType.ACCEPTED):
        book.apply(order.order_id, event)

    first = book.apply(order.order_id, OrderEventType.PARTIAL_FILL,
                       fill=_fill(order, 400, price=10.0, execution="ex_1"))
    assert first.status is OrderStatus.PARTIALLY_FILLED
    assert first.remaining == 600

    second = book.apply(order.order_id, OrderEventType.PARTIAL_FILL,
                        fill=_fill(order, 600, price=10.5, execution="ex_2"))

    # Completing the quantity closes the order even via a PARTIAL_FILL event.
    assert second.status is OrderStatus.FILLED
    assert second.remaining == 0
    assert second.average_fill_price == pytest.approx((400 * 10.0 + 600 * 10.5) / 1000)


def test_fills_cannot_exceed_the_order_quantity():
    book = OrderBook()
    order = book.open(_intent(quantity=1_000))
    for event in (OrderEventType.RISK_APPROVED, OrderEventType.SUBMITTED, OrderEventType.ACCEPTED):
        book.apply(order.order_id, event)
    book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=_fill(order, 900, execution="ex_1"))

    with pytest.raises(ValueError, match="exceed order quantity"):
        book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=_fill(order, 200, execution="ex_2"))


# -- cost accounting ---------------------------------------------------------
def test_slippage_is_derived_from_the_price_never_charged_as_a_fee():
    order = OrderBook().open(_intent())
    fill = Fill(
        execution_id="ex_1", order_id=order.order_id, symbol="600000.SH", side=Side.BUY,
        quantity=1_000, price=10.02, reference_price=10.0, commission=25.0,
    )

    assert fill.slippage == pytest.approx(20.0)
    assert fill.fees == pytest.approx(25.0), "slippage must not appear in fees"
    assert fill.cash_delta == pytest.approx(-(10_020.0 + 25.0))


def test_a_sell_returns_cash_net_of_fees():
    order = OrderBook().open(_intent(side=Side.SELL))
    fill = Fill(
        execution_id="ex_1", order_id=order.order_id, symbol="600000.SH", side=Side.SELL,
        quantity=1_000, price=10.0, reference_price=10.0,
        commission=25.0, stamp_duty=50.0, transfer_fee=1.0,
    )

    assert fill.cash_delta == pytest.approx(10_000.0 - 76.0)


# -- T+1 lots ----------------------------------------------------------------
def test_a_lot_bought_today_is_not_sellable_today():
    lot = PositionLot(
        position_lot_id="lot_1", symbol="600000.SH", quantity=1_000,
        cost_price=10.0, acquired_on="2026-08-03",
    )

    assert not lot.sellable_on("2026-08-03")
    assert lot.sellable_on("2026-08-04")


def test_sellable_quantity_counts_only_settled_lots():
    lots = [
        PositionLot("lot_1", "600000.SH", 1_000, 10.0, "2026-08-03"),
        PositionLot("lot_2", "600000.SH", 500, 10.5, "2026-08-04"),
    ]

    assert total_quantity(lots) == 1_500
    assert sellable_quantity(lots, "2026-08-04") == 1_000
    assert sellable_quantity(lots, "2026-08-05") == 1_500


def test_a_lot_traces_back_to_the_fill_that_created_it():
    book = OrderBook()
    order = book.open(_intent())
    for event in (OrderEventType.RISK_APPROVED, OrderEventType.SUBMITTED, OrderEventType.ACCEPTED):
        book.apply(order.order_id, event)
    filled = book.apply(order.order_id, OrderEventType.FILL, fill=_fill(order, 1_000))

    lot = PositionLot.from_fill(filled.fills[0], trade_date="2026-08-03")

    assert lot.lineage.execution_id == "ex_1"
    assert lot.lineage.order_id == order.order_id
    assert lot.lineage.signal_id == order.lineage.signal_id


# -- parent/child ------------------------------------------------------------
def test_child_orders_are_linked_to_their_parent():
    book = OrderBook()
    parent = book.open(_intent(quantity=10_000))
    child_intent = OrderIntent.create(
        symbol="600000.SH", side=Side.BUY, quantity=2_000, trade_date="2026-08-03",
        lineage=parent.lineage, reference_price=10.0,
    )

    child = book.open(child_intent, parent_order_id=parent.order_id)

    assert child.parent_order_id == parent.order_id
    assert child.lineage.parent_order_id == parent.order_id
    assert book.children_of(parent.order_id) == (child,)


def test_reopening_the_same_intent_returns_the_same_order():
    """The domain-level half of the idempotency guarantee."""
    book = OrderBook()
    intent = _intent()

    first = book.open(intent)
    second = book.open(intent)

    assert first.order_id == second.order_id
    assert len(book.orders()) == 1
