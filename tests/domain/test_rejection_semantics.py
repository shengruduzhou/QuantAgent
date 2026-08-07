"""A rejection may only erase an order that never traded.

`ACCEPTED -> REJECTED` was added in Round 4 because a venue can acknowledge an
order and then refuse it at fill time. That is real, but unconstrained it
becomes a way to make executed quantity vanish: the fills are money that already
moved, and no later event can un-move it.

The rule these pin: a rejection requires `cumulative_quantity == 0`, no fills,
and an explicit reason. An order that has traded is cancelled for its remainder
instead, which preserves what happened.
"""

from __future__ import annotations

import pytest

from quantagent.domain.accounting import AccountState
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import (
    ALLOWED_TRANSITIONS,
    Fill,
    IllegalTransition,
    OrderBook,
    OrderEventType,
    OrderIntent,
    OrderStatus,
    Side,
    Signal,
)

SYMBOL = "600000.SH"
RUN = Lineage(research_id="res_1", strategy_id="str_1", strategy_version_id="sv_1", run_id="run_1")


def _accepted(book: OrderBook, quantity: int = 1_000):
    signal = Signal.create(symbol=SYMBOL, trade_date="2026-08-04", score=0.5, lineage=RUN)
    intent = OrderIntent.create(
        symbol=SYMBOL, side=Side.BUY, quantity=quantity, trade_date="2026-08-04",
        lineage=signal.lineage, reference_price=10.0,
    )
    order = book.open(intent)
    for event in (OrderEventType.RISK_APPROVED, OrderEventType.SUBMITTED, OrderEventType.ACCEPTED):
        book.apply(order.order_id, event)
    return book.state_of(order.order_id)


def _fill(order, quantity: int, execution: str = "ex_1") -> Fill:
    return Fill(
        execution_id=execution, order_id=order.order_id, symbol=SYMBOL, side=order.side,
        quantity=quantity, price=10.0, reference_price=10.0, commission=25.0,
        filled_at="2026-08-04", lineage=order.lineage.derive(execution_id=execution),
    )


# -- the permitted case ------------------------------------------------------
def test_accepted_then_rejected_with_zero_fills_is_allowed():
    book = OrderBook()
    order = _accepted(book)

    rejected = book.apply(
        order.order_id, OrderEventType.REJECTED, reason="insufficient_cash",
    )

    assert rejected.status is OrderStatus.REJECTED
    assert rejected.cumulative_quantity == 0
    assert rejected.leaves_quantity == 0, "a rejected order leaves nothing working"
    assert rejected.reason == "insufficient_cash"


def test_a_rejection_must_carry_an_explicit_reason():
    book = OrderBook()
    order = _accepted(book)

    with pytest.raises(ValueError, match="explicit reason"):
        book.apply(order.order_id, OrderEventType.REJECTED)

    assert book.state_of(order.order_id).status is OrderStatus.ACCEPTED


# -- the forbidden cases -----------------------------------------------------
def test_accepted_then_rejected_after_a_fill_is_refused():
    """The fill is money that already moved; a rejection cannot unwind it."""
    book = OrderBook()
    order = _accepted(book)
    book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=_fill(order, 400))

    with pytest.raises(IllegalTransition):
        book.apply(order.order_id, OrderEventType.REJECTED, reason="too_late")

    state = book.state_of(order.order_id)
    assert state.cumulative_quantity == 400, "the executed quantity must survive"
    assert len(state.fills) == 1


def test_partially_filled_to_rejected_is_not_in_the_transition_table():
    assert OrderStatus.REJECTED not in ALLOWED_TRANSITIONS[OrderStatus.PARTIALLY_FILLED]


def test_a_partially_filled_order_cancels_only_its_remainder():
    book = OrderBook()
    order = _accepted(book, quantity=1_000)
    book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=_fill(order, 400))

    book.apply(order.order_id, OrderEventType.CANCEL_REQUESTED)
    cancelled = book.apply(order.order_id, OrderEventType.CANCELLED, reason="operator_cancel")

    assert cancelled.status is OrderStatus.CANCELLED
    assert cancelled.cumulative_quantity == 400, "fills are preserved"
    assert cancelled.leaves_quantity == 0, "only the remainder is terminated"
    assert cancelled.remaining == 600


def test_a_duplicate_rejection_is_refused():
    book = OrderBook()
    order = _accepted(book)
    book.apply(order.order_id, OrderEventType.REJECTED, reason="insufficient_cash")

    with pytest.raises(IllegalTransition):
        book.apply(order.order_id, OrderEventType.REJECTED, reason="insufficient_cash")


def test_a_rejection_after_cancellation_is_refused():
    book = OrderBook()
    order = _accepted(book)
    book.apply(order.order_id, OrderEventType.CANCEL_REQUESTED)
    book.apply(order.order_id, OrderEventType.CANCELLED, reason="operator_cancel")

    with pytest.raises(IllegalTransition):
        book.apply(order.order_id, OrderEventType.REJECTED, reason="late_reject")


def test_a_rejection_after_a_complete_fill_is_refused():
    book = OrderBook()
    order = _accepted(book, quantity=1_000)
    book.apply(order.order_id, OrderEventType.FILL, fill=_fill(order, 1_000))

    with pytest.raises(IllegalTransition):
        book.apply(order.order_id, OrderEventType.REJECTED, reason="late_reject")

    assert book.state_of(order.order_id).cumulative_quantity == 1_000


# -- a refused rejection moves no money --------------------------------------
def test_a_refused_rejection_leaves_cash_and_position_untouched():
    book = OrderBook()
    order = _accepted(book)
    fill = _fill(order, 400)
    book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=fill)
    account = AccountState.opening(1_000_000.0).apply_fill(fill, "2026-08-04")
    before = account.content_hash()

    with pytest.raises(IllegalTransition):
        book.apply(order.order_id, OrderEventType.REJECTED, reason="too_late")

    assert account.content_hash() == before
    assert account.position(SYMBOL) == 400


# -- purpose is tracked separately from state --------------------------------
def test_event_purpose_is_recorded_separately_from_resulting_state():
    """`event_type` says what happened; `status` says where the order now is."""
    book = OrderBook()
    order = _accepted(book, quantity=1_000)
    book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=_fill(order, 1_000, "ex_9"))

    last = book.history_of(order.order_id)[-1]
    state = book.state_of(order.order_id)

    # The event's purpose was a partial fill; completing the quantity moved the
    # order to FILLED. Collapsing the two would lose the venue's own wording.
    assert last.event_type is OrderEventType.PARTIAL_FILL
    assert state.status is OrderStatus.FILLED
    assert state.cumulative_quantity == 1_000
    assert state.last_quantity == 1_000
    assert state.leaves_quantity == 0
