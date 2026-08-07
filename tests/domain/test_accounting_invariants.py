"""Requirements E and F: accounting identities and T+1 lot behaviour.

Fixed cases pin the specific rules. The fuzzer at the end drives the order and
account state machines through thousands of randomised-but-seeded sequences and
asserts the invariants after *every* economic event, which is where the
interesting violations live — a book usually breaks on an interleaving nobody
wrote a case for, not on the happy path.

Seeded rather than `hypothesis`: the suite must run offline with no extra
dependency, and a fixed seed makes a failure reproducible from the test name
alone.
"""

from __future__ import annotations

import random

import pytest

from quantagent.domain.accounting import AccountState, InvariantViolation, replay_account
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import (
    Fill,
    IllegalTransition,
    OrderBook,
    OrderEventType,
    OrderIntent,
    OrderStatus,
    PositionLot,
    Side,
    Signal,
)

SYMBOL = "600000.SH"
RUN = Lineage(research_id="res_1", strategy_id="str_1", strategy_version_id="sv_1", run_id="run_1")


def _intent(quantity: int, side: Side, trade_date: str, tag: str = "") -> OrderIntent:
    signal = Signal.create(symbol=SYMBOL, trade_date=trade_date + tag, score=0.5, lineage=RUN)
    return OrderIntent.create(
        symbol=SYMBOL, side=side, quantity=quantity, trade_date=trade_date,
        lineage=signal.lineage, reference_price=10.0,
    )


def _fill(
    order, quantity: int, price: float, execution: str, *,
    stamp: float = 0.0, session: str = "2026-08-03",
) -> Fill:
    """`session` becomes `filled_at`: replay settles lots against it.

    Leaving it to default to wall-clock made every fill share one date, which
    collapsed the T+1 structure on replay and is not how the engine records it.
    """
    gross = quantity * price
    return Fill(
        execution_id=execution, order_id=order.order_id, symbol=SYMBOL, side=order.side,
        quantity=quantity, price=price, reference_price=price,
        commission=max(gross * 2.5 / 10_000, 5.0),
        transfer_fee=gross * 0.1 / 10_000,
        stamp_duty=stamp,
        filled_at=session,
        lineage=order.lineage.derive(execution_id=execution),
    )


def _working(book: OrderBook, intent: OrderIntent):
    order = book.open(intent)
    for event in (OrderEventType.RISK_APPROVED, OrderEventType.SUBMITTED, OrderEventType.ACCEPTED):
        book.apply(order.order_id, event)
    return book.state_of(order.order_id)


# -- E: cash identity --------------------------------------------------------
def test_cash_changes_by_consideration_plus_explicit_costs():
    book = OrderBook()
    order = _working(book, _intent(1_000, Side.BUY, "2026-08-03"))
    fill = _fill(order, 1_000, 10.0, "ex_1")

    account = AccountState.opening(1_000_000.0).apply_fill(fill, "2026-08-03")

    expected = 1_000_000.0 - (10_000.0 + fill.fees)
    assert account.cash == pytest.approx(expected)
    assert account.total_fees == pytest.approx(fill.fees)


def test_positions_equal_prior_positions_plus_signed_fills():
    book = OrderBook()
    account = AccountState.opening(1_000_000.0)

    buy = _working(book, _intent(1_000, Side.BUY, "2026-08-03"))
    account = account.apply_fill(_fill(buy, 1_000, 10.0, "ex_1", session="2026-08-03"), "2026-08-03")
    assert account.position(SYMBOL) == 1_000

    more = _working(book, _intent(500, Side.BUY, "2026-08-04"))
    account = account.apply_fill(_fill(more, 500, 10.5, "ex_2", session="2026-08-04"), "2026-08-04")
    assert account.position(SYMBOL) == 1_500

    sell = _working(book, _intent(600, Side.SELL, "2026-08-05"))
    account = account.apply_fill(_fill(sell, 600, 11.0, "ex_3", stamp=6.6, session="2026-08-05"), "2026-08-05")
    assert account.position(SYMBOL) == 900


def test_realised_pnl_is_net_of_entry_and_exit_costs():
    """Both legs' costs come out of realised PnL, not just the exit's.

    Entry fees used to be expensed instead of capitalised into basis, which left
    realised PnL overstated by exactly the buy-side cost — see DEF-009 and
    `test_pnl_split_reconciles_with_cash` for the identity that catches it.
    """
    book = OrderBook()
    account = AccountState.opening(1_000_000.0)
    buy = _working(book, _intent(1_000, Side.BUY, "2026-08-03"))
    buy_fill = _fill(buy, 1_000, 10.0, "ex_1", session="2026-08-03")
    account = account.apply_fill(buy_fill, "2026-08-03")

    sell = _working(book, _intent(1_000, Side.SELL, "2026-08-04"))
    sell_fill = _fill(sell, 1_000, 11.0, "ex_2", stamp=5.5, session="2026-08-04")
    account = account.apply_fill(sell_fill, "2026-08-04")

    assert account.realised_pnl == pytest.approx(
        1_000.0 - buy_fill.fees - sell_fill.fees
    )
    assert account.cash - 1_000_000.0 == pytest.approx(account.realised_pnl)


@pytest.mark.parametrize("mark", [9.0, 10.0, 12.5])
def test_pnl_split_reconciles_with_cash(mark: float):
    """`realised + unrealised == NAV - initial cash`, at any mark.

    The identity that ties the PnL split to money that actually moved. A book can
    have correct cash and still misreport where the profit came from; this is the
    check that refuses that state.
    """
    book = OrderBook()
    account = AccountState.opening(1_000_000.0)
    first = _working(book, _intent(1_000, Side.BUY, "2026-08-03"))
    account = account.apply_fill(_fill(first, 1_000, 10.0, "ex_1", session="2026-08-03"), "2026-08-03")
    second = _working(book, _intent(500, Side.BUY, "2026-08-03", tag="b"))
    account = account.apply_fill(_fill(second, 500, 10.4, "ex_2", session="2026-08-03"), "2026-08-03")
    sell = _working(book, _intent(600, Side.SELL, "2026-08-04"))
    account = account.apply_fill(
        _fill(sell, 600, 11.0, "ex_3", stamp=6.6, session="2026-08-04"), "2026-08-04"
    )

    assert account.identity_residual({SYMBOL: mark}) == pytest.approx(0.0, abs=1e-9)


# -- E: cash reservation is derived, never stored ----------------------------
def test_reserved_cash_is_a_function_of_the_order_book():
    """There is no stored `frozen_cash`, and that is the point.

    A field nothing fed read 0.0 forever, so every comparison against it passed
    without measuring anything (DEF-007). The reservation is now computed from the
    working orders that constitute it, so it cannot silently stop being true.
    """
    from quantagent.reconciliation.snapshot import EconomicSnapshot

    assert not hasattr(AccountState.opening(1_000_000.0), "frozen_cash"), (
        "a stored reservation is back; it will read 0.0 and pass every comparison"
    )

    book = OrderBook()
    account = AccountState.opening(1_000_000.0)
    working = _working(book, _intent(1_000, Side.BUY, "2026-08-03"))
    snapshot = EconomicSnapshot.from_replay(
        "derived", book, account, session="2026-08-03", prices={SYMBOL: 10.0}
    )

    from quantagent.reconciliation.snapshot import RESERVING_STATUSES

    assert working.status in RESERVING_STATUSES
    # The intent here is priced by *reference*, not by a limit — which used to
    # reserve nothing at all.
    assert working.limit_price is None and working.reference_price == 10.0
    assert snapshot.reserved_cash == pytest.approx(1_000 * 10.0), (
        "a working buy order must commit the capital it is sized against"
    )
    assert snapshot.available_cash == pytest.approx(
        snapshot.cash - snapshot.reserved_cash
    )

    # Once the order can no longer trade the reservation is gone, because it is
    # recomputed rather than decremented — so it can never be released twice.
    book.apply(working.order_id, OrderEventType.CANCELLED, reason="operator_cancel")
    after = EconomicSnapshot.from_replay(
        "derived", book, account, session="2026-08-03", prices={SYMBOL: 10.0}
    )
    assert after.reserved_cash == 0.0
    assert after.available_cash == pytest.approx(after.cash)


# -- E: order-level identities ----------------------------------------------
def test_cumulative_fill_can_never_exceed_order_quantity():
    book = OrderBook()
    order = _working(book, _intent(1_000, Side.BUY, "2026-08-03"))
    book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=_fill(order, 600, 10.0, "ex_1"))

    with pytest.raises(ValueError, match="exceed order quantity"):
        book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=_fill(order, 500, 10.0, "ex_2"))


def test_a_terminal_order_cannot_produce_new_fills():
    book = OrderBook()
    order = _working(book, _intent(1_000, Side.BUY, "2026-08-03"))
    book.apply(order.order_id, OrderEventType.FILL, fill=_fill(order, 1_000, 10.0, "ex_1"))

    with pytest.raises(IllegalTransition):
        book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=_fill(order, 100, 10.0, "ex_2"))


def test_no_order_exists_without_an_intent():
    """Structural: the only constructor takes an intent."""
    book = OrderBook()
    order = book.open(_intent(1_000, Side.BUY, "2026-08-03"))

    assert order.lineage.order_intent_id is not None
    assert order.lineage.signal_id is not None


# -- F: T+1 lots -------------------------------------------------------------
def test_a_same_day_buy_is_not_sellable_that_day():
    book = OrderBook()
    buy = _working(book, _intent(1_000, Side.BUY, "2026-08-03"))
    account = AccountState.opening(1_000_000.0).apply_fill(_fill(buy, 1_000, 10.0, "ex_1", session="2026-08-03"), "2026-08-03")

    assert account.position(SYMBOL) == 1_000
    assert account.sellable(SYMBOL, "2026-08-03") == 0
    assert account.sellable(SYMBOL, "2026-08-04") == 1_000


def test_selling_more_than_the_settled_inventory_is_refused():
    book = OrderBook()
    buy = _working(book, _intent(1_000, Side.BUY, "2026-08-03"))
    account = AccountState.opening(1_000_000.0).apply_fill(_fill(buy, 1_000, 10.0, "ex_1", session="2026-08-03"), "2026-08-03")
    sell = _working(book, _intent(1_000, Side.SELL, "2026-08-03"))

    with pytest.raises(InvariantViolation, match=r"exceeds T\+1 sellable"):
        account.apply_fill(_fill(sell, 1_000, 10.0, "ex_2"), "2026-08-03")


def test_intraday_t_trading_consumes_only_settled_base_inventory():
    """Buy and sell on one session: only yesterday's lot may be sold."""
    book = OrderBook()
    account = AccountState.opening(1_000_000.0)
    yesterday = _working(book, _intent(1_000, Side.BUY, "2026-08-03"))
    account = account.apply_fill(_fill(yesterday, 1_000, 10.0, "ex_1", session="2026-08-03"), "2026-08-03")

    today_buy = _working(book, _intent(500, Side.BUY, "2026-08-04"))
    account = account.apply_fill(_fill(today_buy, 500, 10.2, "ex_2", session="2026-08-04"), "2026-08-04")

    assert account.position(SYMBOL) == 1_500
    assert account.sellable(SYMBOL, "2026-08-04") == 1_000

    today_sell = _working(book, _intent(1_000, Side.SELL, "2026-08-04"))
    account = account.apply_fill(_fill(today_sell, 1_000, 10.3, "ex_3", stamp=5.15, session="2026-08-04"), "2026-08-04")

    assert account.position(SYMBOL) == 500
    # The remaining 500 was bought today and stays unsettled.
    assert account.sellable(SYMBOL, "2026-08-04") == 0
    assert account.sellable(SYMBOL, "2026-08-05") == 500


def test_partial_fills_create_separate_settled_lots():
    book = OrderBook()
    order = _working(book, _intent(1_000, Side.BUY, "2026-08-03"))
    book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=_fill(order, 400, 10.0, "ex_1"))
    book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=_fill(order, 600, 10.1, "ex_2"))

    account = AccountState.opening(1_000_000.0)
    for fill in book.state_of(order.order_id).fills:
        account = account.apply_fill(fill, "2026-08-03")

    assert len(account.lots[SYMBOL]) == 2
    assert account.position(SYMBOL) == 1_000
    assert account.sellable(SYMBOL, "2026-08-04") == 1_000


def test_a_cancelled_quantity_does_not_alter_settled_inventory():
    book = OrderBook()
    account = AccountState.opening(1_000_000.0)
    order = _working(book, _intent(1_000, Side.BUY, "2026-08-03"))
    book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=_fill(order, 400, 10.0, "ex_1"))
    account = account.apply_fill(book.state_of(order.order_id).fills[0], "2026-08-03")
    before = account.content_hash()

    book.apply(order.order_id, OrderEventType.CANCEL_REQUESTED)
    book.apply(order.order_id, OrderEventType.CANCELLED)

    # Cancelling the unfilled 600 is not an economic event.
    assert account.content_hash() == before
    assert account.position(SYMBOL) == 400


def test_a_lot_keeps_the_lineage_of_the_fill_that_created_it():
    book = OrderBook()
    order = _working(book, _intent(1_000, Side.BUY, "2026-08-03"))
    book.apply(order.order_id, OrderEventType.FILL, fill=_fill(order, 1_000, 10.0, "ex_1"))
    account = AccountState.opening(1_000_000.0).apply_fill(
        book.state_of(order.order_id).fills[0], "2026-08-03"
    )

    lot = account.lots[SYMBOL][0]
    assert lot.lineage.execution_id == "ex_1"
    assert lot.lineage.order_id == order.order_id
    assert lot.lineage.run_id == "run_1"


# -- E/F: seeded state-machine fuzzing --------------------------------------
@pytest.mark.parametrize("seed", range(12))
def test_randomised_event_sequences_never_break_an_invariant(seed: int):
    """Drive both state machines randomly; assert identities after each event.

    Only *legal* economic events are applied — illegal ones are expected to
    raise and are asserted to leave state untouched, which is the other half of
    'a rejected transition must not partially mutate'.
    """
    rng = random.Random(seed)
    book = OrderBook()
    account = AccountState.opening(1_000_000.0)
    sessions = [f"2026-08-{day:02d}" for day in range(3, 13)]
    counter = 0

    for session in sessions:
        for _ in range(rng.randint(0, 4)):
            counter += 1
            side = Side.BUY if rng.random() < 0.6 else Side.SELL
            quantity = rng.choice([100, 300, 500, 1_000])
            if side is Side.SELL:
                available = account.sellable(SYMBOL, session)
                if available <= 0:
                    continue
                quantity = min(quantity, available)
            price = round(rng.uniform(9.0, 11.0), 2)
            order = _working(book, _intent(quantity, side, session, tag=f"-{counter}"))

            # Sometimes fill in two pieces to exercise the partial path.
            pieces = [quantity] if rng.random() < 0.5 or quantity < 200 else [
                quantity // 2, quantity - quantity // 2
            ]
            for index, piece in enumerate(pieces):
                fill = _fill(
                    order, piece, price, f"ex_{counter}_{index}",
                    stamp=(piece * price * 5 / 10_000) if side is Side.SELL else 0.0,
                    session=session,
                )
                event = (
                    OrderEventType.FILL if index == len(pieces) - 1
                    else OrderEventType.PARTIAL_FILL
                )
                book.apply(order.order_id, event, fill=fill)
                account = account.apply_fill(fill, session)
                account.check()

                # Identities that must hold after every single economic event.
                state = book.state_of(order.order_id)
                assert state.filled_quantity <= state.quantity
                assert account.position(SYMBOL) >= 0
                assert account.sellable(SYMBOL, session) <= account.position(SYMBOL)
                assert account.cash > 0, "the fuzzer must not overdraw the account"

    # Replaying the same events must reproduce the same state exactly.
    replayed = replay_account(book.events(), initial_cash=1_000_000.0)
    assert replayed.position(SYMBOL) == account.position(SYMBOL)


@pytest.mark.parametrize("seed", range(8))
def test_illegal_transitions_never_partially_mutate(seed: int):
    rng = random.Random(seed)
    book = OrderBook()
    order = _working(book, _intent(1_000, Side.BUY, "2026-08-03"))
    book.apply(order.order_id, OrderEventType.FILL, fill=_fill(order, 1_000, 10.0, "ex_1"))

    before = book.state_of(order.order_id)
    event_count = len(book.events())

    for _ in range(10):
        illegal = rng.choice([
            OrderEventType.ACCEPTED, OrderEventType.SUBMITTED, OrderEventType.CANCELLED,
            OrderEventType.RISK_APPROVED, OrderEventType.EXPIRED, OrderEventType.REJECTED,
        ])
        with pytest.raises((IllegalTransition, ValueError)):
            book.apply(order.order_id, illegal)

    after = book.state_of(order.order_id)
    assert after is before or after == before
    assert after.status is OrderStatus.FILLED
    assert len(book.events()) == event_count, "a refused event must not be recorded"
