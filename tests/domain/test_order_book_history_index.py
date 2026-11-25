"""OrderBook.history_of is served from a per-order index (cycle 32 performance fix).

The index must return exactly what the previous full scan of the append-only
event list returned: same events, same order, for interleaved orders, ignored
duplicate executions and a book rebuilt from the canonical ledger.
"""

from __future__ import annotations

from quantagent.domain.ledger import CanonicalLedger, mirror_event, mirror_open
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import Fill, Order, OrderBook, OrderEventType, OrderIntent, Side, Signal

RUN = Lineage(research_id="res_1", strategy_id="str_1", strategy_version_id="sv_1", run_id="run_1")
SESSION = "2026-08-03"


def _intent(symbol: str, quantity: int = 1_000) -> OrderIntent:
    signal = Signal.create(symbol=symbol, trade_date=SESSION, score=0.9, lineage=RUN)
    return OrderIntent.create(symbol=symbol, side=Side.BUY, quantity=quantity, trade_date=SESSION,
                              lineage=signal.lineage, reference_price=10.0)


def _fill(order: Order, quantity: int, execution: str) -> Fill:
    return Fill(execution_id=execution, order_id=order.order_id, symbol=order.symbol, side=order.side,
                quantity=quantity, price=10.0, reference_price=10.0, commission=1.0,
                lineage=order.lineage.derive(execution_id=execution))


def _full_scan(book: OrderBook, order_id: str) -> tuple:
    return tuple(event for event in book.events() if event.order_id == order_id)


def _assert_index_equals_scan(book: OrderBook) -> None:
    for order in book.orders():
        assert book.history_of(order.order_id) == _full_scan(book, order.order_id)
    assert book.history_of("never-opened") == ()
    flattened = sorted((e for o in book.orders() for e in book.history_of(o.order_id)), key=lambda e: e.sequence)
    assert tuple(flattened) == book.events()


def _drive(book: OrderBook, ledger: CanonicalLedger | None = None) -> list[Order]:
    def open_(intent):
        return mirror_open(book, ledger, intent, trade_date=SESSION) if ledger is not None else book.open(intent)

    def apply(order_id, event_type, **kw):
        if ledger is not None:
            return mirror_event(book, ledger, order_id, event_type, trade_date=SESSION, **kw)
        return book.apply(order_id, event_type, **kw)

    orders = [open_(_intent(s)) for s in ("600000.SH", "000001.SZ", "300750.SZ")]
    for step in (OrderEventType.RISK_APPROVED, OrderEventType.SUBMITTED, OrderEventType.ACCEPTED):
        for order in orders:  # interleaved across orders
            apply(order.order_id, step)
    apply(orders[0].order_id, OrderEventType.FILL, fill=_fill(orders[0], 400, "ex_a"))
    apply(orders[2].order_id, OrderEventType.FILL, fill=_fill(orders[2], 1_000, "ex_c"))
    apply(orders[0].order_id, OrderEventType.FILL, fill=_fill(orders[0], 600, "ex_b"))
    return orders


def test_index_matches_full_scan_for_interleaved_orders():
    book = OrderBook()
    _drive(book)
    _assert_index_equals_scan(book)
    assert [e.event_type for e in book.history_of(book.orders()[1].order_id)][-1] is OrderEventType.ACCEPTED


def test_duplicate_execution_leaves_no_trace_in_either_view():
    book = OrderBook()
    orders = _drive(book)
    before = book.history_of(orders[0].order_id)
    duplicate = book.events()[-1].fill  # re-delivered execution of ex_b
    book.apply(orders[0].order_id, OrderEventType.FILL, fill=duplicate)
    assert book.history_of(orders[0].order_id) == before
    _assert_index_equals_scan(book)


def test_duplicate_open_does_not_duplicate_history():
    book = OrderBook()
    intent = _intent("600000.SH")
    first = book.open(intent)
    again = book.open(intent)
    assert again is first
    assert len(book.history_of(first.order_id)) == 1
    _assert_index_equals_scan(book)


def test_index_survives_ledger_reload_and_replay(tmp_path):
    path = tmp_path / "chain.jsonl"
    book = OrderBook()
    _drive(book, CanonicalLedger(path))
    replayed = CanonicalLedger(path).replay_book()
    _assert_index_equals_scan(replayed)
    # Replay re-derives order ids from the recorded intents, so match by symbol.
    def per_symbol(b: OrderBook) -> dict:
        return {o.symbol: [(e.event_type, None if e.fill is None else (e.fill.execution_id, e.fill.quantity))
                           for e in b.history_of(o.order_id)] for o in b.orders()}

    assert per_symbol(replayed) == per_symbol(book)
    assert len(replayed.events()) == len(book.events())
