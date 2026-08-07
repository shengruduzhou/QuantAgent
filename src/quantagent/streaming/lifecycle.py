"""The streaming engine's order lifecycle — which is Module One's, not its own.

The temptation in a streaming engine is to keep a local dict of open orders: it is
faster to write and it makes the matching loop read cleanly. It is also how the
third parallel order model gets created, after the effort of removing the first
two. So this consumer keeps **no order state at all**. It folds every order event
through `OrderBook` and appends it to `CanonicalLedger`, and every economic figure
it reports is a `replay` of that file.

That has a consequence worth stating plainly: the streaming engine inherits
Module One's guarantees rather than reimplementing them. An illegal transition
raises, a re-delivered execution id moves no money, a rejection with fills is
refused, settlement is dated by the fill — none of that is coded here, and none of
it can drift out of agreement with paper, because there is only one implementation.

`mirror_open` and `mirror_event` are used rather than `book.apply` plus a manual
append, for the reason DEF-014 exists: a caller that appends
`history_of(...)[-1]` itself writes the *previous* event whenever the book absorbs
one, and leaves a chain that no longer replays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from quantagent.domain.accounting import AccountState
from quantagent.domain.ledger import CanonicalLedger, mirror_event, mirror_open
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import (
    Fill,
    OrderBook,
    OrderEventType,
    OrderIntent,
    OrderStatus,
    Side,
    Signal,
)
from quantagent.domain.timeline import DecisionFrontier
from quantagent.streaming.events import EventKind, MarketEvent


class UnroutableEvent(RuntimeError):
    """An order event arrived for an order this run never opened.

    Refused rather than opening one implicitly. An implicit order would have no
    intent behind it, and "no order without a recorded intent" is the property that
    makes the chain auditable — `CanonicalLedger.replay` enforces it on the read
    side, so inventing one here would only move the failure later.
    """

    def __init__(self, event: MarketEvent, client_order_id: str) -> None:
        super().__init__(
            f"{event.kind.value} names order {client_order_id!r}, which this run never "
            "opened. An order must be created by an ORDER event carrying its intent "
            "before the venue can report anything about it."
        )
        self.event = event
        self.client_order_id = client_order_id


class MissingEventField(ValueError):
    """An event lacked a field its kind requires. Never defaulted."""


def _required(event: MarketEvent, field_name: str) -> Any:
    if field_name not in event.payload:
        raise MissingEventField(
            f"{event.kind.value} {event.event_id} has no {field_name!r}. This field is "
            "required and is deliberately not defaulted: a guessed quantity, price or "
            "order id books money nobody asked for."
        )
    return event.payload[field_name]


@dataclass
class OrderLifecycle:
    """Folds order events onto the canonical record. Holds no economic state.

    `ledger` and `book` are injected so a streaming run can share the chain a
    paper venue or an OMS is already writing — one economic order, one record.
    """

    ledger: CanonicalLedger
    book: OrderBook = field(default_factory=OrderBook)
    lineage: Lineage = field(default_factory=Lineage)
    initial_cash: float = 1_000_000.0
    #: client order id -> canonical order id. A translation table, not state: every
    #: figure is read back out of the ledger.
    _canonical_ids: dict[str, str] = field(default_factory=dict, init=False)
    #: The session each order belongs to, so settlement is dated by the fill.
    _sessions: dict[str, str] = field(default_factory=dict, init=False)
    handled: int = field(default=0, init=False)

    #: Venue replies and their canonical events. Kept as a table rather than a
    #: chain of `if`s so an unhandled reply is a KeyError naming the value, not a
    #: silent fall-through to "accepted".
    _CALLBACK_EVENTS: Mapping[str, OrderEventType] = field(
        default_factory=lambda: {
            "ACCEPTED": OrderEventType.ACCEPTED,
            "REJECTED": OrderEventType.REJECTED,
        },
        init=False,
        repr=False,
    )

    # -- the consumer -------------------------------------------------------
    def __call__(self, event: MarketEvent, frontier: DecisionFrontier) -> None:
        """Bus handler signature. Non-order events are not this consumer's business."""
        self.handle(event, frontier)

    def handle(self, event: MarketEvent, frontier: DecisionFrontier | None = None) -> None:
        router = {
            EventKind.ORDER: self._on_order,
            EventKind.VENUE_CALLBACK: self._on_callback,
            EventKind.FILL: self._on_fill,
            EventKind.CANCEL: self._on_cancel,
            EventKind.EXPIRY: self._on_expiry,
        }.get(event.kind)
        if router is None:
            return
        router(event)
        self.handled += 1

    # -- handlers -----------------------------------------------------------
    def _on_order(self, event: MarketEvent) -> None:
        client_order_id = str(_required(event, "clientOrderId"))
        session = event.times.event_time.date().isoformat()
        signal = Signal.create(
            symbol=str(event.symbol),
            trade_date=f"{session}-{client_order_id}",
            score=float(event.payload.get("score", 0.0)),
            lineage=self.lineage,
        )
        intent = OrderIntent.create(
            symbol=str(event.symbol),
            side=Side(str(_required(event, "side")).upper()),
            quantity=int(_required(event, "quantity")),
            trade_date=session,
            lineage=signal.lineage,
            limit_price=event.payload.get("limitPrice"),
            reference_price=event.payload.get("referencePrice"),
        )
        canonical = mirror_open(self.book, self.ledger, intent, trade_date=session)
        self._canonical_ids[client_order_id] = canonical.order_id
        self._sessions[client_order_id] = session
        for stage in (OrderEventType.RISK_APPROVED, OrderEventType.SUBMITTED):
            mirror_event(
                self.book, self.ledger, canonical.order_id, stage, trade_date=session
            )

    def _on_callback(self, event: MarketEvent) -> None:
        client_order_id, canonical_id, session = self._resolve(event)
        reply = str(_required(event, "status")).upper()
        if reply not in self._CALLBACK_EVENTS:
            raise MissingEventField(
                f"VENUE_CALLBACK carries status {reply!r}, which has no canonical "
                f"event. Known: {sorted(self._CALLBACK_EVENTS)}. Treating an unknown "
                "reply as an acceptance is how a refused order starts trading."
            )
        mirror_event(
            self.book,
            self.ledger,
            canonical_id,
            self._CALLBACK_EVENTS[reply],
            trade_date=session,
            reason=event.payload.get("reason"),
        )

    def _on_fill(self, event: MarketEvent) -> None:
        client_order_id, canonical_id, session = self._resolve(event)
        canonical = self.book.state_of(canonical_id)
        execution_id = str(_required(event, "executionId"))
        quantity = int(_required(event, "quantity"))
        fill = Fill(
            execution_id=execution_id,
            order_id=canonical_id,
            symbol=canonical.symbol,
            side=canonical.side,
            quantity=quantity,
            price=float(_required(event, "price")),
            reference_price=canonical.limit_price or canonical.reference_price,
            commission=float(event.payload.get("commission", 0.0)),
            stamp_duty=float(event.payload.get("stampDuty", 0.0)),
            transfer_fee=float(event.payload.get("transferFee", 0.0)),
            filled_at=event.times.event_time.isoformat(),
            lineage=canonical.lineage.derive(execution_id=execution_id),
        )
        # Which event this is, is a fact about the order's remaining quantity, not
        # something the venue gets to assert. A venue that mislabelled a partial as
        # final would otherwise close an order that is still working.
        total = canonical.cumulative_quantity + quantity
        kind = (
            OrderEventType.FILL if total >= canonical.quantity
            else OrderEventType.PARTIAL_FILL
        )
        mirror_event(
            self.book, self.ledger, canonical_id, kind, trade_date=session, fill=fill
        )

    def _on_cancel(self, event: MarketEvent) -> None:
        client_order_id, canonical_id, session = self._resolve(event)
        if self.book.state_of(canonical_id).is_terminal:
            # A cancel racing a fill is normal; the fill won. Absorbing it is
            # correct, and it is *not* the same as absorbing a duplicate fill.
            return
        reason = str(event.payload.get("reason") or "operator_cancel")
        for stage in (OrderEventType.CANCEL_REQUESTED, OrderEventType.CANCELLED):
            mirror_event(
                self.book, self.ledger, canonical_id, stage,
                trade_date=session,
                reason=reason if stage is OrderEventType.CANCELLED else None,
            )

    def _on_expiry(self, event: MarketEvent) -> None:
        client_order_id, canonical_id, session = self._resolve(event)
        if self.book.state_of(canonical_id).is_terminal:
            return
        mirror_event(
            self.book, self.ledger, canonical_id, OrderEventType.EXPIRED,
            trade_date=session, reason=str(event.payload.get("reason") or "end_of_session"),
        )

    def _resolve(self, event: MarketEvent) -> tuple[str, str, str]:
        client_order_id = str(_required(event, "clientOrderId"))
        canonical_id = self._canonical_ids.get(client_order_id)
        if canonical_id is None:
            raise UnroutableEvent(event, client_order_id)
        return client_order_id, canonical_id, self._sessions[client_order_id]

    # -- projections --------------------------------------------------------
    def account(self) -> AccountState:
        """Economic state, replayed from the chain. Never maintained here."""
        source = (
            CanonicalLedger(self.ledger.path)
            if self.ledger.path is not None
            else self.ledger
        )
        return source.replay(initial_cash=self.initial_cash)[1]

    def order_book(self) -> OrderBook:
        source = (
            CanonicalLedger(self.ledger.path)
            if self.ledger.path is not None
            else self.ledger
        )
        return source.replay_book()

    def status_of(self, client_order_id: str) -> OrderStatus:
        canonical_id = self._canonical_ids.get(client_order_id)
        if canonical_id is None:
            raise KeyError(f"this run never opened order {client_order_id!r}")
        return self.book.state_of(canonical_id).status


__all__ = ["MissingEventField", "OrderLifecycle", "UnroutableEvent"]
