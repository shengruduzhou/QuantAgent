"""The canonical order and execution entities.

One vocabulary for research simulation, the fast backtest, the streaming
backtest, paper, shadow and any future broker adapter. Today each engine has its
own shapes — `backtest/engine.py` logs dicts, `execution/order_manager.py` has
`OrderIntent`, `paper/orders.py` has a state machine — which is why an order in
one engine cannot be compared to "the same" order in another. Event-level
reconciliation is impossible without a shared type.

Three properties carry the weight:

* **Entities are immutable.** State advances by appending an `OrderEvent` and
  folding it into a *new* snapshot. Nothing is edited in place, so an earlier
  state can never be silently overwritten and the history is always the whole
  history.
* **Every entity carries `Lineage`.** A fill knows which intent, signal,
  strategy version and run produced it, which is what makes the required
  drill-down (strategy -> signal -> order -> fill -> position -> cash -> NAV) a
  query rather than a reconstruction.
* **Illegal transitions raise.** A fill after a terminal cancel is the bug that
  silently inflates a simulated book; it is rejected here rather than absorbed.

Deliberately absent: an unbounded market order. On A-shares an order with no
price bound is how a simulation fills through a limit board and books profit
that was never obtainable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from quantagent.domain.lineage import Lineage, content_id


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


class OrderStatus(str, Enum):
    #: Created but not yet through risk.
    PENDING_RISK = "PENDING_RISK"
    #: Risk approved it; not yet at the venue.
    APPROVED = "APPROVED"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


TERMINAL_STATUSES: frozenset[OrderStatus] = frozenset({
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.REJECTED,
    OrderStatus.EXPIRED,
})

#: A fill arriving while a cancel is in flight is legitimate — the venue may
#: have traded before it processed the cancel — so CANCEL_REQUESTED keeps its
#: fill transitions. Everything terminal is genuinely final.
ALLOWED_TRANSITIONS: Mapping[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.PENDING_RISK: frozenset({OrderStatus.APPROVED, OrderStatus.REJECTED}),
    OrderStatus.APPROVED: frozenset({OrderStatus.SUBMITTED, OrderStatus.CANCELLED, OrderStatus.REJECTED}),
    OrderStatus.SUBMITTED: frozenset({
        OrderStatus.ACCEPTED, OrderStatus.REJECTED, OrderStatus.CANCEL_REQUESTED, OrderStatus.EXPIRED,
    }),
    # REJECTED after ACCEPTED is real: a venue can acknowledge an order and then
    # refuse it at fill time — insufficient cash or a credit breach discovered
    # on execution. Excluding it forced the paper broker to keep its own
    # transition table, which is the duplication this module exists to remove.
    # PARTIALLY_FILLED deliberately keeps no REJECTED edge: rejecting an order
    # that has already traded would erase fills that really happened, so such an
    # order is CANCELLED for its remainder instead.
    OrderStatus.ACCEPTED: frozenset({
        OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCEL_REQUESTED,
        OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.REJECTED,
    }),
    OrderStatus.PARTIALLY_FILLED: frozenset({
        OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCEL_REQUESTED,
        OrderStatus.CANCELLED, OrderStatus.EXPIRED,
    }),
    OrderStatus.CANCEL_REQUESTED: frozenset({
        OrderStatus.CANCELLED, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
    }),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
    OrderStatus.EXPIRED: frozenset(),
}


class OrderEventType(str, Enum):
    CREATED = "CREATED"
    RISK_APPROVED = "RISK_APPROVED"
    RISK_REJECTED = "RISK_REJECTED"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class IllegalTransition(RuntimeError):
    """An event that would move an order somewhere it cannot legally go."""

    def __init__(self, order_id: str, current: OrderStatus, target: OrderStatus) -> None:
        super().__init__(
            f"order {order_id}: {current.value} -> {target.value} is not a legal transition"
        )
        self.order_id = order_id
        self.current = current
        self.target = target


class DuplicateExecution(RuntimeError):
    """One execution id was reported twice with different economics.

    A venue re-delivering the *same* execution is normal and absorbed silently.
    Re-using an execution id for a different quantity, price or fee is not: one
    of the two reports is wrong, and picking either would book money nobody can
    reconcile.
    """

    def __init__(self, order_id: str, execution_id: str, stored: "Fill", attempted: "Fill") -> None:
        super().__init__(
            f"order {order_id}: execution {execution_id} was already applied as "
            f"{stored.quantity}@{stored.price} (fees {stored.fees}); this report says "
            f"{attempted.quantity}@{attempted.price} (fees {attempted.fees})"
        )
        self.order_id = order_id
        self.execution_id = execution_id
        self.stored = stored
        self.attempted = attempted


def same_execution(left: "Fill", right: "Fill") -> bool:
    """Whether two reports of one execution id describe the same economic event."""
    return (
        left.side is right.side
        and left.quantity == right.quantity
        and abs(left.price - right.price) < 1e-9
        and abs(left.fees - right.fees) < 1e-9
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Upstream entities
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Signal:
    """A model's view on a symbol at a point in time."""

    signal_id: str
    symbol: str
    trade_date: str
    score: float
    lineage: Lineage = field(default_factory=Lineage)
    created_at: str = field(default_factory=_now)

    @classmethod
    def create(cls, *, symbol: str, trade_date: str, score: float, lineage: Lineage) -> "Signal":
        signal_id = content_id(
            "sig",
            run_id=lineage.run_id,
            strategy_version_id=lineage.strategy_version_id,
            symbol=symbol,
            trade_date=trade_date,
        )
        return cls(
            signal_id=signal_id, symbol=symbol, trade_date=trade_date, score=score,
            lineage=lineage.derive(signal_id=signal_id),
        )


@dataclass(frozen=True, slots=True)
class TargetPosition:
    """What the portfolio layer wants to hold, before any execution reality."""

    symbol: str
    trade_date: str
    target_weight: float
    target_shares: int
    lineage: Lineage = field(default_factory=Lineage)


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """The economic ask, before risk has seen it.

    Separate from `Order` on purpose: an intent that risk refuses must remain
    queryable as something the strategy *wanted*, not vanish.
    """

    order_intent_id: str
    symbol: str
    side: Side
    quantity: int
    trade_date: str
    limit_price: float | None = None
    reference_price: float | None = None
    lineage: Lineage = field(default_factory=Lineage)
    created_at: str = field(default_factory=_now)

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        side: Side,
        quantity: int,
        trade_date: str,
        lineage: Lineage,
        limit_price: float | None = None,
        reference_price: float | None = None,
    ) -> "OrderIntent":
        if quantity <= 0:
            raise ValueError("an order intent must have a positive quantity")
        intent_id = content_id(
            "oi",
            run_id=lineage.run_id,
            signal_id=lineage.signal_id,
            symbol=symbol,
            side=side.value,
            quantity=int(quantity),
            trade_date=trade_date,
        )
        return cls(
            order_intent_id=intent_id, symbol=symbol, side=side, quantity=int(quantity),
            trade_date=trade_date, limit_price=limit_price, reference_price=reference_price,
            lineage=lineage.derive(order_intent_id=intent_id),
        )


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """First-class evidence of why an order was allowed or refused.

    Carries the rule, threshold and measured value, not just a verdict — a
    refusal an operator cannot audit is indistinguishable from a bug.
    """

    risk_decision_id: str
    approved: bool
    rule: str
    threshold: Any
    measured: Any
    reason: str
    decided_by: str = "risk_engine"
    decided_at: str = field(default_factory=_now)
    lineage: Lineage = field(default_factory=Lineage)

    @classmethod
    def create(
        cls, *, approved: bool, rule: str, threshold: Any, measured: Any,
        reason: str, lineage: Lineage, decided_by: str = "risk_engine",
    ) -> "RiskDecision":
        decision_id = content_id(
            "rd", order_intent_id=lineage.order_intent_id, rule=rule,
            approved=approved, measured=measured,
        )
        return cls(
            risk_decision_id=decision_id, approved=approved, rule=rule, threshold=threshold,
            measured=measured, reason=reason, decided_by=decided_by,
            lineage=lineage.derive(risk_decision_id=decision_id),
        )


# ---------------------------------------------------------------------------
# Execution entities
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Fill:
    """One execution. Costs are itemised so they can never be double-counted."""

    execution_id: str
    order_id: str
    symbol: str
    side: Side
    quantity: int
    price: float
    #: The venue/reference price before slippage, so slippage stays derivable.
    reference_price: float | None = None
    commission: float = 0.0
    stamp_duty: float = 0.0
    transfer_fee: float = 0.0
    filled_at: str = field(default_factory=_now)
    lineage: Lineage = field(default_factory=Lineage)

    @property
    def gross(self) -> float:
        return self.quantity * self.price

    @property
    def fees(self) -> float:
        return self.commission + self.stamp_duty + self.transfer_fee

    @property
    def slippage(self) -> float:
        """Implicit in the price, never charged again as a fee."""
        if self.reference_price is None:
            return 0.0
        return abs(self.price - self.reference_price) * self.quantity

    @property
    def cash_delta(self) -> float:
        """Signed cash effect: negative for a buy, positive for a sell."""
        return -(self.gross + self.fees) if self.side is Side.BUY else (self.gross - self.fees)


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """An immutable fact about an order. The only way state advances."""

    event_type: OrderEventType
    order_id: str
    sequence: int
    event_time: str
    fill: Fill | None = None
    reason: str | None = None
    risk_decision: RiskDecision | None = None
    lineage: Lineage = field(default_factory=Lineage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eventType": self.event_type.value,
            "orderId": self.order_id,
            "sequence": self.sequence,
            "eventTime": self.event_time,
            "reason": self.reason,
            "executionId": self.fill.execution_id if self.fill else None,
            "quantity": self.fill.quantity if self.fill else None,
            "price": self.fill.price if self.fill else None,
            "riskDecisionId": self.risk_decision.risk_decision_id if self.risk_decision else None,
            "lineage": self.lineage.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class Order:
    """An immutable snapshot of one order at one point in its history."""

    order_id: str
    symbol: str
    side: Side
    quantity: int
    status: OrderStatus
    trade_date: str
    limit_price: float | None = None
    #: The price the intent was sized against, carried so a reservation always has
    #: a basis. Without it an order priced only by reference reserved nothing, and
    #: committed capital read as zero for an order that was genuinely working.
    reference_price: float | None = None
    filled_quantity: int = 0
    parent_order_id: str | None = None
    broker_order_id: str | None = None
    lineage: Lineage = field(default_factory=Lineage)
    fills: tuple[Fill, ...] = ()
    reason: str | None = None

    @property
    def remaining(self) -> int:
        return max(0, self.quantity - self.filled_quantity)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def average_fill_price(self) -> float | None:
        if not self.fills:
            return None
        volume = sum(fill.quantity for fill in self.fills)
        return sum(fill.quantity * fill.price for fill in self.fills) / volume if volume else None

    @property
    def total_fees(self) -> float:
        return sum(fill.fees for fill in self.fills)

    @classmethod
    def from_intent(cls, intent: OrderIntent, *, parent_order_id: str | None = None) -> "Order":
        order_id = content_id("ord", order_intent_id=intent.order_intent_id, parent=parent_order_id)
        return cls(
            order_id=order_id, symbol=intent.symbol, side=intent.side, quantity=intent.quantity,
            status=OrderStatus.PENDING_RISK, trade_date=intent.trade_date,
            limit_price=intent.limit_price, reference_price=intent.reference_price,
            parent_order_id=parent_order_id,
            lineage=intent.lineage.derive(order_id=order_id, parent_order_id=parent_order_id),
        )

    def apply(self, event: OrderEvent) -> "Order":
        """Fold an event into a *new* snapshot. Never mutates."""
        target = _TARGET_STATUS[event.event_type]
        if event.event_type in {OrderEventType.PARTIAL_FILL, OrderEventType.FILL}:
            if event.fill is None:
                raise ValueError(f"{event.event_type.value} requires a fill")
            # One execution id, one economic effect. A venue that re-delivers an
            # execution — a retried callback, a reconnect replay, a gateway
            # retry — must not move money twice, so an identical re-report
            # returns this same snapshot and records nothing. Returning `self`
            # rather than a copy is load-bearing: `OrderBook.apply` reads that
            # identity to decide the event never happened.
            existing = next(
                (f for f in self.fills if f.execution_id == event.fill.execution_id), None
            )
            if existing is not None:
                if same_execution(existing, event.fill):
                    return self
                raise DuplicateExecution(
                    self.order_id, event.fill.execution_id, existing, event.fill
                )
            # Legality first: a fill against a terminal order is a lifecycle
            # violation, and reporting it as an arithmetic overflow would point
            # an operator at the quantity rather than at the dead order.
            if self.is_terminal:
                self._assert_transition(target)
            filled = self.filled_quantity + event.fill.quantity
            if filled > self.quantity:
                raise ValueError(
                    f"order {self.order_id}: fills total {filled} exceed order quantity {self.quantity}"
                )
            target = OrderStatus.FILLED if filled == self.quantity else OrderStatus.PARTIALLY_FILLED
            self._assert_transition(target)
            return replace(
                self, status=target, filled_quantity=filled, fills=(*self.fills, event.fill),
            )
        self._assert_transition(target)
        if target is OrderStatus.REJECTED:
            self._assert_rejectable(event)
        return replace(self, status=target, reason=event.reason or self.reason)

    def _assert_rejectable(self, event: OrderEvent) -> None:
        """A rejection may only erase an order that never traded.

        `ACCEPTED -> REJECTED` exists because a venue can acknowledge an order
        and then refuse it at fill time. It must not become a way to make
        executed quantity disappear: if anything filled, the fills are real
        money that already moved, and the correct disposition is to cancel the
        remainder. `PARTIALLY_FILLED -> REJECTED` is absent from the transition
        table for the same reason; this guard covers the case where a fill and a
        rejection race on an order still marked ACCEPTED.
        """
        if self.filled_quantity != 0 or self.fills:
            raise IllegalTransition(self.order_id, self.status, OrderStatus.REJECTED)
        if not (event.reason or "").strip():
            raise ValueError(
                f"order {self.order_id}: a rejection must carry an explicit reason"
            )

    def _assert_transition(self, target: OrderStatus) -> None:
        if target not in ALLOWED_TRANSITIONS[self.status]:
            raise IllegalTransition(self.order_id, self.status, target)

    @property
    def leaves_quantity(self) -> int:
        """Quantity still working. Zero once the order can no longer trade."""
        return 0 if self.is_terminal else self.remaining

    @property
    def last_quantity(self) -> int:
        """Quantity of the most recent execution, distinct from the cumulative."""
        return self.fills[-1].quantity if self.fills else 0

    @property
    def cumulative_quantity(self) -> int:
        """Explicit alias: kept separate from `status` and from `leaves_quantity`."""
        return self.filled_quantity


_TARGET_STATUS: Mapping[OrderEventType, OrderStatus] = {
    OrderEventType.CREATED: OrderStatus.PENDING_RISK,
    OrderEventType.RISK_APPROVED: OrderStatus.APPROVED,
    OrderEventType.RISK_REJECTED: OrderStatus.REJECTED,
    OrderEventType.SUBMITTED: OrderStatus.SUBMITTED,
    OrderEventType.ACCEPTED: OrderStatus.ACCEPTED,
    OrderEventType.PARTIAL_FILL: OrderStatus.PARTIALLY_FILLED,
    OrderEventType.FILL: OrderStatus.FILLED,
    OrderEventType.CANCEL_REQUESTED: OrderStatus.CANCEL_REQUESTED,
    OrderEventType.CANCELLED: OrderStatus.CANCELLED,
    OrderEventType.REJECTED: OrderStatus.REJECTED,
    OrderEventType.EXPIRED: OrderStatus.EXPIRED,
}


# ---------------------------------------------------------------------------
# Positions and cash
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PositionLot:
    """One dated parcel of shares. T+1 sellability is a property of the lot."""

    position_lot_id: str
    symbol: str
    quantity: int
    cost_price: float
    acquired_on: str
    lineage: Lineage = field(default_factory=Lineage)

    def sellable_on(self, trade_date: str) -> bool:
        """A-share T+1: a lot bought today cannot be sold today."""
        return trade_date > self.acquired_on

    @classmethod
    def from_fill(cls, fill: Fill, trade_date: str) -> "PositionLot":
        """Cost price is all-in: the fill price plus this fill's share of fees.

        A lot whose cost excluded entry fees reports a gain before the position
        has actually recovered what it cost to open, and it disagrees with the
        weighted-average basis a sale realises against (DEF-009).
        """
        lot_id = content_id("lot", execution_id=fill.execution_id)
        all_in = (fill.gross + fill.fees) / fill.quantity if fill.quantity else fill.price
        return cls(
            position_lot_id=lot_id, symbol=fill.symbol, quantity=fill.quantity,
            cost_price=all_in, acquired_on=trade_date,
            lineage=fill.lineage.derive(position_lot_id=lot_id),
        )


@dataclass(frozen=True, slots=True)
class CorporateAction:
    """A split, bonus issue or cash dividend. Economic, not administrative.

    Classified as operational telemetry until DEF-020 measured what that cost: a
    0.50/share dividend and a 2:1 split left the paper portfolio holding 2,000
    shares and 990,494.90 in cash while the canonical replay still said 1,000
    shares and 989,994.90. A dividend moves cash and a split changes share count,
    so both belong on the record of account like any fill.

    `share_ratio` scales the position and inversely scales cost basis, so a pure
    share adjustment moves no PnL — the price adjusts by the same factor on the ex
    date. `cash_per_share` is income and *does* move realised PnL, because the mark
    drops by the dividend on the ex date: without booking it as income the
    accounting identity would fail by exactly the amount received.
    """

    corporate_action_id: str
    symbol: str
    ex_date: str
    share_ratio: float = 1.0
    cash_per_share: float = 0.0
    lineage: Lineage = field(default_factory=Lineage)

    def __post_init__(self) -> None:
        if self.share_ratio <= 0:
            raise ValueError(
                f"share_ratio must be positive, got {self.share_ratio}: a "
                "non-positive ratio would delete or invert a position"
            )
        if self.cash_per_share < 0:
            raise ValueError(
                f"cash_per_share must not be negative, got {self.cash_per_share}: a "
                "negative dividend is a capital call, which this model does not have"
            )

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        ex_date: str,
        lineage: Lineage,
        share_ratio: float = 1.0,
        cash_per_share: float = 0.0,
    ) -> "CorporateAction":
        action_id = content_id(
            "ca",
            symbol=symbol,
            ex_date=ex_date,
            share_ratio=round(float(share_ratio), 9),
            cash_per_share=round(float(cash_per_share), 9),
            run_id=lineage.run_id,
        )
        return cls(
            corporate_action_id=action_id, symbol=symbol, ex_date=ex_date,
            share_ratio=float(share_ratio), cash_per_share=float(cash_per_share),
            lineage=lineage,
        )

    @property
    def moves_money(self) -> bool:
        return self.share_ratio != 1.0 or self.cash_per_share != 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "corporateActionId": self.corporate_action_id,
            "symbol": self.symbol,
            "exDate": self.ex_date,
            "shareRatio": self.share_ratio,
            "cashPerShare": self.cash_per_share,
            "lineage": self.lineage.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CorporateAction":
        return cls(
            corporate_action_id=str(payload["corporateActionId"]),
            symbol=str(payload["symbol"]),
            ex_date=str(payload["exDate"]),
            share_ratio=float(payload.get("shareRatio", 1.0)),
            cash_per_share=float(payload.get("cashPerShare", 0.0)),
            lineage=Lineage.from_mapping(payload.get("lineage")),
        )


@dataclass(frozen=True, slots=True)
class CashMovement:
    """Every cash effect, attributable to the fill that caused it."""

    amount: float
    reason: str
    occurred_at: str = field(default_factory=_now)
    lineage: Lineage = field(default_factory=Lineage)


def sellable_quantity(lots: Iterable[PositionLot], trade_date: str) -> int:
    return sum(lot.quantity for lot in lots if lot.sellable_on(trade_date))


def total_quantity(lots: Iterable[PositionLot]) -> int:
    return sum(lot.quantity for lot in lots)


class OrderBook:
    """Append-only order history.

    Holds every event ever applied. `state_of` folds them; `history_of` returns
    the raw record. Nothing is overwritten, so the audit trail is the source of
    truth rather than a derived log that can drift from it.
    """

    def __init__(self) -> None:
        self._orders: dict[str, Order] = {}
        self._events: list[OrderEvent] = []

    def open(self, intent: OrderIntent, *, parent_order_id: str | None = None) -> Order:
        order = Order.from_intent(intent, parent_order_id=parent_order_id)
        if order.order_id in self._orders:
            return self._orders[order.order_id]
        self._orders[order.order_id] = order
        self._events.append(
            OrderEvent(
                event_type=OrderEventType.CREATED, order_id=order.order_id,
                sequence=len(self._events), event_time=_now(), lineage=order.lineage,
            )
        )
        return order

    def apply(
        self,
        order_id: str,
        event_type: OrderEventType,
        *,
        fill: Fill | None = None,
        reason: str | None = None,
        risk_decision: RiskDecision | None = None,
    ) -> Order:
        order = self._orders[order_id]
        event = OrderEvent(
            event_type=event_type, order_id=order_id, sequence=len(self._events),
            event_time=_now(), fill=fill, reason=reason, risk_decision=risk_decision,
            lineage=order.lineage,
        )
        updated = order.apply(event)  # raises before the event is recorded
        if updated is order:
            # A re-delivered execution. Appending the event would make the log
            # claim two executions happened, and every projection built from it
            # would double-count — so the duplicate leaves no trace in history.
            return order
        self._events.append(event)
        self._orders[order_id] = updated
        return updated

    def state_of(self, order_id: str) -> Order:
        return self._orders[order_id]

    def history_of(self, order_id: str) -> tuple[OrderEvent, ...]:
        return tuple(event for event in self._events if event.order_id == order_id)

    def events(self) -> tuple[OrderEvent, ...]:
        return tuple(self._events)

    def orders(self) -> tuple[Order, ...]:
        return tuple(self._orders.values())

    def fills(self) -> tuple[Fill, ...]:
        return tuple(event.fill for event in self._events if event.fill is not None)

    def children_of(self, parent_order_id: str) -> tuple[Order, ...]:
        return tuple(o for o in self._orders.values() if o.parent_order_id == parent_order_id)


__all__ = [
    "ALLOWED_TRANSITIONS", "CashMovement", "CorporateAction", "DuplicateExecution", "Fill", "IllegalTransition",
    "Order", "OrderBook", "OrderEvent", "OrderEventType", "OrderIntent", "OrderStatus",
    "PositionLot", "RiskDecision", "Side", "Signal", "TERMINAL_STATUSES", "TargetPosition",
    "same_execution", "sellable_quantity", "total_quantity",
]
