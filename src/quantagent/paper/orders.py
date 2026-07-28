"""Order model for the local paper broker.

Deliberately absent: an unconstrained market order. On A-shares an order with no
price bound is how a backtest fills through a limit board and reports profit
that was never obtainable, so the most permissive type here is a
*marketable limit* -- it crosses the spread but still carries a worst price it
will not trade beyond.

Parent orders (TWAP/VWAP/POV) are schedules, not instructions to the exchange.
They slice into child limit orders, and only children ever reach the book, which
keeps participation and impact modelling honest.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence
from uuid import uuid4

BUY = "BUY"
SELL = "SELL"
SIDES: tuple[str, ...] = (BUY, SELL)

# --- order types ------------------------------------------------------------
LIMIT = "LIMIT"
#: Crosses the spread but carries a bounded worst price. The permissive end.
MARKETABLE_LIMIT = "MARKETABLE_LIMIT"
TWAP = "TWAP"
VWAP = "VWAP"
POV = "POV"

ORDER_TYPES: tuple[str, ...] = (LIMIT, MARKETABLE_LIMIT, TWAP, VWAP, POV)
#: Types that schedule child orders rather than resting on the book themselves.
PARENT_TYPES: frozenset[str] = frozenset({TWAP, VWAP, POV})

# --- order states -----------------------------------------------------------
NEW = "NEW"
ACCEPTED = "ACCEPTED"
PARTIALLY_FILLED = "PARTIALLY_FILLED"
FILLED = "FILLED"
CANCEL_REQUESTED = "CANCEL_REQUESTED"
CANCELLED = "CANCELLED"
REJECTED = "REJECTED"

ORDER_STATES: tuple[str, ...] = (
    NEW, ACCEPTED, PARTIALLY_FILLED, FILLED, CANCEL_REQUESTED, CANCELLED, REJECTED,
)
TERMINAL_STATES: frozenset[str] = frozenset({FILLED, CANCELLED, REJECTED})

#: Legal transitions. Enforced so a fill cannot arrive after a cancel, which is
#: the bug that silently inflates a simulated book.
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    NEW: frozenset({ACCEPTED, REJECTED}),
    ACCEPTED: frozenset({PARTIALLY_FILLED, FILLED, CANCEL_REQUESTED, CANCELLED, REJECTED}),
    PARTIALLY_FILLED: frozenset({PARTIALLY_FILLED, FILLED, CANCEL_REQUESTED, CANCELLED}),
    CANCEL_REQUESTED: frozenset({CANCELLED, FILLED, PARTIALLY_FILLED}),
    FILLED: frozenset(),
    CANCELLED: frozenset(),
    REJECTED: frozenset(),
}


class OrderStateError(RuntimeError):
    """Raised on an illegal order state transition."""


@dataclass
class Order:
    symbol: str
    side: str
    quantity: float
    order_type: str = LIMIT
    limit_price: float | None = None
    board: str = "SH_Main"
    order_id: str = field(default_factory=lambda: str(uuid4()))
    parent_id: str | None = None
    strategy_id: str | None = None
    state: str = NEW
    filled_quantity: float = 0.0
    filled_notional: float = 0.0
    fees_paid: float = 0.0
    reject_reason: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    is_full_liquidation: bool = False

    def __post_init__(self) -> None:
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {self.side!r}")
        if self.order_type not in ORDER_TYPES:
            raise ValueError(f"unknown order type {self.order_type!r}")
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.order_type in (LIMIT, MARKETABLE_LIMIT) and self.limit_price is None:
            raise ValueError(
                f"{self.order_type} requires a limit price; this broker exposes no "
                "unconstrained market order, because an unbounded fill on an "
                "A-share limit board reports profit that was never obtainable"
            )
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit price must be positive")

    # -- state machine ----------------------------------------------------
    def transition(self, new_state: str) -> None:
        if new_state not in ORDER_STATES:
            raise OrderStateError(f"unknown order state {new_state!r}")
        if new_state not in ALLOWED_TRANSITIONS[self.state]:
            raise OrderStateError(
                f"illegal transition {self.state} -> {new_state} for order "
                f"{self.order_id}"
            )
        self.state = new_state

    @property
    def remaining(self) -> float:
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def is_open(self) -> bool:
        return self.state not in TERMINAL_STATES

    @property
    def average_price(self) -> float | None:
        return (self.filled_notional / self.filled_quantity) if self.filled_quantity else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "remaining": self.remaining,
            "average_price": self.average_price,
            "is_open": self.is_open,
        }


@dataclass
class ParentOrder:
    """A execution schedule that emits child limit orders."""

    symbol: str
    side: str
    quantity: float
    order_type: str
    slices: int = 4
    participation_rate: float = 0.10
    board: str = "SH_Main"
    parent_id: str = field(default_factory=lambda: str(uuid4()))
    strategy_id: str | None = None

    def __post_init__(self) -> None:
        if self.order_type not in PARENT_TYPES:
            raise ValueError(
                f"{self.order_type!r} is not a parent order type; expected one of "
                f"{sorted(PARENT_TYPES)}"
            )
        if self.slices < 1:
            raise ValueError("slices must be >= 1")
        if not 0 < self.participation_rate <= 1:
            raise ValueError("participation_rate must be in (0, 1]")

    def schedule(
        self, *, reference_prices: Sequence[float],
        volumes: Sequence[float] | None = None,
    ) -> list[Order]:
        """Slice into child limit orders.

        TWAP splits evenly in time. VWAP weights by observed volume. POV sizes
        each slice as a fraction of the volume actually traded in that interval,
        so it can legitimately under-fill when the market is thin -- that is the
        point of the policy, not a defect.
        """
        if not reference_prices:
            return []
        count = min(self.slices, len(reference_prices))
        prices = list(reference_prices)[:count]

        if self.order_type == TWAP or not volumes:
            weights = [1.0 / count] * count
        elif self.order_type == VWAP:
            window = list(volumes)[:count]
            total = sum(window)
            weights = [v / total for v in window] if total > 0 else [1.0 / count] * count
        else:  # POV
            window = list(volumes)[:count]
            targets = [v * self.participation_rate for v in window]
            capped = min(sum(targets), self.quantity)
            weights = (
                [t / sum(targets) * (capped / self.quantity) for t in targets]
                if sum(targets) > 0 else [0.0] * count
            )

        children: list[Order] = []
        for price, weight in zip(prices, weights):
            size = self.quantity * weight
            if size <= 0:
                continue
            children.append(Order(
                symbol=self.symbol, side=self.side, quantity=size,
                order_type=MARKETABLE_LIMIT, limit_price=price,
                board=self.board, parent_id=self.parent_id,
                strategy_id=self.strategy_id,
            ))
        return children

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    notional: float
    commission: float
    stamp_duty: float
    transfer_fee: float
    market_time: str | None = None
    fill_id: str = field(default_factory=lambda: str(uuid4()))
    partial: bool = False

    @property
    def fees(self) -> float:
        return self.commission + self.stamp_duty + self.transfer_fee

    @property
    def cash_delta(self) -> float:
        """Signed cash effect: a buy consumes notional plus fees."""
        if self.side == BUY:
            return -(self.notional + self.fees)
        return self.notional - self.fees

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"fees": self.fees, "cash_delta": self.cash_delta}
