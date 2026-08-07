"""Every economically meaningful figure, derived from canonical events alone.

Nothing here reads an engine's own cash field, portfolio object or trade
dataframe. A snapshot is a pure function of `(OrderBook, AccountState)` as
rebuilt by `CanonicalLedger.replay`, which is what makes "engine A agrees with
engine B" a statement about the record of account rather than about two
independently maintained sets of numbers.

Two derivations deserve calling out, because they are the reason the dimensions
below are comparable at all:

* **Reserved cash and reserved inventory are projections of the order book.**
  A working buy order commits cash; a working sell order commits shares. Storing
  those as a mutable balance would create exactly the second record of truth
  Module One exists to remove, so they are recomputed from open orders every
  time — which also means the commitment is released exactly once, because it is
  never decremented. `AccountState` deliberately has no `frozen_cash` field:
  it had one that no event fed, so it read 0.0 forever and every comparison
  against it passed without measuring anything (DEF-007, now closed by deleting
  the field).

* **Orders are keyed logically, not by id.** Order ids are content-addressed
  over lineage, so the same economic order carries different ids in two engines.
  Cross-engine comparison therefore keys on
  `(symbol, side, quantity, trade_date)` plus an occurrence index, and the id is
  reported as detail rather than compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

from quantagent.domain.accounting import AccountState
from quantagent.domain.orders import (
    Order,
    OrderBook,
    OrderEvent,
    OrderStatus,
    Side,
    TERMINAL_STATUSES,
)

#: An order commits cash or inventory from the moment it reaches the venue until
#: it can no longer trade. Before SUBMITTED it is still an internal intent, so
#: reserving against it would overstate committed capital; after a terminal
#: status the commitment is released exactly once because it is recomputed
#: rather than decremented.
RESERVING_STATUSES: frozenset[OrderStatus] = frozenset(
    {
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.CANCEL_REQUESTED,
    }
)


def _round(value: float, places: int = 6) -> float:
    return round(float(value), places)


@dataclass(frozen=True, slots=True)
class OrderFacts:
    """Per-order figures that must survive replay and match across engines."""

    logical_key: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    status: str
    cumulative_quantity: int
    last_quantity: int
    leaves_quantity: int
    fill_count: int
    average_fill_price: float | None
    fees: float
    reason: str | None
    #: Event purposes in the order they were recorded, e.g.
    #: ("CREATED", "RISK_APPROVED", "SUBMITTED", "ACCEPTED", "FILL").
    event_sequence: tuple[str, ...]
    #: Which lineage links are present. A missing link breaks drill-down.
    lineage_links: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "logicalKey": self.logical_key,
            "orderId": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "status": self.status,
            "cumulativeQuantity": self.cumulative_quantity,
            "lastQuantity": self.last_quantity,
            "leavesQuantity": self.leaves_quantity,
            "fillCount": self.fill_count,
            "averageFillPrice": self.average_fill_price,
            "fees": _round(self.fees),
            "reason": self.reason,
            "eventSequence": list(self.event_sequence),
            "lineageLinks": list(self.lineage_links),
        }


_LINEAGE_LINKS = (
    "run_id",
    "strategy_version_id",
    "signal_id",
    "order_intent_id",
    "order_id",
    "execution_id",
)


def _lineage_links(order: Order) -> tuple[str, ...]:
    present = [name for name in _LINEAGE_LINKS if getattr(order.lineage, name, None)]
    for fill in order.fills:
        if fill.lineage.execution_id and "execution_id" not in present:
            present.append("execution_id")
    return tuple(present)


@dataclass(frozen=True, slots=True)
class EconomicSnapshot:
    """The complete economic picture of one run, as folded from its event log."""

    label: str
    session: str
    #: -- cash ------------------------------------------------------------
    cash: float
    reserved_cash: float
    available_cash: float
    #: -- pnl -------------------------------------------------------------
    realised_pnl: float
    unrealised_pnl: float
    market_value: float
    nav: float
    #: -- costs, itemised so a double charge cannot hide in a total -------
    commission: float
    stamp_duty: float
    transfer_fee: float
    fees_total: float
    slippage: float
    #: -- inventory -------------------------------------------------------
    positions: Mapping[str, int]
    settled_inventory: Mapping[str, int]
    sellable_inventory: Mapping[str, int]
    reserved_inventory: Mapping[str, int]
    lots: Mapping[str, tuple[tuple[Any, ...], ...]]
    #: -- orders ----------------------------------------------------------
    orders: Mapping[str, OrderFacts]
    counts: Mapping[str, int]
    event_sequence: tuple[tuple[str, str], ...]
    lineage_gaps: tuple[str, ...]
    #: `realised + unrealised - (NAV - initial cash)`. Each path must satisfy this
    #: on its own, so it is reported as evidence rather than diffed across paths —
    #: two engines both satisfying it to float precision are both correct, and
    #: comparing their residuals would only compare rounding noise.
    identity_residual: float = 0.0
    #: Execution ids the log repeated and replay refused to apply twice.
    duplicate_executions: tuple[str, ...] = ()

    # -- construction -------------------------------------------------------
    @classmethod
    def from_replay(
        cls,
        label: str,
        book: OrderBook,
        account: AccountState,
        *,
        session: str,
        prices: Mapping[str, float],
    ) -> "EconomicSnapshot":
        orders = _logical_orders(book)
        reserved_cash = 0.0
        reserved_inventory: dict[str, int] = {}
        for order in book.orders():
            if order.status not in RESERVING_STATUSES:
                continue
            if order.side is Side.BUY:
                # Reservation basis, in order of preference: the order's own price
                # bound, the price the intent was sized against, then whatever it
                # has actually filled at. Fees are not reserved; that convention is
                # stated here rather than being an accident of whichever cost model
                # happened to be configured. An order priced only by reference used
                # to fall through to 0.0 and report a working order as committing
                # nothing.
                basis = (
                    order.limit_price
                    or order.reference_price
                    or order.average_fill_price
                    or 0.0
                )
                reserved_cash += order.leaves_quantity * float(basis)
            else:
                reserved_inventory[order.symbol] = (
                    reserved_inventory.get(order.symbol, 0) + order.leaves_quantity
                )

        commission = sum(fill.commission for fill in book.fills())
        stamp_duty = sum(fill.stamp_duty for fill in book.fills())
        transfer_fee = sum(fill.transfer_fee for fill in book.fills())
        slippage = sum(fill.slippage for fill in book.fills())

        positions = {symbol: account.position(symbol) for symbol in sorted(account.lots)}
        settled = {
            symbol: account.sellable(symbol, session) for symbol in sorted(account.lots)
        }
        sellable = {
            symbol: max(0, settled.get(symbol, 0) - reserved_inventory.get(symbol, 0))
            for symbol in sorted(account.lots)
        }
        lots = {
            symbol: tuple(
                sorted(
                    (lot.quantity, _round(lot.cost_price), lot.acquired_on)
                    for lot in account.lots[symbol]
                )
            )
            for symbol in sorted(account.lots)
        }

        statuses = [order.status for order in book.orders()]
        counts = {
            "orders": len(statuses),
            "fills": len(book.fills()),
            "filled": sum(1 for s in statuses if s is OrderStatus.FILLED),
            "partially_filled": sum(
                1 for s in statuses if s is OrderStatus.PARTIALLY_FILLED
            ),
            "rejected": sum(1 for s in statuses if s is OrderStatus.REJECTED),
            "cancelled": sum(1 for s in statuses if s is OrderStatus.CANCELLED),
            "expired": sum(1 for s in statuses if s is OrderStatus.EXPIRED),
            "working": sum(1 for s in statuses if s not in TERMINAL_STATUSES),
        }

        gaps = tuple(
            f"{facts.logical_key}:{name}"
            for facts in orders.values()
            for name in ("run_id", "signal_id", "order_intent_id", "order_id")
            if name not in facts.lineage_links
        )

        return cls(
            label=label,
            session=session,
            cash=_round(account.cash),
            reserved_cash=_round(reserved_cash),
            available_cash=_round(account.cash - reserved_cash),
            realised_pnl=_round(account.realised_pnl),
            unrealised_pnl=_round(account.unrealised_pnl(prices)),
            market_value=_round(account.market_value(prices)),
            nav=_round(account.nav(prices)),
            commission=_round(commission),
            stamp_duty=_round(stamp_duty),
            transfer_fee=_round(transfer_fee),
            fees_total=_round(commission + stamp_duty + transfer_fee),
            slippage=_round(slippage),
            positions=positions,
            settled_inventory=settled,
            sellable_inventory=sellable,
            reserved_inventory=dict(sorted(reserved_inventory.items())),
            lots=lots,
            orders=orders,
            counts=counts,
            event_sequence=tuple(
                (facts.logical_key, event)
                for facts in orders.values()
                for event in facts.event_sequence
            ),
            lineage_gaps=gaps,
            identity_residual=account.identity_residual(prices),
            duplicate_executions=tuple(account.duplicate_executions),
        )

    # -- comparison surface -------------------------------------------------
    def flatten(self) -> dict[str, Any]:
        """One flat dimension -> value mapping, which is what gets diffed."""
        flat: dict[str, Any] = {
            "cash": self.cash,
            "reserved_cash": self.reserved_cash,
            "available_cash": self.available_cash,
            "realised_pnl": self.realised_pnl,
            "unrealised_pnl": self.unrealised_pnl,
            "market_value": self.market_value,
            "nav": self.nav,
            "commission": self.commission,
            "stamp_duty": self.stamp_duty,
            "transfer_fee": self.transfer_fee,
            "fees_total": self.fees_total,
            "slippage": self.slippage,
            "lineage_gaps": list(self.lineage_gaps),
        }
        for name, mapping in (
            ("position", self.positions),
            ("settled_inventory", self.settled_inventory),
            ("sellable_inventory", self.sellable_inventory),
            ("reserved_inventory", self.reserved_inventory),
        ):
            for symbol, value in mapping.items():
                flat[f"{name}[{symbol}]"] = value
        for symbol, parcels in self.lots.items():
            flat[f"lots[{symbol}]"] = [list(parcel) for parcel in parcels]
        for name, value in self.counts.items():
            flat[f"count[{name}]"] = value
        for key, facts in self.orders.items():
            flat[f"order[{key}].status"] = facts.status
            flat[f"order[{key}].cumulative_quantity"] = facts.cumulative_quantity
            flat[f"order[{key}].last_quantity"] = facts.last_quantity
            flat[f"order[{key}].leaves_quantity"] = facts.leaves_quantity
            flat[f"order[{key}].fill_count"] = facts.fill_count
            flat[f"order[{key}].fees"] = _round(facts.fees)
            flat[f"order[{key}].event_sequence"] = list(facts.event_sequence)
            flat[f"order[{key}].lineage_links"] = list(facts.lineage_links)
        return flat

    def content_hash(self) -> str:
        payload = json.dumps(self.flatten(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "session": self.session,
            "cash": self.cash,
            "reservedCash": self.reserved_cash,
            "availableCash": self.available_cash,
            "realisedPnl": self.realised_pnl,
            "unrealisedPnl": self.unrealised_pnl,
            "marketValue": self.market_value,
            "nav": self.nav,
            "commission": self.commission,
            "stampDuty": self.stamp_duty,
            "transferFee": self.transfer_fee,
            "feesTotal": self.fees_total,
            "slippage": self.slippage,
            "positions": dict(self.positions),
            "settledInventory": dict(self.settled_inventory),
            "sellableInventory": dict(self.sellable_inventory),
            "reservedInventory": dict(self.reserved_inventory),
            "lots": {s: [list(p) for p in parcels] for s, parcels in self.lots.items()},
            "counts": dict(self.counts),
            "orders": {k: v.to_dict() for k, v in self.orders.items()},
            "lineageGaps": list(self.lineage_gaps),
            "identityResidual": self.identity_residual,
            "duplicateExecutions": list(self.duplicate_executions),
            "contentHash": self.content_hash(),
        }


def _logical_orders(book: OrderBook) -> dict[str, OrderFacts]:
    """Key every order by its economics so two engines can be lined up.

    Occurrence index disambiguates genuinely repeated economics (the same buy
    placed twice in one session is two orders, and collapsing them would hide a
    duplicate rather than reveal it).
    """
    seen: dict[str, int] = {}
    facts: dict[str, OrderFacts] = {}
    for order in book.orders():
        base = f"{order.symbol}|{order.side.value}|{order.quantity}|{order.trade_date}"
        index = seen.get(base, 0)
        seen[base] = index + 1
        key = base if index == 0 else f"{base}#{index}"
        history = book.history_of(order.order_id)
        facts[key] = OrderFacts(
            logical_key=key,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            status=order.status.value,
            cumulative_quantity=order.cumulative_quantity,
            last_quantity=order.last_quantity,
            leaves_quantity=order.leaves_quantity,
            fill_count=len(order.fills),
            average_fill_price=(
                _round(order.average_fill_price)
                if order.average_fill_price is not None
                else None
            ),
            fees=order.total_fees,
            reason=order.reason,
            event_sequence=tuple(event.event_type.value for event in history),
            lineage_links=_lineage_links(order),
        )
    return dict(sorted(facts.items()))


__all__ = ["EconomicSnapshot", "OrderFacts", "RESERVING_STATUSES"]
