"""Wire adapter letting `OrderManager` route to the local paper broker.

Before this existed the OMS could only reach `VirtualBroker`, so the production
chain *intent -> risk -> OMS -> venue -> canonical ledger* had no paper leg and
could not be reconciled against a paper run. That gap is why Module One's
composite replay could not be measured.

Two properties keep it a wire adapter rather than a second engine:

* **It holds no economic state.** Cash, positions, fills and order state live in
  the paper broker and in the canonical ledger. Everything this class returns is
  translated on the spot from `PaperBroker`'s own objects.
* **It does not open canonical orders.** The OMS opens the order; this adapter
  hands the canonical id to the venue via `attach_canonical` so the venue appends
  ACCEPTED/FILL/REJECTED to *that* order. One economic order, one record.

The market snapshot is supplied by the caller. An adapter that invented its own
price would make the venue's fill price a property of the adapter rather than of
the market data the run was given.
"""

from __future__ import annotations

from typing import Callable, Mapping

from quantagent.execution.broker_base import (
    BrokerBase,
    Order,
    OrderSide,
    OrderState,
    OrderStatus,
    Position,
)
from quantagent.paper import orders as po
from quantagent.paper.broker import MarketSnapshot, PaperBroker

#: Paper's state vocabulary in the OMS wire vocabulary. Paper's NEW means
#: "at the venue, unacknowledged", which the wire calls SUBMITTED.
_TO_WIRE: Mapping[str, OrderStatus] = {
    po.NEW: OrderStatus.SUBMITTED,
    po.ACCEPTED: OrderStatus.SUBMITTED,
    po.PARTIALLY_FILLED: OrderStatus.PARTIAL,
    po.FILLED: OrderStatus.FILLED,
    po.CANCEL_REQUESTED: OrderStatus.SUBMITTED,
    po.CANCELLED: OrderStatus.CANCELLED,
    po.REJECTED: OrderStatus.REJECTED,
}

MarketSource = Callable[[str, str], MarketSnapshot]


class PaperBrokerAdapter(BrokerBase):
    """`BrokerBase` in front of `PaperBroker`. Translation only, no bookkeeping."""

    #: Read by `assert_forensic_isolation`: this adapter can reach paper
    #: execution, so a forensic replay must never be pointed at it.
    is_local_simulation = True
    has_broker_connection = False

    def __init__(self, broker: PaperBroker, market_source: MarketSource) -> None:
        self.broker = broker
        self.market_source = market_source
        #: client_order_id -> paper order id. A translation table, not state:
        #: every economic figure is read back out of the paper broker.
        self._paper_ids: dict[str, str] = {}
        #: client_order_id -> canonical order id the OMS already opened.
        self._pending_canonical: dict[str, str] = {}
        self._trade_callbacks: list[Callable[..., None]] = []
        self._last_prices: dict[str, float] = {}

    # -- canonical handshake -------------------------------------------------
    def attach_canonical(self, client_order_id: str, canonical_order_id: str) -> None:
        """Remember which canonical order the next submission belongs to."""
        self._pending_canonical[client_order_id] = canonical_order_id

    # -- BrokerBase ----------------------------------------------------------
    def submit(self, order: Order) -> OrderState:
        trade_date = str(order.timestamp)[:10]
        market = self.market_source(order.symbol, trade_date)
        self._last_prices[order.symbol] = market.last_price
        paper_order = po.Order(
            symbol=order.symbol,
            side=po.BUY if order.side is OrderSide.BUY else po.SELL,
            quantity=float(order.quantity),
            order_type=po.LIMIT,
            limit_price=self._price_bound(order, market),
            board=market.board,
            strategy_id=order.strategy_version or None,
        )
        self._paper_ids[order.client_order_id] = paper_order.order_id
        canonical_order_id = self._pending_canonical.pop(order.client_order_id, None)
        if canonical_order_id is not None:
            self.broker.attach_canonical(paper_order.order_id, canonical_order_id)
        settled = self.broker.submit(paper_order, market)
        return self._state_of(order.client_order_id, settled)

    def cancel(self, client_order_id: str) -> OrderState:
        paper_id = self._paper_ids[client_order_id]
        settled = self.broker.cancel(paper_id)
        return self._state_of(client_order_id, settled)

    def query_order(self, client_order_id: str) -> OrderState:
        paper_id = self._paper_ids[client_order_id]
        return self._state_of(client_order_id, self.broker.orders[paper_id])

    def query_positions(self) -> list[Position]:
        return [
            Position(
                symbol=symbol,
                available_shares=int(position.sellable),
                frozen_shares=int(position.pending_settlement),
                avg_cost=float(position.average_cost),
            )
            for symbol, position in self.broker.portfolio.positions.items()
            if not position.is_flat
        ]

    def query_account_value(self) -> float:
        return float(self.broker.portfolio.equity(self._last_prices))

    def on_trade(self, callback) -> None:
        self._trade_callbacks.append(callback)

    # -- translation ---------------------------------------------------------
    @staticmethod
    def _price_bound(order: Order, market: MarketSnapshot) -> float:
        """Every paper order carries a worst price; there is no market order.

        A missing OMS price becomes a bound derived from the snapshot rather than
        an unbounded order, because an unbounded fill on a limit board books
        profit that was never obtainable.
        """
        if order.price:
            return float(order.price)
        limits = market.limits()
        if order.side is OrderSide.BUY:
            ceiling = limits.limit_up or market.last_price * 1.1
            return float(min(market.last_price * 1.01, ceiling))
        floor = limits.limit_down or market.last_price * 0.9
        return float(max(market.last_price * 0.99, floor))

    def _state_of(self, client_order_id: str, paper_order: po.Order) -> OrderState:
        average = (
            paper_order.filled_notional / paper_order.filled_quantity
            if paper_order.filled_quantity
            else 0.0
        )
        return OrderState(
            client_order_id=client_order_id,
            broker_order_id=paper_order.order_id,
            status=_TO_WIRE[paper_order.state],
            filled_quantity=int(paper_order.filled_quantity),
            avg_price=float(average),
            last_message=paper_order.reject_reason or "",
        )


__all__ = ["PaperBrokerAdapter"]
