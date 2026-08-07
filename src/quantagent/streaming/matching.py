"""The streaming matcher: it decides its own fills, or reconciliation is circular.

A streaming engine handed the fills a venue already produced will always agree with
that venue. That agreement is worth nothing — it compares a recording to itself. So
this matcher derives fills from market events: it reads a `BAR`, decides for each
working order whether it trades, at what quantity and at what price, and publishes
`VENUE_CALLBACK` and `FILL` events for `OrderLifecycle` to fold onto the canonical
ledger.

What is shared with the paper broker, and what is not, is the whole design:

* **Shared: the rules.** Price bands, tradability, lot rounding, transaction costs
  *and the fill-price formula* all come from `backtest.ashare_rules`. The pricing
  formula was briefly duplicated here, which would have made agreement with the
  paper broker a coincidence maintained by hand rather than a property. Two implementations of the A-share
  rulebook would drift, and the drift would show up as a reconciliation difference
  nobody could resolve — one of them would be right and there would be no way to
  say which.
* **Not shared: the control flow.** Paper validates and fills inside one
  synchronous `submit`. This matcher is driven by events, holds orders across bars,
  and answers the venue's questions as separate events. That is the difference
  reconciliation is meant to test: if an event-driven flow over the same rulebook
  produces a different number than a synchronous one, one of them has a bug.

The consequence to expect, stated up front: streaming and paper *should* agree
exactly, because the rulebook is the contract. Any difference between them is a
control-flow defect, not a modelling choice — unlike the fast engine, which
legitimately differs on price because it models fills with a bar column rather than
a participation model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from quantagent.backtest import ashare_rules as rules
from quantagent.domain.orders import Order, OrderStatus, Side
from quantagent.domain.timeline import EventTime
from quantagent.streaming.bus import EventBus
from quantagent.streaming.events import EventKind, MarketEvent
from quantagent.streaming.lifecycle import OrderLifecycle


@dataclass(frozen=True)
class MatcherConfig:
    """Deliberately mirrors `paper.broker.BrokerConfig` field for field.

    The two are given identical values in reconciliation, so a difference in
    results cannot be attributed to a difference in parameters.
    """

    participation_cap: float = 0.10
    commission_rate: float = rules.DEFAULT_COMMISSION_RATE
    slippage_bps: float = 5.0
    impact_coefficient: float = 0.10
    allow_st_buy: bool = False


@dataclass
class MatchingVenue:
    """Answers order events from market events. Holds no economic state.

    Working orders are read from the `OrderLifecycle`'s book — which is itself a
    projection of the ledger — rather than kept in a local dict, for the same
    reason the lifecycle keeps none: a second list of open orders is a second
    record of what is working, and the two drift.
    """

    lifecycle: OrderLifecycle
    bus: EventBus
    config: MatcherConfig = field(default_factory=MatcherConfig)
    #: Available settled inventory per symbol, replayed on demand from the ledger.
    #: Cached only within one event, never across.
    _executions: int = field(default=0, init=False)

    @property
    def executions(self) -> int:
        return self._executions

    # -- the consumer -------------------------------------------------------
    def __call__(self, event: MarketEvent, frontier: Any = None) -> None:
        self.handle(event)

    def handle(self, event: MarketEvent) -> None:
        if event.kind is EventKind.ORDER:
            self._acknowledge(event)
        elif event.kind is EventKind.BAR:
            self._match_against(event)

    # -- venue behaviour ----------------------------------------------------
    def _acknowledge(self, event: MarketEvent) -> None:
        """Accept or refuse a new order, using the rulebook rather than a guess."""
        client_order_id = str(event.payload["clientOrderId"])
        side = str(event.payload["side"]).upper()
        session = event.times.event_time.date().isoformat()
        band = self._band(event)
        limit_price = event.payload.get("limitPrice")

        reason = None
        quantity = rules.round_to_lot(
            float(event.payload["quantity"]),
            board=str(event.payload.get("board", "SH_Main")),
            side=side,
            is_full_liquidation=bool(event.payload.get("isFullLiquidation", False)),
        )
        if quantity <= 0:
            reason = "quantity rounds below the board minimum lot"
        elif limit_price is not None and band is not None and not band.unlimited:
            if band.limit_up is not None and float(limit_price) > band.limit_up + 1e-9:
                reason = f"limit {limit_price} exceeds the price ceiling {band.limit_up}"
            elif band.limit_down is not None and float(limit_price) < band.limit_down - 1e-9:
                reason = f"limit {limit_price} is below the price floor {band.limit_down}"
        if reason is None and side == "SELL":
            sellable = self._sellable(str(event.symbol), session)
            if quantity - sellable > 1e-9:
                reason = (
                    f"sell of {quantity:.0f} exceeds the T+1-settled {sellable:.0f}"
                )

        self._reply(
            event, client_order_id,
            status="REJECTED" if reason else "ACCEPTED",
            reason=reason,
        )

    def _match_against(self, bar: MarketEvent) -> None:
        """Fill whatever the bar allows, for every order still working on it."""
        for order in self._working(str(bar.symbol)):
            self._try_fill(order, bar)

    def _try_fill(self, order: Order, bar: MarketEvent) -> None:
        last_price = float(bar.payload["close"])
        volume = float(bar.payload.get("volume", 0.0))
        board = str(bar.payload.get("board", "SH_Main"))
        session = bar.times.event_time.date().isoformat()

        verdict = rules.tradability(
            is_suspended=bool(bar.payload.get("isSuspended", False)),
            at_limit_up=self._at_limit(bar, last_price, upper=True),
            at_limit_down=self._at_limit(bar, last_price, upper=False),
        )
        if order.side is Side.BUY and not verdict.can_buy:
            return
        if order.side is Side.SELL and not verdict.can_sell:
            return

        available = volume * self.config.participation_cap
        quantity = rules.round_to_lot(
            min(order.leaves_quantity, available), board=board, side=order.side.value
        )
        if quantity <= 0:
            return
        # A resting limit that the market has not reached does not trade. Without
        # this the matcher would fill a buy below the market and book a price the
        # market never offered.
        if order.limit_price is not None:
            if order.side is Side.BUY and last_price > order.limit_price:
                return
            if order.side is Side.SELL and last_price < order.limit_price:
                return

        price = self._execution_price(order, bar, quantity, last_price)
        costs = rules.trading_costs(
            notional_cny=quantity * price,
            side=order.side.value,
            trade_date=session,
            commission_rate=self.config.commission_rate,
        )

        # DEF-019: without this the matcher booked whatever the participation cap
        # allowed and let cash go negative — a streaming run would report fills the
        # account could never have funded, and the accounting layer would not object
        # because a negative balance is not one of its invariants. Refused at fill
        # time rather than at acknowledgement, because that is when the price and
        # therefore the cost are known.
        shortfall = self._funding_shortfall(order, quantity, price, costs, session)
        if shortfall is not None:
            self._reply(bar, self._client_order_id(order), status="REJECTED", reason=shortfall)
            return

        self._executions += 1
        execution_id = f"strm-{self._executions:06d}"
        client_order_id = self._client_order_id(order)
        self.bus.publish(
            MarketEvent(
                kind=EventKind.FILL,
                times=EventTime.immediate(bar.times.event_time),
                symbol=order.symbol,
                sequence=self._executions,
                payload={
                    "clientOrderId": client_order_id,
                    "executionId": execution_id,
                    "quantity": int(quantity),
                    "price": price,
                    "commission": costs.commission,
                    "stampDuty": costs.stamp_duty,
                    "transferFee": costs.transfer_fee,
                },
            )
        )

    def _execution_price(
        self, order: Order, bar: MarketEvent, quantity: int, last_price: float
    ) -> float:
        """The shared pricing formula, not a second copy of it.

        This is deliberately a delegation. If the matcher carried its own copy, its
        agreement with the paper broker would be a coincidence maintained by hand,
        and the reconciliation result would say nothing about control flow — which
        is the only thing the two implementations genuinely differ on.
        """
        return rules.execution_price(
            last_price=last_price,
            side=order.side.value,
            quantity=quantity,
            session_volume=float(bar.payload.get("volume", 0.0)),
            slippage_bps=self.config.slippage_bps,
            impact_coefficient=self.config.impact_coefficient,
            limits=self._band(bar),
            limit_price=order.limit_price,
        )

    def _funding_shortfall(
        self,
        order: Order,
        quantity: int,
        price: float,
        costs: rules.TradingCosts,
        session: str,
    ) -> str | None:
        """Why this fill cannot be funded, or None when it can.

        Read from a replay rather than a running balance: the ledger is the account,
        and a matcher keeping its own cash figure would be the second record of truth
        that Module One exists to remove.
        """
        account = self.lifecycle.account()
        if order.side is Side.BUY:
            needed = quantity * price + costs.total
            if needed - account.cash > 1e-9:
                return (
                    f"buy of {quantity} needs {needed:.2f} but only "
                    f"{account.cash:.2f} is available"
                )
            return None
        available = account.sellable(order.symbol, session)
        if quantity - available > 1e-9:
            return (
                f"sell of {quantity} exceeds the T+1-settled {available}"
            )
        return None

    # -- helpers ------------------------------------------------------------
    def _band(self, event: MarketEvent) -> rules.PriceLimits | None:
        previous_close = event.payload.get("previousClose")
        if previous_close is None:
            return None
        return rules.price_limits(
            board=str(event.payload.get("board", "SH_Main")),
            previous_close=float(previous_close),
            trade_date=event.times.event_time.date().isoformat(),
            is_st=bool(event.payload.get("isSt", False)),
        )

    def _at_limit(self, bar: MarketEvent, last_price: float, *, upper: bool) -> bool:
        band = self._band(bar)
        if band is None or band.unlimited:
            return False
        edge = band.limit_up if upper else band.limit_down
        if edge is None:
            return False
        return last_price >= edge - 1e-9 if upper else last_price <= edge + 1e-9

    def _working(self, symbol: str) -> list[Order]:
        return [
            order
            for order in self.lifecycle.book.orders()
            if order.symbol == symbol
            and order.status in {OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED}
        ]

    def _sellable(self, symbol: str, session: str) -> int:
        """Settled inventory, replayed from the chain rather than tracked here."""
        return self.lifecycle.account().sellable(symbol, session)

    def _client_order_id(self, order: Order) -> str:
        for client_order_id, canonical_id in self.lifecycle._canonical_ids.items():
            if canonical_id == order.order_id:
                return client_order_id
        raise KeyError(f"no client order id maps to canonical order {order.order_id}")

    def _reply(
        self, event: MarketEvent, client_order_id: str, *, status: str, reason: str | None
    ) -> None:
        self.bus.publish(
            MarketEvent(
                kind=EventKind.VENUE_CALLBACK,
                times=EventTime.immediate(event.times.event_time),
                symbol=event.symbol,
                sequence=event.sequence,
                payload={
                    "clientOrderId": client_order_id,
                    "status": status,
                    **({"reason": reason} if reason else {}),
                },
            )
        )


__all__ = ["MatcherConfig", "MatchingVenue"]
