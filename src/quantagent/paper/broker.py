"""Deterministic local paper broker.

Entirely local. There is no connector, no credential, no account id and no
network call anywhere in this module -- a simulated order cannot leave the
process because there is nothing to leave through.

Determinism is a property, not an aspiration: given the same ledger, the same
market data and the same seed, the sequence of fills is identical. That is what
makes historical replay reproducible and what lets a recovery test assert that
replaying the ledger reconstructs exactly the state it recorded.

Fidelity is bounded by the data. With daily bars the broker models participation
against session volume and never claims queue position; the mission's rule is
respected by refusing to expose a queue-based fill model unless order-level data
is supplied. A-share rules -- T+1, board lots, price limits, ST bands,
suspensions, session phases, fees -- come from
:mod:`quantagent.backtest.ashare_rules` rather than being re-implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from quantagent.backtest import ashare_rules as rules
from quantagent.data.microstructure import contracts as mc
from quantagent.paper import ledger as lg
from quantagent.paper.orders import (
    ACCEPTED,
    BUY,
    CANCEL_REQUESTED,
    CANCELLED,
    FILLED,
    LIMIT,
    MARKETABLE_LIMIT,
    PARTIALLY_FILLED,
    REJECTED,
    SELL,
    Fill,
    Order,
    ParentOrder,
)
from quantagent.paper.portfolio import (
    InsufficientCash,
    InsufficientSellable,
    Portfolio,
)


@dataclass
class MarketSnapshot:
    """What the broker knows about one symbol at one point in time."""

    symbol: str
    trade_date: str
    last_price: float
    previous_close: float
    session_volume: float
    board: str = "SH_Main"
    clock: str = "10:00:00"
    is_suspended: bool = False
    is_st: bool = False
    sessions_since_listing: int | None = None
    high: float | None = None
    low: float | None = None

    def limits(self) -> rules.PriceLimits:
        return rules.price_limits(
            board=self.board, previous_close=self.previous_close,
            trade_date=self.trade_date, is_st=self.is_st,
            sessions_since_listing=self.sessions_since_listing,
        )

    @property
    def at_limit_up(self) -> bool:
        limits = self.limits()
        return limits.limit_up is not None and self.last_price >= limits.limit_up - 1e-9

    @property
    def at_limit_down(self) -> bool:
        limits = self.limits()
        return limits.limit_down is not None and self.last_price <= limits.limit_down + 1e-9

    @property
    def phase(self) -> str:
        return mc.session_phase(self.clock[:5], board=self.board)


@dataclass
class BrokerConfig:
    participation_cap: float = 0.10
    commission_rate: float = rules.DEFAULT_COMMISSION_RATE
    slippage_bps: float = 5.0
    #: Square-root impact coefficient, applied to the participation fraction.
    impact_coefficient: float = 0.10
    allow_st_buy: bool = False


class PaperBroker:
    """Local order simulator writing every state change to the event ledger."""

    #: Named so a reader cannot mistake this for a venue connector.
    is_local_simulation = True
    has_broker_connection = False

    def __init__(
        self,
        portfolio: Portfolio,
        event_ledger: lg.EventLedger,
        *,
        run_id: str,
        config: BrokerConfig | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.ledger = event_ledger
        self.run_id = run_id
        self.config = config or BrokerConfig()
        self.orders: dict[str, Order] = {}
        self.fills: list[Fill] = []
        self.killed: bool = False
        self.kill_reason: str | None = None

    # -- ledger helper -----------------------------------------------------
    def _emit(self, event_type: str, payload: Mapping[str, Any], *,
              symbol: str | None = None, market_time: str | None = None) -> None:
        self.ledger.append(
            event_type, run_id=self.run_id,
            portfolio_id=self.portfolio.portfolio_id,
            payload=payload, symbol=symbol, market_time=market_time,
        )

    # -- kill switch -------------------------------------------------------
    def arm_kill_switch(self, scope: str, reason: str) -> None:
        self._emit(lg.KILL_SWITCH_ARMED, {"scope": scope, "reason": reason})

    def trigger_kill_switch(self, reason: str, *, scope: str = "GLOBAL") -> None:
        self.killed = True
        self.kill_reason = reason
        self._emit(lg.KILL_SWITCH_TRIGGERED, {"scope": scope, "reason": reason})

    # -- validation --------------------------------------------------------
    def _reject(self, order: Order, reason: str, market: MarketSnapshot | None) -> Order:
        order.reject_reason = reason
        order.transition(REJECTED)
        self._emit(lg.ORDER_REJECTED,
                   {"order": order.to_dict(), "reason": reason},
                   symbol=order.symbol,
                   market_time=market.clock if market else None)
        return order

    def _validate(self, order: Order, market: MarketSnapshot) -> str | None:
        """Return a rejection reason, or None when the order may proceed."""
        if self.killed:
            return f"kill switch active: {self.kill_reason}"

        if market.phase not in mc.CONTINUOUS_PHASES and market.phase not in mc.AUCTION_PHASES:
            return f"outside a tradable session phase ({market.phase})"

        position = self.portfolio.position(order.symbol)
        verdict = rules.tradability(
            is_suspended=market.is_suspended,
            at_limit_up=market.at_limit_up,
            at_limit_down=market.at_limit_down,
            holding_acquired_today=(
                order.side == SELL and position.sellable < order.quantity
                and position.pending_settlement > 0
            ),
        )
        if order.side == BUY and not verdict.can_buy:
            return "; ".join(verdict.reasons)
        if order.side == SELL and not verdict.can_sell:
            return "; ".join(verdict.reasons)

        if order.side == BUY and market.is_st and not self.config.allow_st_buy:
            return "ST buy blocked by policy"

        sized = rules.round_to_lot(
            order.quantity, board=order.board, side=order.side,
            is_full_liquidation=order.is_full_liquidation,
        )
        if sized <= 0:
            minimum, step = rules.LOT_RULES.get(order.board, (100, 100))
            return (f"quantity {order.quantity:g} rounds below the {order.board} "
                    f"minimum lot ({minimum}/{step})")
        order.quantity = float(sized)

        limits = market.limits()
        if order.limit_price is not None and not limits.unlimited:
            if limits.limit_up is not None and order.limit_price > limits.limit_up + 1e-9:
                return (f"limit {order.limit_price} exceeds the price ceiling "
                        f"{limits.limit_up}")
            if limits.limit_down is not None and order.limit_price < limits.limit_down - 1e-9:
                return (f"limit {order.limit_price} is below the price floor "
                        f"{limits.limit_down}")

        if order.side == SELL and order.quantity - self.portfolio.sellable(order.symbol) > 1e-9:
            return (f"sell of {order.quantity:.0f} exceeds the T+1-settled "
                    f"{self.portfolio.sellable(order.symbol):.0f}")

        return None

    # -- pricing -----------------------------------------------------------
    def _execution_price(self, order: Order, market: MarketSnapshot,
                         quantity: float) -> float:
        """Fill price including slippage and square-root market impact."""
        base = market.last_price
        slippage = base * (self.config.slippage_bps / 10_000.0)
        participation = (
            quantity / market.session_volume if market.session_volume > 0 else 0.0
        )
        impact = base * self.config.impact_coefficient * (participation ** 0.5)
        adjustment = slippage + impact
        price = base + adjustment if order.side == BUY else base - adjustment

        limits = market.limits()
        if not limits.unlimited:
            if limits.limit_up is not None:
                price = min(price, limits.limit_up)
            if limits.limit_down is not None:
                price = max(price, limits.limit_down)
        if order.limit_price is not None:
            price = min(price, order.limit_price) if order.side == BUY \
                else max(price, order.limit_price)
        return round(price, 2)

    # -- submission --------------------------------------------------------
    def submit(self, order: Order, market: MarketSnapshot) -> Order:
        """Validate, accept and attempt to fill a single order."""
        self.orders[order.order_id] = order
        self._emit(lg.ORDER_CREATED, {"order": order.to_dict()},
                   symbol=order.symbol, market_time=market.clock)

        reason = self._validate(order, market)
        if reason:
            return self._reject(order, reason, market)

        order.transition(ACCEPTED)
        self._emit(lg.ORDER_ACCEPTED, {"order_id": order.order_id,
                                       "quantity": order.quantity},
                   symbol=order.symbol, market_time=market.clock)
        return self._attempt_fill(order, market)

    def _attempt_fill(self, order: Order, market: MarketSnapshot) -> Order:
        available = market.session_volume * self.config.participation_cap
        quantity = min(order.remaining, available)

        # Respect the limit: a resting buy below the market does not trade.
        if order.order_type == LIMIT and order.limit_price is not None:
            if order.side == BUY and market.last_price > order.limit_price:
                return order
            if order.side == SELL and market.last_price < order.limit_price:
                return order

        quantity = rules.round_to_lot(
            quantity, board=order.board, side=order.side,
            is_full_liquidation=order.is_full_liquidation,
        )
        if quantity <= 0:
            return order

        price = self._execution_price(order, market, quantity)
        notional = quantity * price
        costs = rules.trading_costs(
            notional_cny=notional, side=order.side, trade_date=market.trade_date,
            commission_rate=self.config.commission_rate,
        )
        fill = Fill(
            order_id=order.order_id, symbol=order.symbol, side=order.side,
            quantity=float(quantity), price=price, notional=notional,
            commission=costs.commission, stamp_duty=costs.stamp_duty,
            transfer_fee=costs.transfer_fee, market_time=market.clock,
            partial=quantity < order.remaining,
        )

        try:
            self.portfolio.apply_fill(fill)
        except (InsufficientCash, InsufficientSellable) as exc:
            return self._reject(order, str(exc), market)

        self.fills.append(fill)
        order.filled_quantity += fill.quantity
        order.filled_notional += fill.notional
        order.fees_paid += fill.fees

        if order.remaining <= 1e-9:
            order.transition(FILLED)
            event = lg.ORDER_FILLED
        else:
            order.transition(PARTIALLY_FILLED)
            event = lg.ORDER_PARTIALLY_FILLED

        self._emit(event, {"fill": fill.to_dict(), "order_id": order.order_id},
                   symbol=order.symbol, market_time=market.clock)
        self._emit(lg.CASH_CHANGED,
                   {"delta": fill.cash_delta, "cash": self.portfolio.cash},
                   symbol=order.symbol, market_time=market.clock)
        self._emit(lg.POSITION_CHANGED,
                   {"position": self.portfolio.position(order.symbol).to_dict()},
                   symbol=order.symbol, market_time=market.clock)
        return order

    def submit_parent(
        self, parent: ParentOrder, markets: Sequence[MarketSnapshot]
    ) -> list[Order]:
        """Slice a parent order and submit its children in schedule order."""
        children = parent.schedule(
            reference_prices=[m.last_price for m in markets],
            volumes=[m.session_volume for m in markets],
        )
        return [self.submit(child, market)
                for child, market in zip(children, markets)]

    # -- cancellation ------------------------------------------------------
    def cancel(self, order_id: str, market: MarketSnapshot | None = None) -> Order:
        order = self.orders[order_id]
        if not order.is_open:
            return order
        order.transition(CANCEL_REQUESTED)
        self._emit(lg.ORDER_CANCEL_REQUESTED, {"order_id": order_id},
                   symbol=order.symbol)
        order.transition(CANCELLED)
        self._emit(lg.ORDER_CANCELLED,
                   {"order_id": order_id, "unfilled": order.remaining},
                   symbol=order.symbol)
        return order

    # -- session -----------------------------------------------------------
    def mark_to_market(self, prices: Mapping[str, float], *, market_time: str | None = None) -> dict[str, Any]:
        snapshot = self.portfolio.to_dict(prices)
        self._emit(lg.MARK_TO_MARKET, snapshot, market_time=market_time)
        return snapshot

    def apply_corporate_action(self, symbol: str, *, share_ratio: float = 1.0,
                               cash_per_share: float = 0.0) -> dict[str, Any]:
        result = self.portfolio.apply_corporate_action(
            symbol, share_ratio=share_ratio, cash_per_share=cash_per_share
        )
        self._emit(lg.CORPORATE_ACTION_APPLIED, result, symbol=symbol)
        return result

    def close_session(self, trade_date: str) -> dict[str, Any]:
        """Settle T+1 purchases and close the session."""
        settled = self.portfolio.settle()
        payload = {"trade_date": trade_date, "settled": settled,
                   "cash": self.portfolio.cash}
        self._emit(lg.SESSION_CLOSED, payload, market_time=trade_date)
        return payload

    def open_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if o.is_open]
