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
from quantagent.domain.ledger import CanonicalLedger, mirror_event, mirror_open
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import (
    CorporateAction as CanonicalCorporateAction,
    Fill as CanonicalFill,
    OrderBook,
    OrderEventType as CanonicalEventType,
    OrderIntent as CanonicalIntent,
    RiskDecision as CanonicalRiskDecision,
    Side as CanonicalSide,
    Signal,
)
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
        canonical_ledger_path: str | None = None,
        lineage: Lineage | None = None,
        canonical_ledger: CanonicalLedger | None = None,
        book: OrderBook | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.ledger = event_ledger
        self.run_id = run_id
        self.config = config or BrokerConfig()
        self.orders: dict[str, Order] = {}
        self.fills: list[Fill] = []
        #: Execution ids already booked. A set rather than a scan over `fills`
        #: so the idempotency check stays O(1) as a session accumulates.
        self._fill_ids: set[str] = set()
        # Economic record of account. `self.ledger` remains for operational
        # events (kill switch, mark-to-market, session close); every order,
        # fill, rejection and cancellation is mirrored here so a paper run
        # can be reconstructed without reading paper's own structures.
        #
        # An upstream OMS that already opened the canonical order injects its
        # ledger and book here. Constructing our own instead would give one
        # economic order two records of account — the duplication Module One
        # exists to remove — so the venue appends to the OMS's chain rather than
        # starting a parallel one.
        if canonical_ledger is not None and canonical_ledger_path is not None:
            raise ValueError(
                "pass either canonical_ledger or canonical_ledger_path, not both: "
                "two ledgers for one broker is exactly the duplicate record of "
                "account this argument exists to prevent"
            )
        # `is not None`, not `or`: CanonicalLedger defines __len__, so an empty
        # injected ledger is falsy and `or` would silently swap it for a fresh
        # in-memory one — the venue would then write to a chain nobody reads.
        self.canonical = (
            canonical_ledger
            if canonical_ledger is not None
            else CanonicalLedger(canonical_ledger_path)
        )
        # Attaching to a chain that already holds events means starting from the
        # state it records, not from empty. An empty book cannot see that an order
        # id already exists, so it would happily append a second CREATED and
        # RISK_APPROVED for an order the file says is FILLED — writing a chain
        # that no longer replays, and only discovering it at read time when the
        # damage is durable (DEF-014).
        self.book = (
            book if book is not None
            else (self.canonical.replay_book() if len(self.canonical) else OrderBook())
        )
        self.lineage = lineage or Lineage(run_id=run_id)
        self._canonical_ids: dict[str, str] = {}
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

    # -- canonical mirror --------------------------------------------------
    def _canonical_open(self, order: Order, trade_date: str | None = None) -> None:
        """Signal -> Intent -> Order on the canonical ledger, before any fill."""
        session = (trade_date or datetime.now(timezone.utc).isoformat())[:10]
        signal = Signal.create(
            symbol=order.symbol, trade_date=f"{session}-{order.order_id}",
            score=0.0, lineage=self.lineage,
        )
        intent = CanonicalIntent.create(
            symbol=order.symbol, side=CanonicalSide(order.side),
            quantity=int(order.quantity), trade_date=session,
            lineage=signal.lineage, limit_price=order.limit_price,
            reference_price=order.limit_price,
        )
        canonical = mirror_open(self.book, self.canonical, intent, trade_date=session)
        self._canonical_ids[order.order_id] = canonical.order_id
        decision = CanonicalRiskDecision.create(
            approved=True, rule="paper_pretrade", threshold="board+cash+t+1",
            measured="passed", reason="passed paper validation", lineage=canonical.lineage,
        )
        for event in (CanonicalEventType.RISK_APPROVED, CanonicalEventType.SUBMITTED):
            self.book.apply(
                canonical.order_id, event,
                risk_decision=decision if event is CanonicalEventType.RISK_APPROVED else None,
            )
            self.canonical.append(self.book.history_of(canonical.order_id)[-1], trade_date=session)

    def _canonical_event(
        self, order: Order, event_type: "CanonicalEventType", *,
        fill: "CanonicalFill | None" = None, reason: str | None = None,
        trade_date: str | None = None,
    ) -> None:
        canonical_id = self._canonical_ids.get(order.order_id)
        if canonical_id is None:
            return
        # `trade_date` stays None when the caller does not know the session. It
        # used to fall back to the wall clock, which silently wrote *today* into a
        # settlement-relevant field: a cancel issued without a market snapshot
        # stamped the current date, and replay then re-dated the order's earlier
        # fill to it, moving the T+1 lot a day forward (DEF-016). The bug was
        # invisible whenever the wall clock happened to match the session being
        # simulated. An absent date is recoverable — replay falls back to the
        # fill's own session — a wrong one is not.
        mirror_event(
            self.book, self.canonical, canonical_id, event_type,
            trade_date=(trade_date[:10] if trade_date else None),
            fill=fill, reason=reason,
        )

    def _canonical_fill(self, order: Order, fill: Fill, trade_date: str | None = None) -> None:
        canonical_id = self._canonical_ids.get(order.order_id)
        if canonical_id is None:
            return
        session = (trade_date or datetime.now(timezone.utc).isoformat())[:10]
        canonical = self.book.state_of(canonical_id)
        economic = CanonicalFill(
            execution_id=fill.fill_id, order_id=canonical_id, symbol=fill.symbol,
            side=CanonicalSide(fill.side), quantity=int(fill.quantity), price=float(fill.price),
            reference_price=float(order.limit_price or fill.price),
            commission=float(fill.commission), stamp_duty=float(fill.stamp_duty),
            transfer_fee=float(fill.transfer_fee), filled_at=session,
            lineage=canonical.lineage.derive(execution_id=fill.fill_id),
        )
        total = canonical.filled_quantity + economic.quantity
        event = (
            CanonicalEventType.FILL if total >= canonical.quantity
            else CanonicalEventType.PARTIAL_FILL
        )
        mirror_event(
            self.book, self.canonical, canonical_id, event, trade_date=session, fill=economic,
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
        self._canonical_event(order, CanonicalEventType.REJECTED, reason=reason,
                              trade_date=getattr(market, 'trade_date', None))
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
        """Fill price including slippage and square-root market impact.

        Delegates to `ashare_rules.execution_price`. The formula used to live here
        and was copied into the streaming matcher; two copies of a pricing formula
        agree only until one is edited, and the disagreement then arrives as a
        reconciliation difference with no way to say which side is right.
        """
        return rules.execution_price(
            last_price=market.last_price,
            side=order.side,
            quantity=quantity,
            session_volume=market.session_volume,
            slippage_bps=self.config.slippage_bps,
            impact_coefficient=self.config.impact_coefficient,
            limits=market.limits(),
            limit_price=order.limit_price,
        )

    def attach_canonical(self, order_id: str, canonical_order_id: str) -> None:
        """Adopt an order the upstream OMS already opened canonically.

        The venue does not re-open an order that already exists in the record of
        account; it appends its own lifecycle events (ACCEPTED, FILL, REJECTED,
        CANCELLED) to the order the OMS created. Without this the same economic
        order appears twice — once as the OMS's and once as paper's — and every
        aggregate built from the ledger double-counts it.
        """
        self._canonical_ids[order_id] = canonical_order_id

    # -- submission --------------------------------------------------------
    def submit(self, order: Order, market: MarketSnapshot) -> Order:
        """Validate, accept and attempt to fill a single order."""
        self.orders[order.order_id] = order
        if order.order_id not in self._canonical_ids:
            self._canonical_open(order, trade_date=getattr(market, 'trade_date', None))

        reason = self._validate(order, market)
        if reason:
            return self._reject(order, reason, market)

        order.transition(ACCEPTED)
        self._canonical_event(order, CanonicalEventType.ACCEPTED,
                              trade_date=getattr(market, 'trade_date', None))
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
            self.apply_execution_report(order, fill, trade_date=market.trade_date)
        except (InsufficientCash, InsufficientSellable) as exc:
            return self._reject(order, str(exc), market)
        return order

    def apply_execution_report(
        self, order: Order, fill: Fill, *, trade_date: str | None = None
    ) -> bool:
        """Book one execution. Idempotent on `fill_id`; returns False on a repeat.

        A venue re-delivers execution reports — a retried callback, a session
        reconnect, a gateway that timed out after the venue had already answered.
        Every one of those must move money exactly once, so the identity check
        lives here rather than in each caller: this is the only path by which a
        fill reaches the portfolio, the canonical ledger and the order.
        """
        if fill.fill_id in self._fill_ids:
            return False
        self.portfolio.apply_fill(fill)
        self._fill_ids.add(fill.fill_id)
        self.fills.append(fill)
        self._canonical_fill(order, fill, trade_date=trade_date)
        order.filled_quantity += fill.quantity
        order.filled_notional += fill.notional
        order.fees_paid += fill.fees

        # The canonical fill above already recorded this economically; the
        # paper transition below only advances paper's own view of the order.
        if order.remaining <= 1e-9:
            order.transition(FILLED)
        else:
            order.transition(PARTIALLY_FILLED)
        return True

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
        session = getattr(market, "trade_date", None)
        order.transition(CANCEL_REQUESTED)
        self._canonical_event(
            order, CanonicalEventType.CANCEL_REQUESTED, trade_date=session
        )
        order.transition(CANCELLED)
        self._canonical_event(
            order, CanonicalEventType.CANCELLED, reason='operator_cancel', trade_date=session
        )
        return order

    # -- session -----------------------------------------------------------
    def mark_to_market(self, prices: Mapping[str, float], *, market_time: str | None = None) -> dict[str, Any]:
        snapshot = self.portfolio.to_dict(prices)
        self._emit(lg.MARK_TO_MARKET, snapshot, market_time=market_time)
        return snapshot

    def apply_corporate_action(self, symbol: str, *, share_ratio: float = 1.0,
                               cash_per_share: float = 0.0,
                               ex_date: str | None = None) -> dict[str, Any]:
        """Apply a split, bonus issue or cash dividend, canonically.

        This used to mutate the portfolio and emit to the legacy operational log
        only, which classified a cash dividend and a share adjustment as telemetry.
        Measured cost of that classification (DEF-020): a 0.50/share dividend plus a
        2:1 split left the portfolio holding 2,000 shares and 990,494.90 in cash
        while the canonical replay still said 1,000 shares and 989,994.90 — a 500.00
        cash divergence and 1,000 shares, on a record of account that is supposed to
        be the only one.

        The canonical append comes first. If it fails the portfolio is untouched, so
        the two cannot end up disagreeing in the direction that matters: the ledger
        may know about an action the in-memory portfolio has not applied yet, and a
        restart replays it. The reverse — portfolio adjusted, ledger silent — is the
        divergence that cannot be recovered from.
        """
        session = (ex_date or datetime.now(timezone.utc).isoformat())[:10]
        action = CanonicalCorporateAction.create(
            symbol=symbol, ex_date=session, lineage=self.lineage,
            share_ratio=share_ratio, cash_per_share=cash_per_share,
        )
        self.canonical.append_corporate_action(action, trade_date=session)
        result = self.portfolio.apply_corporate_action(
            symbol, share_ratio=share_ratio, cash_per_share=cash_per_share
        )
        return result | {"corporateActionId": action.corporate_action_id}

    def close_session(self, trade_date: str) -> dict[str, Any]:
        """Settle T+1 purchases and close the session."""
        settled = self.portfolio.settle()
        payload = {"trade_date": trade_date, "settled": settled,
                   "cash": self.portfolio.cash}
        self._emit(lg.SESSION_CLOSED, payload, market_time=trade_date)
        return payload

    def open_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if o.is_open]
