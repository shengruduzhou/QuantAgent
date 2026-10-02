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
import math
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


class InvalidMarketSnapshot(ValueError):
    """A snapshot whose price or volume is not a measurement.

    A NaN session volume used to disable the participation cap, the pre-trade
    participation check and market impact all at once: ``min(remaining, nan)``
    is ``remaining`` and ``nan > 0`` is False, so an unmeasured bar filled the
    whole order at zero impact. An unmeasured value is refused, never defaulted.
    """


def snapshot_invalid_reason(
    *,
    last_price: Any,
    previous_close: Any,
    session_volume: Any,
    high: Any = None,
    low: Any = None,
) -> str | None:
    """Why these snapshot values cannot be priced against, or None."""
    def finite(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    problems: list[str] = []
    for name, value in (("last_price", last_price), ("previous_close", previous_close)):
        if not finite(value) or float(value) <= 0:
            problems.append(f"{name}={value!r} is not a finite positive price")
    if not finite(session_volume) or float(session_volume) < 0:
        problems.append(f"session_volume={session_volume!r} is not a finite non-negative volume")
    for name, value in (("high", high), ("low", low)):
        if value is not None and (not finite(value) or float(value) <= 0):
            problems.append(f"{name}={value!r} is not a finite positive price")
    return "; ".join(problems) or None


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

    def __post_init__(self) -> None:
        reason = self.invalid_reason()
        if reason is not None:
            raise InvalidMarketSnapshot(
                f"market_data_invalid for {self.symbol} on {self.trade_date}: {reason}"
            )

    def invalid_reason(self) -> str | None:
        """Re-checkable: the dataclass is mutable, so the venue checks again."""
        return snapshot_invalid_reason(
            last_price=self.last_price, previous_close=self.previous_close,
            session_volume=self.session_volume, high=self.high, low=self.low,
        )

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

    @property
    def observed_at(self) -> datetime | None:
        """Exchange-local (Asia/Shanghai, naive) time of this quote, or None."""
        try:
            return datetime.fromisoformat(f"{str(self.trade_date)[:10]}T{str(self.clock)[:8]}")
        except ValueError:
            return None


#: Industry bucket for held names missing from the venue's industry map.
UNKNOWN_INDUSTRY = "__unknown__"


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
        risk_engine: "RiskEngine | None" = None,
        industry_map: Mapping[str, str] | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.ledger = event_ledger
        self.run_id = run_id
        self.config = config or BrokerConfig()
        # Portfolio-level risk (single-name weight, industry concentration,
        # gross exposure, daily loss, drawdown, participation). `_validate`
        # below only ever covered instrument-level rules, so a paper venue
        # constructed without this enforces NO portfolio limit at all. That was
        # the production state: `quantagent.paper.risk` was imported by nothing
        # except its own test file while three production call sites built this
        # broker.
        #
        # None is permitted -- a reconciliation harness comparing engines must
        # not have risk injected into the economic comparison -- but it is
        # never silent: `risk_engine_attached` publishes which regime a run was
        # in, so "no portfolio risk was applied" is readable from the run rather
        # than assumed from the absence of rejections.
        self.risk_engine = risk_engine
        # symbol -> industry, used to measure industry concentration. Absent or
        # incomplete while the industry limit is below 1.0 means a BUY of an
        # unmapped name is refused as `industry_unmeasured`, never waved through.
        self.industry_map: dict[str, str] = dict(industry_map or {})
        # Last mark per symbol from every snapshot this venue has seen, and the
        # venue's simulated clock (the newest quote time observed). The risk
        # engine values the *whole* book against these; valuing it with only the
        # order's own price raised UnpriceablePosition on the second name held.
        self.marks: dict[str, float] = {}
        self._mark_times: dict[str, datetime] = {}
        self.clock: datetime | None = None
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
        self.last_order_decision = None

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

    @property
    def risk_engine_attached(self) -> bool:
        """Whether portfolio-level limits are enforced on this venue."""
        return self.risk_engine is not None

    def _validate(self, order: Order, market: MarketSnapshot) -> str | None:
        """Return a rejection reason, or None when the order may proceed."""
        if self.killed:
            return f"kill switch active: {self.kill_reason}"

        invalid = market.invalid_reason()
        if invalid is not None:
            return f"market_data_invalid: {invalid}"

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

        # Portfolio-level limits last: everything above is a property of the
        # instrument and the order, this is a property of the book they would
        # join. A rejection here is final -- `RiskDecision` deliberately carries
        # no override path.
        if self.risk_engine is not None:
            prices = dict(self.marks)
            prices.setdefault(order.symbol, market.last_price)
            decision = self.risk_engine.check_order(
                order,
                self.portfolio,
                reference_price=market.last_price,
                session_volume=market.session_volume,
                prices=prices,
                quote_age_seconds=self.quote_age_seconds(market),
                industry=self.industry_map.get(order.symbol),
                industry_weights=self.industry_weights(prices),
            )
            self.last_order_decision = decision
            if not decision.approved:
                return "portfolio risk rejected: " + ",".join(decision.failed)

        return None

    # -- marks -------------------------------------------------------------
    def observe(self, market: MarketSnapshot) -> None:
        """Record a snapshot's price as the symbol's mark if it is the newest."""
        stamp = market.observed_at
        if stamp is None or market.invalid_reason() is not None:
            return
        self._record_mark(market.symbol, float(market.last_price), stamp)

    def observe_marks(self, prices: Mapping[str, float], *, trade_date: str,
                      clock: str) -> None:
        """Record a full mark map observed at one exchange time.

        The continuous loop knows every held and target symbol's price for the
        session; handing the venue the whole map is what lets the risk engine
        value the entire book rather than only the order's own symbol.
        """
        stamp = datetime.fromisoformat(f"{str(trade_date)[:10]}T{str(clock)[:8]}")
        for symbol, price in prices.items():
            value = float(price)
            if math.isfinite(value) and value > 0:
                self._record_mark(str(symbol), value, stamp)

    def _record_mark(self, symbol: str, price: float, stamp: datetime) -> None:
        previous = self._mark_times.get(symbol)
        if previous is None or stamp >= previous:
            self.marks[symbol] = price
            self._mark_times[symbol] = stamp
        if self.clock is None or stamp > self.clock:
            self.clock = stamp

    def quote_age_seconds(self, market: MarketSnapshot) -> float:
        """Age of this quote against the venue clock (newest quote observed).

        A quote older than something the venue has already seen is measured as
        that much older; a quote whose time cannot be read is infinitely old.
        """
        stamp = market.observed_at
        if stamp is None:
            return math.inf
        if self.clock is None or stamp >= self.clock:
            return 0.0
        return (self.clock - stamp).total_seconds()

    def unpriceable_symbols(self) -> tuple[str, ...]:
        return self.portfolio.unpriceable(self.marks)

    def industry_weights(self, prices: Mapping[str, float]) -> dict[str, float] | None:
        """Held weight per industry, or None when the book cannot be valued."""
        if self.portfolio.unpriceable(prices):
            return None
        equity = max(self.portfolio.equity(prices), 1e-9)
        weights: dict[str, float] = {}
        for symbol, position in self.portfolio.positions.items():
            if position.is_flat:
                continue
            industry = self.industry_map.get(symbol, UNKNOWN_INDUSTRY)
            weights[industry] = weights.get(industry, 0.0) + abs(
                position.market_value(prices[symbol])
            ) / equity
        return weights

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
        self.observe(market)
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
