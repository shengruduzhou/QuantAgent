from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha1
from typing import Iterable
from uuid import uuid4

import pandas as pd

from quantagent.execution.broker_base import (
    BrokerBase,
    Order,
    OrderIntent,
    OrderSide,
    OrderState,
    OrderStatus,
    OrderType,
)
from quantagent.execution.constraints import (
    ExecutionConstraintEvaluator,
    OrderIntentRecord,
)
from quantagent.domain.idempotency import IdempotencyStore, order_intent_key
from quantagent.domain.ledger import CanonicalLedger, mirror_open
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import (
    OrderBook,
    OrderEventType as CanonicalEventType,
    OrderIntent as CanonicalIntent,
    RiskDecision as CanonicalRiskDecision,
    Side as CanonicalSide,
    Signal,
)
from quantagent.quant_math.ashare import AshareRuleEngine


class MissingIdempotencyLineage(RuntimeError):
    """An economic submission arrived without the identity needed to de-duplicate it."""


class IdempotencyConflict(RuntimeError):
    """The same idempotency key was reused with different economics."""

    def __init__(self, key: str, stored: str, attempted: str) -> None:
        super().__init__(
            f"idempotency key {key!r} was already used with fingerprint {stored!r}; "
            f"this request has {attempted!r}"
        )
        self.key = key
        self.stored = stored
        self.attempted = attempted


class ForensicHarnessLeak(RuntimeError):
    """A forensic replay was pointed at a broker that can execute economically."""


class PreTradeRiskRejected(RuntimeError):
    """A production pre-trade rule refused an order before the broker was touched."""


def _request_fingerprint(order: Order) -> str:
    """Deterministic digest of everything that makes the request economically distinct."""
    return sha1(
        "|".join(
            str(part) for part in (
                order.symbol, order.side.value, int(order.quantity),
                order.order_type.value, order.price, order.strategy_version,
            )
        ).encode("utf-8")
    ).hexdigest()[:16]


def assert_forensic_isolation(broker: object) -> None:
    """Refuse a forensic harness wired to anything that can really trade.

    Reproducing a historical bug means deliberately disabling the duplicate
    guard. That is only safe against an in-memory double; against paper, shadow
    or a broker it would reintroduce the very duplicates the guard prevents.
    """
    name = type(broker).__module__ + "." + type(broker).__qualname__
    forbidden = ("paper", "shadow", "live", "qmt", "broker_gateway")
    if any(token in name.lower() for token in forbidden):
        raise ForensicHarnessLeak(
            f"forensic_replay=True cannot be used with {name}: it can reach economic execution"
        )


@dataclass(frozen=True)
class OrderManagerConfig:
    lot_size: int = 100
    min_order_value_yuan: float = 100.0
    allow_odd_lot_sell_only_for_full_liquidation: bool = True
    max_orders_per_symbol_per_day: int = 5
    block_buy_limit_up: bool = True
    block_sell_limit_down: bool = True
    max_participation_rate: float = 0.05
    #: A target weight at or below this is treated as "no position wanted".
    #: Rank/softmax weighting leaves a long tail of ~1e-7 weights that can never
    #: round to a tradable lot; without a floor each one becomes a skipped-order
    #: record for a symbol the book was never going to touch.
    negligible_weight: float = 1e-6
    strategy_version: str = "v4.0"


@dataclass
class OrderRecord:
    order: Order
    state: OrderState
    submitted_at: str
    last_updated_at: str


@dataclass
class OrderManager:
    """Idempotent order router. Generates per-symbol delta orders from target weights.

    The canonical ``RISK_APPROVED`` event is a fact, not a routing convenience.
    A QMT live gateway advertises that it requires explicit risk approval; in
    that case the OMS runs :class:`ExecutionConstraintEvaluator` before opening
    the broker side of the lifecycle and replaces ``risk_check_result`` with the
    literal ``approved`` only when the evaluator passes.  A rejected order is
    recorded canonically as ``RISK_REJECTED`` and never reaches ``broker.submit``.

    Non-live/research brokers retain their existing economic path, but their
    canonical decision is now named ``order_manager_basic_admissibility`` so it
    no longer falsely claims that cash/T+1/band/portfolio production risk was
    checked when it was not.
    """

    broker: BrokerBase
    config: OrderManagerConfig = field(default_factory=OrderManagerConfig)
    rule_engine: AshareRuleEngine = field(default_factory=AshareRuleEngine)
    counts_today: dict[str, int] = field(default_factory=dict)
    last_skipped_orders: list[dict[str, object]] = field(default_factory=list)
    skipped_orders: list[dict[str, object]] = field(default_factory=list)
    #: Canonical record of account. `history` above is an in-memory projection
    #: for the broker wire protocol; economic truth lives here and survives a
    #: restart. Pass `ledger_path` to make it durable.
    ledger_path: str | None = None
    lineage: Lineage = field(default_factory=Lineage)
    #: Durable claim-once guard. `history` alone could not protect a restart:
    #: a worker killed between broker.submit() and _update() lost the record and
    #: resubmitted the same intent on recovery.
    idempotency_path: str | None = None
    #: Isolated harness for reproducing historical (pre-INC-E1) behaviour.
    #: Bypasses the durable guard so a shipped bug stays reproducible, and
    #: is refused by any broker that can reach real, paper or shadow
    #: execution — see `assert_forensic_isolation`.
    forensic_replay: bool = False
    #: Injected when a canonical-aware venue (the paper broker) must append its
    #: own lifecycle events to *this* chain. Sharing one ledger and one book is
    #: what keeps a single economic order to a single record of account; letting
    #: each side build its own would double-count every order that crosses them.
    canonical_ledger: CanonicalLedger | None = None
    order_book: OrderBook | None = None
    #: Production constraint evaluator.  It is only promoted to an authoritative
    #: live gate when the broker declares ``require_risk_approval`` in live mode;
    #: this avoids silently changing historical backtest economics while still
    #: making the real broker path fail closed.
    constraint_evaluator: ExecutionConstraintEvaluator = field(default_factory=ExecutionConstraintEvaluator)

    def __post_init__(self) -> None:
        if self.forensic_replay:
            assert_forensic_isolation(self.broker)
        if self.canonical_ledger is not None and self.ledger_path is not None:
            raise ValueError(
                "pass either canonical_ledger or ledger_path, not both: two chains "
                "for one order manager is the duplicate record of account this "
                "argument exists to prevent"
            )
        # `is not None`, not `or`: CanonicalLedger defines __len__, so an empty
        # injected ledger is falsy and `or` would silently swap it for a fresh
        # in-memory one — every event would be written to a chain nobody reads.
        self.canonical = (
            self.canonical_ledger
            if self.canonical_ledger is not None
            else CanonicalLedger(self.ledger_path)
        )
        # As in PaperBroker: a chain that already holds events determines the
        # starting state. Beginning empty against a populated ledger lets the
        # manager re-open an order the file says is terminal, which appends events
        # that make the chain unreplayable (DEF-014).
        self.book = (
            self.order_book if self.order_book is not None
            else (self.canonical.replay_book() if len(self.canonical) else OrderBook())
        )
        #: True when the venue writes its own ACCEPTED/FILL/REJECTED events to the
        #: shared chain. The OMS then stops at SUBMITTED rather than folding the
        #: broker reply a second time.
        self._venue_is_canonical = callable(getattr(self.broker, "attach_canonical", None))
        self.claims = IdempotencyStore(self.idempotency_path)
        #: Broker wire request/reply cache, keyed by client_order_id. Not
        #: economic truth — see the `history` property.
        self._wire: dict[str, OrderRecord] = {}
        #: canonical order_id -> broker client_order_id, the one-way link
        #: that lets the wire cache be projected from the ledger.
        self._client_order_ids: dict[str, str] = {}
        #: Intraday live-submit history used by the stateful constraint DSL.
        #: It counts orders that actually passed local risk and were attempted
        #: at the venue.  Runtime-state persistence across process restart is a
        #: separate production gate; broker preflight remains authoritative for
        #: open orders/trades/positions after restart.
        self._risk_intents_today: list[OrderIntentRecord] = []

    def reset_daily_counters(self) -> None:
        self.counts_today.clear()
        self._risk_intents_today.clear()

    def reconcile(self, target_weights: pd.Series, prices: pd.Series, nav: float,
                  signal_id: str = "manual") -> list[OrderState]:
        positions = {p.symbol: p for p in self.broker.query_positions()}
        intents = self.target_weights_to_order_intents(
            target_weights, prices, nav, positions=positions, signal_id=signal_id)
        orders: list[Order] = []
        for intent in intents:
            if self.counts_today.get(intent.symbol, 0) >= self.config.max_orders_per_symbol_per_day:
                continue
            orders.append(
                Order(
                    client_order_id=intent.intent_id,
                    symbol=intent.symbol,
                    side=intent.side,
                    quantity=intent.quantity,
                    order_type=OrderType.LIMIT,
                    price=intent.reference_price,
                    note="target_weight_reconcile",
                    signal_id=intent.signal_id,
                    model_version=intent.model_version,
                    feature_version=intent.feature_version,
                    strategy_version=intent.strategy_version,
                    risk_check_result=intent.risk_check_result,
                    timestamp=intent.timestamp,
                )
            )
        return list(self._submit_all(orders))

    def target_weights_to_order_intents(
        self,
        target_weights: pd.Series,
        prices: pd.Series,
        nav: float,
        positions: dict[str, object] | None = None,
        signal_id: str = "manual",
        model_version: str = "unknown",
        feature_version: str = "unknown",
        risk_check_result: str = "not_checked",
    ) -> list[OrderIntent]:
        positions = positions or {p.symbol: p for p in self.broker.query_positions()}
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        intents: list[OrderIntent] = []
        self.last_skipped_orders = []
        for symbol, weight in target_weights.reindex(prices.index).fillna(0.0).items():
            price = float(prices.loc[symbol])
            if pd.isna(price) or price <= 0 or nav <= 0:
                self._skip(str(symbol), OrderSide.BUY, 0, float(weight), 0.0 if pd.isna(price) else price, "skipped_invalid_price", now, 0.0)
                continue
            current = positions.get(str(symbol))
            current_shares = int(getattr(current, "available_shares", 0) + getattr(current, "frozen_shares", 0)) if current else 0
            raw_target = float(weight) * nav / price
            # Nothing held and nothing wanted is not an order that was skipped —
            # it is an absence of intent. Auditing it as a skip buried the real
            # skips: a 400-name universe produced 1,971 of these against 320
            # genuine ones, and the backtest read as though execution friction
            # had blocked 93% of its trading.
            if current_shares == 0 and abs(float(weight)) <= self.config.negligible_weight:
                continue
            delta_shares = raw_target - current_shares
            delta_value = abs(delta_shares) * price
            if raw_target >= current_shares:
                side = OrderSide.BUY
                quantity = self.rule_engine.round_order_quantity(str(symbol), "buy", delta_shares)
                if quantity <= 0:
                    # Real intent that cannot be expressed in whole lots. The
                    # implied size is what makes it drillable: "wanted 0.0025
                    # shares" is a portfolio-weight problem, not a venue rule.
                    self._skip(
                        str(symbol), side, 0, float(weight), price,
                        "skipped_below_min_lot", now, delta_value,
                        implied_shares=delta_shares,
                    )
                    continue
                if quantity * price < self.config.min_order_value_yuan:
                    self._skip(str(symbol), side, quantity, float(weight), price, "skipped_small_order", now, quantity * price)
                    continue
            else:
                side = OrderSide.SELL
                desired_sell = current_shares - raw_target
                full_liquidation = current_shares < self.config.lot_size and float(weight) <= 1e-6
                if full_liquidation and self.config.allow_odd_lot_sell_only_for_full_liquidation:
                    quantity = current_shares
                else:
                    quantity = int(desired_sell // self.config.lot_size * self.config.lot_size)
                if quantity <= 0:
                    reason = (
                        "skipped_not_full_odd_lot_liquidation"
                        if current_shares < self.config.lot_size and not full_liquidation
                        else "skipped_below_min_lot"
                    )
                    self._skip(
                        str(symbol), side, 0, float(weight), price, reason, now, delta_value,
                        implied_shares=-desired_sell,
                    )
                    continue
                if quantity * price < self.config.min_order_value_yuan and not full_liquidation:
                    self._skip(str(symbol), side, quantity, float(weight), price, "skipped_small_order", now, quantity * price)
                    continue
            if quantity <= 0:
                continue
            intent_id = self._make_id(str(symbol), side, signal_id=signal_id, model_version=model_version)
            intents.append(
                OrderIntent(
                    intent_id=intent_id,
                    symbol=str(symbol),
                    side=side,
                    quantity=quantity,
                    target_weight=float(weight),
                    reference_price=price,
                    signal_id=signal_id,
                    model_version=model_version,
                    feature_version=feature_version,
                    strategy_version=self.config.strategy_version,
                    risk_check_result=risk_check_result,
                    timestamp=now,
                )
            )
        return intents

    def _skip(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        target_weight: float,
        reference_price: float,
        reason: str,
        timestamp: str,
        delta_value: float,
        implied_shares: float | None = None,
    ) -> None:
        row = {
            "symbol": symbol,
            "side": side.value,
            "quantity": int(quantity),
            "target_weight": float(target_weight),
            "reference_price": float(reference_price),
            "reason": reason,
            "delta_value": float(delta_value),
            # The fractional share count the weight asked for. Without it a
            # sub-lot skip is indistinguishable from a venue rejection.
            "implied_shares": None if implied_shares is None else float(implied_shares),
            "timestamp": timestamp,
        }
        self.last_skipped_orders.append(row)
        self.skipped_orders.append(row)

    def submit_orders(self, orders: Iterable[Order]) -> list[OrderState]:
        """Route explicit orders through the full guarded path.

        `reconcile` covers the weight-driven case, but an OMS must also accept an
        order somebody stated directly — a manual instruction from the web app, a
        replay of a recorded intent, a reconciliation fix-up. Routing those through
        the same `_submit_all` is what stops a second, unguarded submission path
        from appearing next to the protected one.
        """
        return list(self._submit_all(orders))

    def cancel_all_open(self) -> list[OrderState]:
        results: list[OrderState] = []
        for record in self.history.values():
            if record.state.status in {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL}:
                state = self.broker.cancel(record.order.client_order_id)
                self._update(record.order, state)
                results.append(state)
        return results

    def _submit_all(self, orders: Iterable[Order]) -> Iterable[OrderState]:
        for original_order in orders:
            order = original_order
            # Durable claim before the broker is touched. The in-memory
            # `history` check that used to guard this was lost on restart, so a
            # worker killed after submit but before _update resubmitted the same
            # intent on recovery — a second economic order.
            if not self.forensic_replay and not self.lineage.run_id:
                raise MissingIdempotencyLineage(
                    f"order {order.client_order_id} has no lineage.run_id; economic "
                    "submission requires canonical lineage and an idempotency key"
                )
            key = order_intent_key(
                run_id=self.lineage.run_id or "forensic",
                signal_id=order.signal_id or "manual",
                symbol=order.symbol,
                side=order.side.value,
                quantity=int(order.quantity),
                trade_date=str(order.timestamp)[:10],
            )
            fingerprint = _request_fingerprint(order)
            if not self.forensic_replay:
                claim = self.claims.claim(
                    key,
                    payload={"clientOrderId": order.client_order_id, "fingerprint": fingerprint},
                )
                if not claim.granted:
                    stored = claim.record.payload.get("fingerprint")
                    if stored is not None and stored != fingerprint:
                        raise IdempotencyConflict(key, str(stored), fingerprint)
                    existing = self._wire.get(str(claim.record.payload.get("clientOrderId") or ""))
                    if existing is not None:
                        yield existing.state
                    continue
            if self.forensic_replay and order.client_order_id in self.history:
                continue

            # Open the canonical intent/order in PENDING_RISK first.  Risk is
            # applied next and therefore cannot be retrospectively invented
            # after the venue has already seen the order.
            canonical_order = self._open_canonical_pending(order)
            self._client_order_ids[canonical_order.order_id] = order.client_order_id
            order, risk_decision, risk_intent = self._pretrade_decision(order, canonical_order)
            if not risk_decision.approved:
                self._apply_canonical_risk(canonical_order, risk_decision, approved=False)
                state = OrderState(
                    client_order_id=order.client_order_id,
                    broker_order_id=None,
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    avg_price=0.0,
                    last_message=f"pretrade_risk_rejected:{risk_decision.reason}",
                )
                self._update(order, state)
                if not self.forensic_replay:
                    self.claims.resolve(
                        key,
                        outcome=order.client_order_id,
                        payload={
                            "clientOrderId": order.client_order_id,
                            "orderId": canonical_order.order_id,
                            "fingerprint": fingerprint,
                            "riskApproved": False,
                            "riskReason": risk_decision.reason,
                        },
                    )
                yield state
                continue

            canonical_order = self._apply_canonical_risk(canonical_order, risk_decision, approved=True)
            if risk_intent is not None:
                # Count the local approved attempt before touching the venue so
                # a broker-level rejection still counts toward order-rate and
                # turnover constraints for the current process/session.
                self._risk_intents_today.append(risk_intent)
            if self._venue_is_canonical:
                self.broker.attach_canonical(order.client_order_id, canonical_order.order_id)
            state = self.broker.submit(order)
            self._update(order, state)
            if not self._venue_is_canonical:
                self._record_canonical_state(canonical_order, state)
            if not self.forensic_replay:
                self.claims.resolve(
                    key, outcome=order.client_order_id,
                    payload={
                        "clientOrderId": order.client_order_id,
                        "orderId": canonical_order.order_id,
                        "fingerprint": fingerprint,
                        "riskApproved": True,
                    },
                )
            self.counts_today[order.symbol] = self.counts_today.get(order.symbol, 0) + 1
            yield state

    def _requires_production_pretrade(self) -> bool:
        """Whether the attached broker declares a fail-closed live risk contract."""
        broker_config = getattr(self.broker, "config", None)
        if broker_config is None:
            return False
        return bool(
            getattr(broker_config, "require_risk_approval", False)
            and getattr(broker_config, "live_trading_enabled", False)
            and not getattr(broker_config, "dry_run", True)
        )

    def _pretrade_decision(
        self,
        order: Order,
        canonical_order,
    ) -> tuple[Order, CanonicalRiskDecision, OrderIntentRecord | None]:
        """Evaluate pre-submit risk and return a canonical first-class decision."""
        explicit = (order.risk_check_result or "not_checked").strip().lower()
        if explicit in {"rejected", "reject", "blocked", "failed", "fail", "denied"}:
            return (
                order,
                CanonicalRiskDecision.create(
                    approved=False,
                    rule="upstream_risk",
                    threshold="approved",
                    measured=explicit,
                    reason="upstream risk explicitly rejected the order",
                    lineage=canonical_order.lineage,
                ),
                None,
            )

        # Basic order-shape admissibility is always real and auditable.  It does
        # not pretend to be portfolio/live risk.
        if int(order.quantity) <= 0:
            return (
                order,
                CanonicalRiskDecision.create(
                    approved=False,
                    rule="order_manager_basic_admissibility",
                    threshold="quantity>0",
                    measured=int(order.quantity),
                    reason="order quantity must be positive",
                    lineage=canonical_order.lineage,
                ),
                None,
            )
        if order.order_type == OrderType.LIMIT and (order.price is None or float(order.price) <= 0):
            return (
                order,
                CanonicalRiskDecision.create(
                    approved=False,
                    rule="order_manager_basic_admissibility",
                    threshold="limit_price>0",
                    measured=order.price,
                    reason="limit order requires a positive price",
                    lineage=canonical_order.lineage,
                ),
                None,
            )

        if not self._requires_production_pretrade():
            decision = CanonicalRiskDecision.create(
                approved=True,
                rule="order_manager_basic_admissibility",
                threshold="positive_quantity+valid_limit_price",
                measured={"upstream": explicit, "mode": "non_live_or_research"},
                reason="basic OMS admissibility passed; this is not production risk certification",
                lineage=canonical_order.lineage,
                decided_by="order_manager",
            )
            return order, decision, None

        # Live QMT: an unbounded market order has no price with which to enforce
        # order-value/deviation/board protection.  The production path therefore
        # fails closed even though the low-level broker adapter knows how to send
        # a QMT market-price type.
        if order.order_type == OrderType.MARKET or order.price is None or float(order.price) <= 0:
            return (
                order,
                CanonicalRiskDecision.create(
                    approved=False,
                    rule="execution_constraint_dsl",
                    threshold="bounded_limit_order",
                    measured={"order_type": order.order_type.value, "price": order.price},
                    reason="live A-share order requires a bounded positive reference/limit price",
                    lineage=canonical_order.lineage,
                ),
                None,
            )

        timestamp = pd.Timestamp(
            order.timestamp or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )
        nav: float | None = None
        try:
            queried_nav = float(self.broker.query_account_value())
            if pd.notna(queried_nav) and queried_nav > 0:
                nav = queried_nav
        except Exception:
            # Missing account state must not be converted into an invented NAV.
            # The DSL will skip NAV-dependent checks; QMT preflight is separately
            # required to make a live submit reachable at all.
            nav = None
        risk_intent = OrderIntentRecord(
            intent_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=int(order.quantity),
            price=float(order.price),
            timestamp=timestamp,
            order_value=float(order.quantity) * float(order.price),
            portfolio_nav=nav,
            daily_volume_hint=None,
        )
        report = self.constraint_evaluator.evaluate([*self._risk_intents_today, risk_intent])
        decision = CanonicalRiskDecision.create(
            approved=report.passed,
            rule="execution_constraint_dsl",
            threshold=self.constraint_evaluator.constraints.as_dict(),
            measured=report.to_dict(),
            reason=(
                "all production pre-submit execution constraints passed"
                if report.passed
                else "blocking execution constraint violation: "
                + ",".join(sorted(report.by_constraint))
            ),
            lineage=canonical_order.lineage,
            decided_by="execution_constraint_dsl",
        )
        if not report.passed:
            return order, decision, None
        return replace(order, risk_check_result="approved"), decision, risk_intent

    # -- canonical emission --------------------------------------------------
    def _open_canonical_pending(self, order: Order):
        """Signal -> OrderIntent -> PENDING_RISK order on the canonical ledger."""
        session = str(order.timestamp)[:10]
        signal = Signal.create(
            symbol=order.symbol, trade_date=f"{session}-{order.signal_id or 'manual'}",
            score=0.0, lineage=self.lineage,
        )
        intent = CanonicalIntent.create(
            symbol=order.symbol,
            side=CanonicalSide(order.side.value.upper()),
            quantity=int(order.quantity),
            trade_date=session,
            lineage=signal.lineage,
            reference_price=order.price,
        )
        return mirror_open(self.book, self.canonical, intent, trade_date=session)

    def _apply_canonical_risk(self, canonical_order, decision: CanonicalRiskDecision, *, approved: bool):
        session = canonical_order.trade_date
        risk_event = CanonicalEventType.RISK_APPROVED if approved else CanonicalEventType.RISK_REJECTED
        self.book.apply(
            canonical_order.order_id,
            risk_event,
            risk_decision=decision,
            reason=None if approved else decision.reason,
        )
        self.canonical.append(self.book.history_of(canonical_order.order_id)[-1], trade_date=session)
        if not approved:
            return self.book.state_of(canonical_order.order_id)
        self.book.apply(canonical_order.order_id, CanonicalEventType.SUBMITTED)
        self.canonical.append(self.book.history_of(canonical_order.order_id)[-1], trade_date=session)
        return self.book.state_of(canonical_order.order_id)

    def _record_canonical_state(self, canonical, state: OrderState) -> None:
        """Fold the broker's reply into the canonical order.

        The broker's own status enum stays at the wire boundary; only the
        canonical state machine decides what the order *is*.
        """
        session = str(getattr(state, "timestamp", "") or "")[:10] or None
        mapping = {
            OrderStatus.REJECTED: CanonicalEventType.REJECTED,
            OrderStatus.CANCELLED: CanonicalEventType.CANCELLED,
        }
        event = mapping.get(state.status, CanonicalEventType.ACCEPTED)
        try:
            self.book.apply(canonical.order_id, event, reason=getattr(state, "reason", None))
        except Exception:
            # An unexpected broker status must not corrupt the canonical chain;
            # the order simply stays in its last legal state and the divergence
            # is visible because the ledger has no matching event.
            return
        self.canonical.append(self.book.history_of(canonical.order_id)[-1], trade_date=session)

    def _update(self, order: Order, state: OrderState) -> None:
        """Refresh the wire-protocol cache for one order.

        `_wire` is a cache of broker request/reply pairs, not economic truth:
        `history` rebuilds from the canonical ledger and `rebuild_history`
        proves the two agree. Cash, position and quantity are never read from
        here.
        """
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        record = self._wire.get(order.client_order_id)
        submitted_at = record.submitted_at if record is not None else now
        self._wire[order.client_order_id] = OrderRecord(
            order=order, state=state, submitted_at=submitted_at, last_updated_at=now,
        )

    @property
    def history(self) -> dict[str, OrderRecord]:
        """Broker-wire view of every order the canonical ledger knows about.

        Derived, not maintained: an order absent from the ledger cannot appear
        here, so the projection cannot drift into claiming an order the record
        of account does not have. Previously this dict was appended to
        independently and was the second copy of economic truth Module One
        exists to remove.
        """
        projected: dict[str, OrderRecord] = {}
        for canonical in self.book.orders():
            client_order_id = self._client_order_ids.get(canonical.order_id)
            if client_order_id is None:
                continue
            cached = self._wire.get(client_order_id)
            if cached is None:
                continue
            projected[client_order_id] = cached
        return projected

    def rebuild_history(self) -> dict[str, OrderRecord]:
        """Rebuild the projection from the ledger alone.

        Used by recovery and by the audit: if this disagrees with `history`,
        the cache had state the ledger did not, which is the defect class this
        design forbids.

        Reads the chain this manager actually writes to, re-opened from disk when
        it is durable. It used to construct `CanonicalLedger(self.ledger_path)`,
        which returned an *empty* ledger whenever the chain was injected or
        in-memory — so the audit compared `history` against nothing and passed
        (DEF-010).
        """
        source = (
            CanonicalLedger(self.canonical.path)
            if self.canonical.path is not None
            else self.canonical
        )
        book = source.replay_book()
        known = {order.order_id for order in book.orders()}
        return {
            client_order_id: record
            for order_id, client_order_id in self._client_order_ids.items()
            if order_id in known and (record := self._wire.get(client_order_id)) is not None
        }

    @staticmethod
    def _make_id(symbol: str, side: OrderSide, signal_id: str = "", model_version: str = "") -> str:
        seed = f"{symbol}-{side.value}-{signal_id}-{model_version}"
        suffix = uuid4().hex[:10] if not signal_id else sha1(seed.encode("utf-8")).hexdigest()[:10]
        return f"{symbol}-{side.value}-{suffix}"
