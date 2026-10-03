"""Risk controls for the local paper desk: pre-trade, portfolio, operational.

Three layers because they fail differently. Pre-trade rejects a single order
before it reaches the book. Portfolio limits constrain the shape of the whole
book and can be breached by a trade that was individually fine. Operational
checks catch the system lying to itself -- a stale heartbeat, a negative
position, a cash figure that disagrees with the ledger.

**A risk rejection is final.** There is no override parameter, no force flag and
no confidence score that outranks it. That is the single most important property
here: the failure mode being prevented is a strategy component talking its way
past the component that checked.

Kill switches are scoped (order / strategy / portfolio / global) and, once
triggered, stay triggered until a human clears them explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from quantagent.paper import ledger as lg
from quantagent.paper.orders import BUY, SELL, Order
from quantagent.paper.portfolio import Portfolio

APPROVED = "APPROVED"
REJECTED = "REJECTED"

# --- kill switch scopes -----------------------------------------------------
SCOPE_ORDER = "ORDER"
SCOPE_STRATEGY = "STRATEGY"
SCOPE_PORTFOLIO = "PORTFOLIO"
SCOPE_GLOBAL = "GLOBAL"
SCOPES: tuple[str, ...] = (SCOPE_ORDER, SCOPE_STRATEGY, SCOPE_PORTFOLIO, SCOPE_GLOBAL)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_industry_map(path: str | Path) -> dict[str, str]:
    """symbol -> industry from a sector map (``symbol`` + ``industry`` columns).

    Same rule as the target-weight optimiser's sector lookup (last row per
    symbol), so the venue and the optimiser measure the same industries. For
    forward paper trading the current snapshot is the point-in-time map; it is
    not valid for historical replay.
    """
    import pandas as pd

    from quantagent.portfolio.v7_target_weights import normalise_sector_map

    source = Path(path)
    frame = (
        pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    )
    try:
        frame = normalise_sector_map(frame)
    except ValueError as exc:
        raise ValueError(f"sector map {source}: {exc}") from exc
    frame = frame.dropna(subset=["symbol", "industry"])
    if frame.empty:
        raise ValueError(f"sector map {source} has no symbol/industry rows")
    return frame.groupby("symbol")["industry"].last().astype(str).to_dict()


class RiskRejection(RuntimeError):
    """Raised when risk refuses an action. Deliberately has no override path."""

    def __init__(self, checks: Sequence[str], message: str) -> None:
        super().__init__(message)
        self.failed_checks = list(checks)


@dataclass
class RiskLimits:
    """Configured limits. Every one is checked; none is advisory."""

    max_order_notional: float = 200_000.0
    max_order_shares: float = 1_000_000.0
    max_single_name_weight: float = 0.10
    #: While below 1.0 a BUY whose industry is unknown is refused
    #: (`industry_unmeasured`). 1.0 is the explicit opt-out, never an absent map.
    max_industry_weight: float = 0.30
    max_gross_exposure: float = 1.0
    max_daily_turnover: float = 2.0
    #: Daily-loss breaker as a fraction of the session's opening equity. A
    #: diversified long-only book loses 5% in a day only on tail days (the
    #: certified-panel top-50 book's 99% one-day VaR was 5.86%); a fixed 20,000
    #: CNY on a 1M book (2%) latched reduce-only on roughly one day in ten.
    max_daily_loss_fraction: float = 0.05
    #: Optional absolute CNY cap applied in addition to the fraction.
    max_daily_loss: float | None = None
    max_drawdown: float = 0.20
    max_participation: float = 0.10
    #: A quote older than this is not a price, it is a memory.
    max_quote_age_seconds: int = 300
    #: Fat-finger guard: reject a limit this far from the reference price.
    max_price_deviation: float = 0.10
    min_cash_buffer: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskCheck:
    name: str
    passed: bool
    detail: str = ""
    limit: Any = None
    measured: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskDecision:
    verdict: str
    checks: list[RiskCheck] = field(default_factory=list)
    decided_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def approved(self) -> bool:
        return self.verdict == APPROVED

    @property
    def failed(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict, "approved": self.approved,
            "failed_checks": self.failed, "decided_at": self.decided_at,
            "checks": [c.to_dict() for c in self.checks],
            "override_available": False,
        }


class KillSwitch:
    """Scoped, latching kill switch. Only a human clears it.

    With a ``journal`` attached every trigger and clear is appended to a
    durable ledger, and `restore` rebuilds the latched set from it. Without
    that a restart (crash, deploy, OOM) silently cleared every switch, which
    made a restart the easiest kill-switch bypass there was.
    """

    def __init__(self) -> None:
        self._triggered: dict[str, dict[str, Any]] = {}
        #: ``journal(event_type, payload)`` persists a state change, or None.
        self.journal: Callable[[str, dict[str, Any]], Any] | None = None

    def trigger(self, scope: str, reason: str, *, key: str | None = None) -> dict[str, Any]:
        if scope not in SCOPES:
            raise ValueError(f"unknown kill-switch scope {scope!r}; known: {list(SCOPES)}")
        existing = self._triggered.get(self._id(scope, key))
        if existing is not None:
            # Latched: the first breach is the record; a repeat adds nothing.
            return existing
        record = {
            "scope": scope, "key": key, "reason": reason,
            "triggered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        # In memory first: if persisting fails the switch is still on here,
        # which is the safe direction.
        self._triggered[self._id(scope, key)] = record
        if self.journal is not None:
            self.journal(lg.KILL_SWITCH_TRIGGERED, dict(record))
        return record

    def restore(self, events: Iterable[lg.Event]) -> None:
        """Rebuild the latched set from durable trigger/clear events."""
        for event in events:
            payload = event.payload
            scope = str(payload.get("scope") or SCOPE_GLOBAL)
            key = payload.get("key")
            if event.event_type == lg.KILL_SWITCH_TRIGGERED:
                self._triggered.setdefault(self._id(scope, key), {
                    "scope": scope, "key": key,
                    "reason": str(payload.get("reason") or "unknown"),
                    "triggered_at": payload.get("triggered_at") or event.event_time,
                })
            elif event.event_type == lg.KILL_SWITCH_CLEARED:
                self._triggered.pop(self._id(scope, key), None)

    def is_triggered(self, scope: str, key: str | None = None) -> bool:
        if self._id(SCOPE_GLOBAL, None) in self._triggered:
            return True  # global halts everything below it
        return self._id(scope, key) in self._triggered

    def active(self) -> list[dict[str, Any]]:
        return list(self._triggered.values())

    def blocking(self, side: str, strategy_id: str | None = None) -> list[dict[str, Any]]:
        """Switches that refuse an order on ``side``.

        GLOBAL refuses everything. PORTFOLIO is reduce-only: it refuses a
        risk-increasing BUY and lets a SELL through, because a breached
        drawdown limit that also froze exits would hold the very loss it exists
        to cap. STRATEGY refuses that strategy's orders, and every strategy's
        when triggered without a key.
        """
        found: list[dict[str, Any]] = []
        for record in self._triggered.values():
            scope, key = record["scope"], record.get("key")
            if scope == SCOPE_GLOBAL:
                found.append(record)
            elif scope == SCOPE_PORTFOLIO and side == BUY:
                found.append(record)
            elif scope == SCOPE_STRATEGY and (key is None or key == strategy_id):
                found.append(record)
        return found

    def clear(self, scope: str, key: str | None = None, *,
              human_confirmation: bool = False) -> bool:
        """Clear a switch. Refuses without explicit human confirmation."""
        if not human_confirmation:
            raise RiskRejection(
                ["kill_switch_clear"],
                "clearing a kill switch requires explicit human confirmation; "
                "an automatic reset would defeat the control entirely",
            )
        cleared = self._triggered.pop(self._id(scope, key), None) is not None
        if cleared and self.journal is not None:
            self.journal(lg.KILL_SWITCH_CLEARED, {
                "scope": scope, "key": key, "human_confirmation": True,
                "cleared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
        return cleared

    @staticmethod
    def _id(scope: str, key: str | None) -> str:
        return f"{scope}:{key or '*'}"


class RiskEngine:
    """Evaluates pre-trade, portfolio and operational risk. Rejections are final."""

    def __init__(
        self,
        limits: RiskLimits | None = None,
        *,
        kill_switch: KillSwitch | None = None,
        event_ledger: lg.EventLedger | None = None,
        run_id: str = "risk",
        state_ledger: lg.EventLedger | None = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.kill_switch = kill_switch or KillSwitch()
        self.ledger = event_ledger
        self.run_id = run_id
        #: Exchange session (YYYY-MM-DD) the turnover and daily-loss figures
        #: below belong to. None until a venue starts a session.
        self.session_date: str | None = None
        self.session_turnover: float = 0.0
        self.session_start_equity: float | None = None
        #: All-time peak equity; drawdown is measured against it.
        self.peak_equity: float | None = None
        self.last_portfolio_decision: RiskDecision | None = None
        #: Exchange session on which a daily-loss breach halted new buys.
        self.loss_halted_session: str | None = None
        self._seen_orders: set[str] = set()
        self._turnover_by_session: dict[str, float] = {}
        self._start_equity_by_session: dict[str, float] = {}
        #: (session, equity) of the newest durable mark-to-market.
        self._last_close: tuple[str | None, float] | None = None
        self.state_ledger: lg.EventLedger | None = None
        self._state_portfolio_id = "risk"
        if state_ledger is not None:
            self.bind_state_ledger(state_ledger)

    def daily_loss_limit_cny(self) -> float:
        """Tightest of the fractional and (optional) absolute daily-loss caps."""
        caps = []
        if self.session_start_equity is not None and self.limits.max_daily_loss_fraction is not None:
            caps.append(float(self.limits.max_daily_loss_fraction) * float(self.session_start_equity))
        if self.limits.max_daily_loss is not None:
            caps.append(float(self.limits.max_daily_loss))
        return min(caps) if caps else float("inf")

    # -- durable state -----------------------------------------------------
    def bind_state_ledger(self, ledger: lg.EventLedger, *, portfolio_id: str = "risk") -> None:
        """Persist risk state to ``ledger`` and rebuild it from what is there.

        Kill switches, per-session turnover, session-start equity and the peak
        equity were in-memory only: a restart cleared a latched switch, forgot
        today's consumed turnover and reset the drawdown reference. They are now
        ledger events, replayed here on construction.
        """
        if self.state_ledger is not None:
            if self.state_ledger.path.resolve() != ledger.path.resolve():
                raise ValueError(
                    "risk state is already bound to another ledger; two records of "
                    "one account's risk state cannot both be authoritative"
                )
            return
        self.state_ledger = ledger
        self._state_portfolio_id = portfolio_id
        events = list(ledger.read())
        self.kill_switch.restore(events)
        self.kill_switch.journal = self._journal
        for event in events:
            payload = event.payload
            if event.event_type == lg.RISK_STATE_UPDATED:
                session = payload.get("session")
                if payload.get("daily_loss_halt"):
                    self.loss_halted_session = session
                if session and _finite(payload.get("session_turnover")):
                    self._turnover_by_session[session] = float(payload["session_turnover"])
                if session and _finite(payload.get("session_start_equity")):
                    self._start_equity_by_session.setdefault(
                        session, float(payload["session_start_equity"])
                    )
                if _finite(payload.get("peak_equity")):
                    self._observe_peak(float(payload["peak_equity"]))
            elif event.event_type == lg.MARK_TO_MARKET and _finite(payload.get("equity")):
                equity = float(payload["equity"])
                self._observe_peak(equity)
                self._last_close = ((event.market_time or "")[:10] or None, equity)

    def _journal(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self.state_ledger is None:
            return
        self.state_ledger.append(
            event_type, run_id=self.run_id, portfolio_id=self._state_portfolio_id,
            payload=dict(payload),
        )

    def _observe_peak(self, equity: float) -> bool:
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
            return True
        return False

    def observe_inception_equity(self, equity: float) -> None:
        """The account's opening capital is a NAV observation like any other."""
        if _finite(equity) and float(equity) > 0:
            self._observe_peak(float(equity))

    def begin_session(self, session: str, *, opening_equity: float | None = None) -> None:
        """Key turnover and daily loss to an exchange session, not process life.

        A long-lived process used to refuse day-2 orders on day-1 turnover, and
        a restart forgot turnover consumed today. Opening equity is the persisted
        figure for this session if any, else the caller's valuation at the marks
        known before the session's first quote, else the newest durable close.
        """
        session = str(session)[:10]
        if session == self.session_date:
            return
        self.session_date = session
        self.session_turnover = self._turnover_by_session.get(session, 0.0)
        start = self._start_equity_by_session.get(session)
        if start is None and opening_equity is not None and _finite(opening_equity):
            start = float(opening_equity)
        if (
            start is None and self._last_close is not None
            and self._last_close[0] is not None and self._last_close[0] < session
        ):
            start = self._last_close[1]
        self.session_start_equity = start
        if start is not None:
            self._record_session_start(session, start)

    def _record_session_start(self, session: str, equity: float) -> None:
        if session in self._start_equity_by_session:
            return
        self._start_equity_by_session[session] = equity
        self._journal(lg.RISK_STATE_UPDATED, {"session": session, "session_start_equity": equity})

    # -- pre-trade ---------------------------------------------------------
    def check_order(
        self,
        order: Order,
        portfolio: Portfolio,
        *,
        reference_price: float,
        session_volume: float = 0.0,
        prices: Mapping[str, float] | None = None,
        quote_age_seconds: float | None = None,
        industry: str | None = None,
        industry_weights: Mapping[str, float] | None = None,
        model_approved: bool | None = None,
        dataset_approved: bool | None = None,
    ) -> RiskDecision:
        """Pre-trade decision. Never raises for a book it cannot value.

        ``prices`` must be the venue's full mark map. A held position without a
        mark makes every weight-based limit unmeasurable, so the order is
        REJECTED with ``book_priceable`` naming the unpriced symbols. Valuing
        the book with the order's own price only (the previous behaviour)
        raised ``UnpriceablePosition`` out of the venue on the second name.

        ``model_approved`` / ``dataset_approved`` / ``quote_age_seconds`` left
        as None mean "not measured by this caller": the check is omitted from
        the decision record rather than recorded as passed.
        """
        checks: list[RiskCheck] = []
        prices = dict(prices or {})
        prices.setdefault(order.symbol, reference_price)
        unpriceable = portfolio.unpriceable(prices)
        equity = None if unpriceable else max(portfolio.equity(prices), 1e-9)
        notional = order.quantity * (order.limit_price or reference_price)

        blocking = self.kill_switch.blocking(order.side, order.strategy_id)
        checks.append(RiskCheck(
            "kill_switch", not blocking,
            "no kill switch blocks this order (PORTFOLIO scope is reduce-only)",
            measured=[
                {"scope": r["scope"], "key": r.get("key"), "reason": r["reason"]}
                for r in blocking
            ] or None))
        if order.side == BUY:
            halted = self.loss_halted_session is not None and (
                self.loss_halted_session == self.session_date
            )
            checks.append(RiskCheck(
                "daily_loss_halt", not halted,
                "no daily-loss halt on this session (buys resume next session)",
                measured=self.loss_halted_session if halted else None))

        checks.append(RiskCheck(
            "duplicate_order", order.order_id not in self._seen_orders,
            "order id has not been submitted before", measured=order.order_id))

        if model_approved is not None:
            checks.append(RiskCheck(
                "model_approved", bool(model_approved),
                "signal comes from an approved model"))
        if dataset_approved is not None:
            checks.append(RiskCheck(
                "dataset_approved", bool(dataset_approved),
                "signal comes from an approved dataset"))

        if quote_age_seconds is not None:
            checks.append(RiskCheck(
                "stale_data", quote_age_seconds <= self.limits.max_quote_age_seconds,
                "quote is fresh enough to price against",
                self.limits.max_quote_age_seconds, quote_age_seconds))

        checks.append(RiskCheck(
            "order_notional", notional <= self.limits.max_order_notional,
            "order notional within limit", self.limits.max_order_notional, notional))

        checks.append(RiskCheck(
            "order_shares", order.quantity <= self.limits.max_order_shares,
            "order size within limit", self.limits.max_order_shares, order.quantity))

        if order.limit_price is not None and reference_price > 0:
            deviation = abs(order.limit_price - reference_price) / reference_price
            checks.append(RiskCheck(
                "fat_finger", deviation <= self.limits.max_price_deviation,
                "limit price is near the reference price",
                self.limits.max_price_deviation, deviation))

        if not (_finite(session_volume) and float(session_volume) >= 0):
            checks.append(RiskCheck(
                "participation", False,
                "session volume is not a measurement; participation cannot be bounded",
                self.limits.max_participation, session_volume))
        elif session_volume > 0:
            participation = order.quantity / session_volume
            checks.append(RiskCheck(
                "participation", participation <= self.limits.max_participation,
                "order participation within limit",
                self.limits.max_participation, participation))

        checks.append(RiskCheck(
            "book_priceable", not unpriceable,
            "every held position has a mark, so book weights are measurable",
            measured=list(unpriceable)))

        if order.side == BUY:
            projected = portfolio.cash - notional
            checks.append(RiskCheck(
                "cash_available", projected >= self.limits.min_cash_buffer,
                "sufficient cash after this order",
                self.limits.min_cash_buffer, projected))

            if equity is not None:
                position_value = (
                    portfolio.position(order.symbol).market_value(reference_price) + notional
                )
                weight = position_value / equity
                checks.append(RiskCheck(
                    "single_name_weight", weight <= self.limits.max_single_name_weight,
                    "single-name weight within limit",
                    self.limits.max_single_name_weight, weight))
                checks.extend(self._industry_checks(
                    order, notional=notional, equity=equity,
                    industry=industry, industry_weights=industry_weights,
                ))
        else:
            checks.append(RiskCheck(
                "position_available",
                portfolio.sellable(order.symbol) + 1e-9 >= order.quantity,
                "sufficient T+1-settled shares",
                portfolio.sellable(order.symbol), order.quantity))

        if equity is not None:
            turnover = (self.session_turnover + notional) / equity
            checks.append(RiskCheck(
                "daily_turnover", turnover <= self.limits.max_daily_turnover,
                "daily turnover within limit", self.limits.max_daily_turnover, turnover))

        decision = RiskDecision(
            verdict=APPROVED if all(c.passed for c in checks) else REJECTED,
            checks=checks,
        )
        if decision.approved:
            self._seen_orders.add(order.order_id)
            self.session_turnover += notional
            if self.session_date is not None:
                self._turnover_by_session[self.session_date] = self.session_turnover
                self._journal(lg.RISK_STATE_UPDATED, {
                    "session": self.session_date, "session_turnover": self.session_turnover,
                })
        self._emit(decision, order)
        return decision

    def _industry_checks(
        self,
        order: Order,
        *,
        notional: float,
        equity: float,
        industry: str | None,
        industry_weights: Mapping[str, float] | None,
    ) -> list[RiskCheck]:
        limit = self.limits.max_industry_weight
        if limit >= 1.0:
            # The explicit opt-out: no claim about industry concentration.
            return []
        if not industry or industry_weights is None:
            # Unmeasured is not "within limit". Skipping here is what let a
            # 100% single-sector book through every production venue.
            return [RiskCheck(
                "industry_unmeasured", False,
                "industry limit cannot be measured: the symbol has no industry or "
                "the book's industry weights were not supplied (a limit of 1.0 is "
                "the explicit opt-out)",
                limit,
                {"symbol": order.symbol, "industry": industry,
                 "industry_weights_supplied": industry_weights is not None},
            )]
        projected_industry = industry_weights.get(industry, 0.0) + notional / equity
        return [RiskCheck(
            "industry_weight", projected_industry <= limit,
            "industry weight within limit", limit, projected_industry)]

    # -- portfolio ---------------------------------------------------------
    def check_portfolio(
        self,
        portfolio: Portfolio,
        prices: Mapping[str, float],
        *,
        industry_map: Mapping[str, str] | None = None,
    ) -> RiskDecision:
        unpriceable = portfolio.unpriceable(prices)
        if unpriceable:
            # Unmeasured is neither a breach nor a pass: no kill switch latches
            # on it, and no limit is reported as satisfied.
            decision = RiskDecision(verdict=REJECTED, checks=[RiskCheck(
                "book_priceable", False,
                "held positions without a mark; portfolio limits are unmeasurable",
                measured=list(unpriceable))])
            self.last_portfolio_decision = decision
            self._emit(decision, None)
            return decision
        checks: list[RiskCheck] = []
        equity = max(portfolio.equity(prices), 1e-9)

        if self.session_start_equity is None:
            self.session_start_equity = equity
            if self.session_date is not None:
                self._record_session_start(self.session_date, equity)
        if self._observe_peak(equity):
            self._journal(lg.RISK_STATE_UPDATED, {
                "session": self.session_date, "peak_equity": equity,
            })

        gross = portfolio.gross_exposure(prices) / equity
        checks.append(RiskCheck(
            "gross_exposure", gross <= self.limits.max_gross_exposure,
            "gross exposure within limit", self.limits.max_gross_exposure, gross))

        for symbol, position in portfolio.positions.items():
            if position.is_flat or symbol not in prices:
                continue
            weight = abs(position.market_value(prices[symbol])) / equity
            checks.append(RiskCheck(
                f"concentration:{symbol}",
                weight <= self.limits.max_single_name_weight,
                "single-name concentration within limit",
                self.limits.max_single_name_weight, weight))

        if industry_map:
            industry_totals: dict[str, float] = {}
            for symbol, position in portfolio.positions.items():
                if position.is_flat or symbol not in prices:
                    continue
                industry = industry_map.get(symbol, "UNKNOWN")
                industry_totals[industry] = industry_totals.get(industry, 0.0) + abs(
                    position.market_value(prices[symbol])
                )
            for industry, value in industry_totals.items():
                checks.append(RiskCheck(
                    f"industry:{industry}",
                    value / equity <= self.limits.max_industry_weight,
                    "industry exposure within limit",
                    self.limits.max_industry_weight, value / equity))

        daily_loss = self.session_start_equity - equity
        loss_limit = self.daily_loss_limit_cny()
        checks.append(RiskCheck(
            "daily_loss", daily_loss <= loss_limit,
            "daily loss within limit", loss_limit, daily_loss))

        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity else 0.0
        checks.append(RiskCheck(
            "drawdown", drawdown <= self.limits.max_drawdown,
            "drawdown within limit", self.limits.max_drawdown, drawdown))

        decision = RiskDecision(
            verdict=APPROVED if all(c.passed for c in checks) else REJECTED,
            checks=checks,
        )
        for check in checks:
            if check.passed:
                continue
            if check.name == "daily_loss":
                # A session-scoped halt, not a latch: new buys stop for the rest
                # of this exchange session and the halt lapses at the next one.
                # Latching reduce-only on a -5% day held a real top-50 book at
                # ~12% exposure from 2018-02 onward (round-29 R7 breaker sim).
                if self.loss_halted_session != self.session_date:
                    self.loss_halted_session = self.session_date
                    self._journal(lg.RISK_STATE_UPDATED, {
                        "session": self.session_date, "daily_loss_halt": True,
                        "daily_loss": check.measured, "daily_loss_limit": check.limit,
                    })
            elif check.name == "drawdown":
                self.kill_switch.trigger(SCOPE_PORTFOLIO,
                                         f"drawdown {check.measured:.2%} exceeds "
                                         f"{check.limit:.2%}")
        self.last_portfolio_decision = decision
        self._emit(decision, None)
        return decision

    # -- operational -------------------------------------------------------
    def check_operational(
        self,
        portfolio: Portfolio,
        *,
        heartbeat_age_seconds: float = 0.0,
        max_heartbeat_age: float = 120.0,
        ledger_valid: bool = True,
        reconciliation_passed: bool = True,
        disk_free_bytes: int | None = None,
        min_disk_free_bytes: int = 1 << 30,
        clock_drift_seconds: float = 0.0,
        max_clock_drift: float = 5.0,
        schema_matches: bool = True,
        consecutive_rejections: int = 0,
        max_consecutive_rejections: int = 5,
    ) -> RiskDecision:
        checks = [
            RiskCheck("heartbeat", heartbeat_age_seconds <= max_heartbeat_age,
                      "process heartbeat is fresh", max_heartbeat_age,
                      heartbeat_age_seconds),
            RiskCheck("ledger_chain", ledger_valid, "event ledger chain verifies"),
            RiskCheck("reconciliation", reconciliation_passed,
                      "live state matches ledger replay"),
            RiskCheck("clock_drift", abs(clock_drift_seconds) <= max_clock_drift,
                      "clock drift within tolerance", max_clock_drift,
                      clock_drift_seconds),
            RiskCheck("schema", schema_matches, "dataset schema matches expectation"),
            RiskCheck("repeated_rejections",
                      consecutive_rejections < max_consecutive_rejections,
                      "order rejections are not repeating",
                      max_consecutive_rejections, consecutive_rejections),
        ]
        if disk_free_bytes is not None:
            checks.append(RiskCheck(
                "disk_space", disk_free_bytes >= min_disk_free_bytes,
                "sufficient free disk for checkpoints",
                min_disk_free_bytes, disk_free_bytes))

        negative = [s for s, p in portfolio.positions.items() if p.total < -1e-9]
        checks.append(RiskCheck("no_negative_positions", not negative,
                                "no short positions in a long-only paper book",
                                measured=negative))

        decision = RiskDecision(
            verdict=APPROVED if all(c.passed for c in checks) else REJECTED,
            checks=checks,
        )
        for check in checks:
            if check.passed:
                continue
            if check.name in ("ledger_chain", "reconciliation"):
                self.kill_switch.trigger(
                    SCOPE_GLOBAL, f"operational failure: {check.name}")
        self._emit(decision, None)
        return decision

    # -- helpers -----------------------------------------------------------
    def _emit(self, decision: RiskDecision, order: Order | None) -> None:
        if self.ledger is None:
            return
        payload = decision.to_dict()
        if order is not None:
            payload["order_id"] = order.order_id
        self.ledger.append(
            lg.RISK_APPROVED if decision.approved else lg.RISK_REJECTED,
            run_id=self.run_id, portfolio_id="risk",
            payload=payload, symbol=order.symbol if order else None,
        )

    def enforce(self, decision: RiskDecision) -> None:
        """Convert a rejection into a hard stop. There is no override argument."""
        if not decision.approved:
            raise RiskRejection(
                decision.failed,
                f"risk rejected the action on {decision.failed}; this decision is "
                "final and cannot be overridden by a strategy component or a vote",
            )
