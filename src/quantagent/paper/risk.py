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
from typing import Any, Iterable, Mapping, Sequence

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
    max_industry_weight: float = 0.30
    max_gross_exposure: float = 1.0
    max_daily_turnover: float = 2.0
    max_daily_loss: float = 20_000.0
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
    """Scoped, latching kill switch. Only a human clears it."""

    def __init__(self) -> None:
        self._triggered: dict[str, dict[str, Any]] = {}

    def trigger(self, scope: str, reason: str, *, key: str | None = None) -> dict[str, Any]:
        if scope not in SCOPES:
            raise ValueError(f"unknown kill-switch scope {scope!r}; known: {list(SCOPES)}")
        record = {
            "scope": scope, "key": key, "reason": reason,
            "triggered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._triggered[self._id(scope, key)] = record
        return record

    def is_triggered(self, scope: str, key: str | None = None) -> bool:
        if self._id(SCOPE_GLOBAL, None) in self._triggered:
            return True  # global halts everything below it
        return self._id(scope, key) in self._triggered

    def active(self) -> list[dict[str, Any]]:
        return list(self._triggered.values())

    def clear(self, scope: str, key: str | None = None, *,
              human_confirmation: bool = False) -> bool:
        """Clear a switch. Refuses without explicit human confirmation."""
        if not human_confirmation:
            raise RiskRejection(
                ["kill_switch_clear"],
                "clearing a kill switch requires explicit human confirmation; "
                "an automatic reset would defeat the control entirely",
            )
        return self._triggered.pop(self._id(scope, key), None) is not None

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
    ) -> None:
        self.limits = limits or RiskLimits()
        self.kill_switch = kill_switch or KillSwitch()
        self.ledger = event_ledger
        self.run_id = run_id
        self.session_turnover: float = 0.0
        self.session_start_equity: float | None = None
        self.peak_equity: float | None = None
        self._seen_orders: set[str] = set()

    # -- pre-trade ---------------------------------------------------------
    def check_order(
        self,
        order: Order,
        portfolio: Portfolio,
        *,
        reference_price: float,
        session_volume: float = 0.0,
        prices: Mapping[str, float] | None = None,
        quote_age_seconds: float = 0.0,
        industry: str | None = None,
        industry_weights: Mapping[str, float] | None = None,
        model_approved: bool = True,
        dataset_approved: bool = True,
    ) -> RiskDecision:
        checks: list[RiskCheck] = []
        prices = dict(prices or {})
        prices.setdefault(order.symbol, reference_price)
        equity = max(portfolio.equity(prices), 1e-9)
        notional = order.quantity * (order.limit_price or reference_price)

        checks.append(RiskCheck(
            "kill_switch", not self.kill_switch.is_triggered(
                SCOPE_STRATEGY, order.strategy_id),
            "no kill switch active for this scope"))

        checks.append(RiskCheck(
            "duplicate_order", order.order_id not in self._seen_orders,
            "order id has not been submitted before", measured=order.order_id))

        checks.append(RiskCheck(
            "model_approved", bool(model_approved),
            "signal comes from an approved model"))
        checks.append(RiskCheck(
            "dataset_approved", bool(dataset_approved),
            "signal comes from an approved dataset"))

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

        if session_volume > 0:
            participation = order.quantity / session_volume
            checks.append(RiskCheck(
                "participation", participation <= self.limits.max_participation,
                "order participation within limit",
                self.limits.max_participation, participation))

        if order.side == BUY:
            projected = portfolio.cash - notional
            checks.append(RiskCheck(
                "cash_available", projected >= self.limits.min_cash_buffer,
                "sufficient cash after this order",
                self.limits.min_cash_buffer, projected))

            position_value = (
                portfolio.position(order.symbol).market_value(reference_price) + notional
            )
            weight = position_value / equity
            checks.append(RiskCheck(
                "single_name_weight", weight <= self.limits.max_single_name_weight,
                "single-name weight within limit",
                self.limits.max_single_name_weight, weight))

            if industry and industry_weights is not None:
                projected_industry = industry_weights.get(industry, 0.0) + notional / equity
                checks.append(RiskCheck(
                    "industry_weight",
                    projected_industry <= self.limits.max_industry_weight,
                    "industry weight within limit",
                    self.limits.max_industry_weight, projected_industry))
        else:
            checks.append(RiskCheck(
                "position_available",
                portfolio.sellable(order.symbol) + 1e-9 >= order.quantity,
                "sufficient T+1-settled shares",
                portfolio.sellable(order.symbol), order.quantity))

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
        self._emit(decision, order)
        return decision

    # -- portfolio ---------------------------------------------------------
    def check_portfolio(
        self,
        portfolio: Portfolio,
        prices: Mapping[str, float],
        *,
        industry_map: Mapping[str, str] | None = None,
    ) -> RiskDecision:
        checks: list[RiskCheck] = []
        equity = max(portfolio.equity(prices), 1e-9)

        if self.session_start_equity is None:
            self.session_start_equity = equity
        self.peak_equity = max(self.peak_equity or equity, equity)

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
        checks.append(RiskCheck(
            "daily_loss", daily_loss <= self.limits.max_daily_loss,
            "daily loss within limit", self.limits.max_daily_loss, daily_loss))

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
                self.kill_switch.trigger(SCOPE_PORTFOLIO,
                                         f"daily loss {check.measured:.2f} exceeds "
                                         f"{check.limit:.2f}")
            elif check.name == "drawdown":
                self.kill_switch.trigger(SCOPE_PORTFOLIO,
                                         f"drawdown {check.measured:.2%} exceeds "
                                         f"{check.limit:.2%}")
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
