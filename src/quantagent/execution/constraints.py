"""ExecutionConstraintDSL — declarative pre-submit checks (spec section 8).

Centralises every A-share execution constraint into one
:class:`ExecutionConstraintSet` so the gate logic can be:

* declared in one place,
* serialised + diffed across config versions,
* replayed against an order intent stream to produce
  :class:`ExecutionConstraintViolation` records.

The DSL is intentionally a *pre-submit* check: an
:class:`ExecutionConstraintEvaluator` consumes a list of order
intents + a market-state snapshot and returns the pass/fail verdict
plus a per-intent violation log. It does NOT mutate the orders.
:class:`quantagent.execution.order_manager.OrderManager` remains the
single seam that converts target_weights into intents, but it is
expected to consult this evaluator before forwarding intents to the
broker.

Spec compliance — every field in section 8 of the v8 spec maps to a
constraint here:

* ``max_orders_per_second``
* ``max_orders_per_day``
* ``max_cancel_ratio``
* ``min_order_resting_time_seconds``
* ``max_single_stock_participation_rate``
* ``max_single_order_value``
* ``max_daily_turnover``
* ``no_spoofing`` / ``no_layering`` / ``no_pull_push`` (heuristic
  detectors below)
* ``auction_mode`` flags (集合竞价 special handling)
* ``qmt_dry_run_required_by_default`` (the production wiring still
  defaults to dry_run; the DSL exposes the flag so callers can audit
  the posture)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# Auction mode
# ---------------------------------------------------------------------------

class AuctionPhase(str, Enum):
    PRE_AUCTION_OPEN = "pre_auction_open"   # 09:15–09:25
    AUCTION_OPEN = "auction_open"           # 09:25 fixing
    CONTINUOUS = "continuous"               # 09:30–11:30, 13:00–14:57
    PRE_AUCTION_CLOSE = "pre_auction_close"  # 14:57–15:00
    AUCTION_CLOSE = "auction_close"         # 15:00 fixing
    CLOSED = "closed"


def classify_auction_phase(ts: datetime | pd.Timestamp) -> AuctionPhase:
    """Classify an A-share timestamp into one of the trading phases.

    All A-share venues observe the same intraday phase boundaries.
    Times outside ``09:15–15:00`` (China local) are reported as
    :attr:`AuctionPhase.CLOSED`.
    """
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        wall = ts.to_pydatetime()
    elif isinstance(ts, datetime):
        wall = ts.replace(tzinfo=None) if ts.tzinfo is not None else ts
    else:
        raise TypeError(f"unsupported timestamp type: {type(ts)!r}")
    t = wall.time()
    if dtime(9, 15) <= t < dtime(9, 25):
        return AuctionPhase.PRE_AUCTION_OPEN
    if dtime(9, 25) <= t < dtime(9, 30):
        return AuctionPhase.AUCTION_OPEN
    if dtime(9, 30) <= t < dtime(11, 30):
        return AuctionPhase.CONTINUOUS
    if dtime(13, 0) <= t < dtime(14, 57):
        return AuctionPhase.CONTINUOUS
    if dtime(14, 57) <= t < dtime(15, 0):
        return AuctionPhase.PRE_AUCTION_CLOSE
    if t == dtime(15, 0):
        return AuctionPhase.AUCTION_CLOSE
    return AuctionPhase.CLOSED


# ---------------------------------------------------------------------------
# Constraint set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionConstraintSet:
    """Declarative bag of A-share execution constraints.

    All limits are *inclusive upper bounds* unless otherwise stated.
    Setting any value to ``None`` disables that check (the DSL never
    blocks for missing data, only for confirmed violations).
    """

    # Rate limits
    max_orders_per_second: int | None = 30
    max_orders_per_day: int | None = 500
    max_cancel_ratio: float | None = 0.50          # cancels / submits ≤ this
    min_order_resting_time_seconds: float | None = 0.5

    # Size limits
    max_single_stock_participation_rate: float | None = 0.10
    max_single_order_value: float | None = 1_000_000.0
    max_daily_turnover: float | None = 2.0          # 2x portfolio NAV/day

    # Auction-mode constraints (集合竞价): tighter rules during
    # 09:15-09:25 and 14:57-15:00 because exchanges look closely at
    #挂撤行为 here.
    auction_mode_max_orders_per_symbol: int | None = 2
    auction_mode_min_resting_time_seconds: float | None = 30.0  # do not flicker
    auction_mode_max_cancel_ratio: float | None = 0.20
    auction_mode_block_layering: bool = True

    # Spoofing / layering / pull-push heuristics
    no_spoofing: bool = True
    no_layering: bool = True
    no_pull_push: bool = True
    # heuristic thresholds (only applied when the heuristic flag is True)
    spoof_max_repeated_cancels: int = 3       # ≥N cancels of same symbol same minute
    layering_max_concurrent_levels: int = 5   # ≥N price levels of same side
    pull_push_min_size_jump: float = 3.0      # next-size ≥ N× prev-size

    # Posture
    qmt_dry_run_required_by_default: bool = True
    live_trading_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            k: getattr(self, k)
            for k in (
                "max_orders_per_second",
                "max_orders_per_day",
                "max_cancel_ratio",
                "min_order_resting_time_seconds",
                "max_single_stock_participation_rate",
                "max_single_order_value",
                "max_daily_turnover",
                "auction_mode_max_orders_per_symbol",
                "auction_mode_min_resting_time_seconds",
                "auction_mode_max_cancel_ratio",
                "auction_mode_block_layering",
                "no_spoofing",
                "no_layering",
                "no_pull_push",
                "spoof_max_repeated_cancels",
                "layering_max_concurrent_levels",
                "pull_push_min_size_jump",
                "qmt_dry_run_required_by_default",
                "live_trading_enabled",
            )
        }


# ---------------------------------------------------------------------------
# Intent record + violation record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderIntentRecord:
    """The DSL-shaped view of an order intent.

    Decoupled from :class:`quantagent.execution.broker_base.OrderIntent`
    so the constraint evaluator can run on synthetic histories during
    backtest and on live intent streams during paper / live trading.
    """

    intent_id: str
    symbol: str
    side: str               # "buy" / "sell" / "cancel"
    quantity: int
    price: float            # limit price; 0 for market
    timestamp: pd.Timestamp
    order_value: float = 0.0
    parent_intent_id: str | None = None   # for cancels — points to the original
    portfolio_nav: float | None = None    # snapshot of NAV at submit time
    daily_volume_hint: float | None = None  # snapshot of symbol's day-volume


@dataclass(frozen=True)
class ExecutionConstraintViolation:
    """One detected breach. Always carries the constraint name + reason."""

    intent_id: str
    symbol: str
    constraint: str
    severity: str           # "block" | "warn"
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionConstraintReport:
    n_intents: int
    n_violations: int
    n_blocking: int
    violations: list[ExecutionConstraintViolation] = field(default_factory=list)
    by_constraint: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.n_blocking == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_intents": self.n_intents,
            "n_violations": self.n_violations,
            "n_blocking": self.n_blocking,
            "by_constraint": dict(self.by_constraint),
            "passed": self.passed,
            "violations": [
                {
                    "intent_id": v.intent_id,
                    "symbol": v.symbol,
                    "constraint": v.constraint,
                    "severity": v.severity,
                    "reason": v.reason,
                    "detail": dict(v.detail),
                }
                for v in self.violations
            ],
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ExecutionConstraintEvaluator:
    """Checks a stream of intents against an :class:`ExecutionConstraintSet`.

    The evaluator is stateful within one ``evaluate()`` call: it
    aggregates per-second / per-day / per-symbol counters so multi-
    intent patterns (cancel ratio, repeated cancels, layering) can be
    detected. It does not persist state between calls so callers can
    safely re-evaluate the same stream.
    """

    def __init__(self, constraints: ExecutionConstraintSet | None = None) -> None:
        self.constraints = constraints or ExecutionConstraintSet()

    def evaluate(self, intents: Sequence[OrderIntentRecord]) -> ExecutionConstraintReport:
        c = self.constraints
        violations: list[ExecutionConstraintViolation] = []
        intents = list(intents)
        n_total = len(intents)

        if n_total == 0:
            return ExecutionConstraintReport(0, 0, 0)

        ordered = sorted(intents, key=lambda i: (i.timestamp, i.intent_id))

        # Posture: live with dry_run_required_by_default=True must error
        if c.live_trading_enabled and c.qmt_dry_run_required_by_default:
            violations.append(
                ExecutionConstraintViolation(
                    intent_id="<global>",
                    symbol="<global>",
                    constraint="qmt_dry_run_required_by_default",
                    severity="block",
                    reason="live_trading_enabled_but_dry_run_required",
                )
            )

        # Daily counters
        if c.max_orders_per_day is not None and n_total > c.max_orders_per_day:
            violations.append(
                ExecutionConstraintViolation(
                    intent_id="<batch>",
                    symbol="<batch>",
                    constraint="max_orders_per_day",
                    severity="block",
                    reason=f"{n_total}_above_{c.max_orders_per_day}",
                    detail={"n_intents": n_total},
                )
            )

        # Per-second rate
        if c.max_orders_per_second is not None:
            by_second: dict[pd.Timestamp, int] = {}
            for it in ordered:
                key = it.timestamp.floor("s") if hasattr(it.timestamp, "floor") else it.timestamp
                by_second[key] = by_second.get(key, 0) + 1
            for second, count in by_second.items():
                if count > c.max_orders_per_second:
                    violations.append(
                        ExecutionConstraintViolation(
                            intent_id="<window>",
                            symbol="<window>",
                            constraint="max_orders_per_second",
                            severity="block",
                            reason=f"{count}_above_{c.max_orders_per_second}",
                            detail={"window": str(second), "count": count},
                        )
                    )

        # Cancel ratio
        n_cancels = sum(1 for it in ordered if it.side == "cancel")
        n_submits = n_total - n_cancels
        if c.max_cancel_ratio is not None and n_submits > 0:
            ratio = n_cancels / max(1, n_submits)
            if ratio > c.max_cancel_ratio:
                violations.append(
                    ExecutionConstraintViolation(
                        intent_id="<batch>",
                        symbol="<batch>",
                        constraint="max_cancel_ratio",
                        severity="block",
                        reason=f"ratio_{ratio:.3f}_above_{c.max_cancel_ratio:.3f}",
                        detail={"n_cancels": n_cancels, "n_submits": n_submits},
                    )
                )

        # Single-order value
        if c.max_single_order_value is not None:
            for it in ordered:
                value = float(it.order_value) if it.order_value > 0 else float(it.quantity) * float(it.price)
                if value > c.max_single_order_value:
                    violations.append(
                        ExecutionConstraintViolation(
                            intent_id=it.intent_id,
                            symbol=it.symbol,
                            constraint="max_single_order_value",
                            severity="block",
                            reason=f"value_{value:.0f}_above_{c.max_single_order_value:.0f}",
                            detail={"order_value": value},
                        )
                    )

        # Per-stock participation rate
        if c.max_single_stock_participation_rate is not None:
            volume_by_symbol: dict[str, int] = {}
            day_volume_hint: dict[str, float] = {}
            for it in ordered:
                volume_by_symbol[it.symbol] = volume_by_symbol.get(it.symbol, 0) + max(0, int(it.quantity))
                if it.daily_volume_hint is not None and it.daily_volume_hint > 0:
                    day_volume_hint[it.symbol] = float(it.daily_volume_hint)
            for symbol, vol in volume_by_symbol.items():
                dvol = day_volume_hint.get(symbol)
                if not dvol:
                    continue
                rate = vol / dvol
                if rate > c.max_single_stock_participation_rate:
                    violations.append(
                        ExecutionConstraintViolation(
                            intent_id="<symbol>",
                            symbol=symbol,
                            constraint="max_single_stock_participation_rate",
                            severity="block",
                            reason=f"rate_{rate:.3f}_above_{c.max_single_stock_participation_rate:.3f}",
                            detail={"intent_volume": vol, "day_volume_hint": dvol},
                        )
                    )

        # Daily turnover (as % of NAV)
        if c.max_daily_turnover is not None:
            navs = [it.portfolio_nav for it in ordered if it.portfolio_nav is not None and it.portfolio_nav > 0]
            if navs:
                nav = float(navs[0])
                gross_value = sum(
                    float(it.order_value) if it.order_value > 0 else float(it.quantity) * float(it.price)
                    for it in ordered if it.side in ("buy", "sell")
                )
                turnover = gross_value / nav
                if turnover > c.max_daily_turnover:
                    violations.append(
                        ExecutionConstraintViolation(
                            intent_id="<batch>",
                            symbol="<batch>",
                            constraint="max_daily_turnover",
                            severity="block",
                            reason=f"turnover_{turnover:.3f}_above_{c.max_daily_turnover:.3f}",
                            detail={"gross_value": gross_value, "nav": nav},
                        )
                    )

        # Resting time + auction-mode rules
        # Group cancels by their parent intent timestamp
        intent_by_id = {it.intent_id: it for it in ordered}
        if c.min_order_resting_time_seconds is not None:
            for it in ordered:
                if it.side == "cancel" and it.parent_intent_id:
                    parent = intent_by_id.get(it.parent_intent_id)
                    if parent is None:
                        continue
                    rest = (it.timestamp - parent.timestamp).total_seconds()
                    floor = float(c.min_order_resting_time_seconds)
                    # tighter floor during auction
                    if c.auction_mode_min_resting_time_seconds and classify_auction_phase(parent.timestamp) in (
                        AuctionPhase.PRE_AUCTION_OPEN, AuctionPhase.PRE_AUCTION_CLOSE,
                    ):
                        floor = float(c.auction_mode_min_resting_time_seconds)
                    if rest < floor:
                        violations.append(
                            ExecutionConstraintViolation(
                                intent_id=it.intent_id,
                                symbol=it.symbol,
                                constraint="min_order_resting_time_seconds",
                                severity="block",
                                reason=f"rest_{rest:.3f}_below_{floor:.3f}",
                                detail={"parent_intent_id": parent.intent_id, "rest_seconds": rest},
                            )
                        )

        # Auction-mode per-symbol max orders
        if c.auction_mode_max_orders_per_symbol is not None:
            auction_counts: dict[tuple[str, str], int] = {}
            for it in ordered:
                phase = classify_auction_phase(it.timestamp)
                if phase not in (AuctionPhase.PRE_AUCTION_OPEN, AuctionPhase.PRE_AUCTION_CLOSE):
                    continue
                key = (it.symbol, phase.value)
                auction_counts[key] = auction_counts.get(key, 0) + 1
            for (symbol, phase), count in auction_counts.items():
                if count > c.auction_mode_max_orders_per_symbol:
                    violations.append(
                        ExecutionConstraintViolation(
                            intent_id="<auction>",
                            symbol=symbol,
                            constraint="auction_mode_max_orders_per_symbol",
                            severity="block",
                            reason=f"count_{count}_above_{c.auction_mode_max_orders_per_symbol}",
                            detail={"phase": phase, "count": count},
                        )
                    )

        # Spoofing heuristic — N cancels of the same symbol within one minute
        if c.no_spoofing:
            by_min: dict[tuple[str, pd.Timestamp], int] = {}
            for it in ordered:
                if it.side != "cancel":
                    continue
                key = (it.symbol, it.timestamp.floor("min") if hasattr(it.timestamp, "floor") else it.timestamp)
                by_min[key] = by_min.get(key, 0) + 1
            for (symbol, minute), count in by_min.items():
                if count >= c.spoof_max_repeated_cancels:
                    violations.append(
                        ExecutionConstraintViolation(
                            intent_id="<heuristic>",
                            symbol=symbol,
                            constraint="no_spoofing",
                            severity="block",
                            reason=f"repeated_cancels_{count}_at_or_above_{c.spoof_max_repeated_cancels}",
                            detail={"minute": str(minute), "count": count},
                        )
                    )

        # Layering heuristic — N concurrent BUYs (or SELLs) at distinct price
        # levels of same symbol within one second
        if c.no_layering:
            by_sec_symbol_side: dict[tuple[str, str, pd.Timestamp], set[float]] = {}
            for it in ordered:
                if it.side not in ("buy", "sell"):
                    continue
                key = (
                    it.symbol,
                    it.side,
                    it.timestamp.floor("s") if hasattr(it.timestamp, "floor") else it.timestamp,
                )
                by_sec_symbol_side.setdefault(key, set()).add(float(it.price))
            for (symbol, side, second), levels in by_sec_symbol_side.items():
                if len(levels) >= c.layering_max_concurrent_levels:
                    violations.append(
                        ExecutionConstraintViolation(
                            intent_id="<heuristic>",
                            symbol=symbol,
                            constraint="no_layering",
                            severity="block",
                            reason=f"levels_{len(levels)}_at_or_above_{c.layering_max_concurrent_levels}",
                            detail={"second": str(second), "side": side, "levels": sorted(levels)},
                        )
                    )

        # Pull-push heuristic — sudden ramp in order size from same symbol/side
        if c.no_pull_push:
            by_symbol_side: dict[tuple[str, str], list[OrderIntentRecord]] = {}
            for it in ordered:
                if it.side not in ("buy", "sell"):
                    continue
                by_symbol_side.setdefault((it.symbol, it.side), []).append(it)
            for (symbol, side), records in by_symbol_side.items():
                for prev, nxt in zip(records, records[1:]):
                    if prev.quantity <= 0:
                        continue
                    if nxt.quantity >= prev.quantity * c.pull_push_min_size_jump:
                        violations.append(
                            ExecutionConstraintViolation(
                                intent_id=nxt.intent_id,
                                symbol=symbol,
                                constraint="no_pull_push",
                                severity="block",
                                reason=f"size_jump_{nxt.quantity}/{prev.quantity}>={c.pull_push_min_size_jump}",
                                detail={
                                    "prev_quantity": prev.quantity,
                                    "next_quantity": nxt.quantity,
                                    "side": side,
                                },
                            )
                        )

        by_constraint: dict[str, int] = {}
        for v in violations:
            by_constraint[v.constraint] = by_constraint.get(v.constraint, 0) + 1
        n_blocking = sum(1 for v in violations if v.severity == "block")
        return ExecutionConstraintReport(
            n_intents=n_total,
            n_violations=len(violations),
            n_blocking=n_blocking,
            violations=violations,
            by_constraint=by_constraint,
        )


__all__ = [
    "AuctionPhase",
    "ExecutionConstraintEvaluator",
    "ExecutionConstraintReport",
    "ExecutionConstraintSet",
    "ExecutionConstraintViolation",
    "OrderIntentRecord",
    "classify_auction_phase",
]
