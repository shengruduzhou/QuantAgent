"""Global operating mode and the live-order rejection boundary.

This module is the single place that answers "may an ordinary QuantAgent job,
agent, API request or UI action send a real order?" and the answer remains
structurally no.  A low-level, audited QMT adapter now exists so its broker
contract can be tested and eventually exercised on a controlled trading host,
but there is still no executable LIVE operating mode and no job/API/agent path
that may arm that adapter.

That distinction matters for auditability.  Claiming that broker-call code does
not exist would now be false; claiming that real-money transmission is not an
executable product capability is accurate and testable.  :data:`LIVE_DISABLED`
is therefore a policy/arming state, not a statement about source-code absence.

The rejection happens **before** agent consultation, broker routing or job
creation. Checking later would mean an intent to trade real money had already
been reasoned about, queued, or partially executed by the time it was refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# operating modes
# ---------------------------------------------------------------------------
#: Offline analysis over stored data. No clock, no orders.
RESEARCH = "RESEARCH"
#: Deterministic replay of a historical window through the local broker.
HISTORICAL_REPLAY = "HISTORICAL_REPLAY"
#: Walk-forward evaluation. Produces metrics, never orders.
BACKTEST = "BACKTEST"
#: Simulated orders against the local broker using current-day data.
PAPER = "PAPER"
#: Signals and target positions generated, orders simulated but not acted on.
SHADOW = "SHADOW"
#: Terminal policy state. **Not executable.** Its purpose is to prove that no
#: normal QuantAgent surface is armed for real-money transmission.
LIVE_DISABLED = "LIVE_DISABLED"

OPERATING_MODES: tuple[str, ...] = (
    RESEARCH, HISTORICAL_REPLAY, BACKTEST, PAPER, SHADOW, LIVE_DISABLED,
)

#: Modes a job may actually run in.
EXECUTABLE_MODES: frozenset[str] = frozenset(
    {RESEARCH, HISTORICAL_REPLAY, BACKTEST, PAPER, SHADOW}
)

#: Modes in which simulated orders are produced. All of them are local.
ORDER_SIMULATING_MODES: frozenset[str] = frozenset(
    {HISTORICAL_REPLAY, PAPER, SHADOW}
)

#: Displayed by every surface, always.
POLICY_BANNER = "LIVE TRADING: DISABLED BY POLICY"
PAPER_BANNER_LINES: tuple[str, ...] = (
    "LOCAL PAPER ONLY",
    "NO ARMED BROKER ROUTE",
    "NO REAL ORDERS FROM PRODUCT SURFACES",
)


class LiveTradingRejected(RuntimeError):
    """Raised when a product-surface request carries live-order intent."""

    def __init__(self, matched: str, where: str, message: str) -> None:
        super().__init__(message)
        self.matched = matched
        self.where = where


class ModeViolation(RuntimeError):
    """Raised when an action is attempted in a mode that does not permit it."""


# ---------------------------------------------------------------------------
# live-intent detection
# ---------------------------------------------------------------------------
#: Chinese phrases that denote real-money intent. Listed explicitly because a
#: transliteration or keyword-stem approach would miss them.
CHINESE_LIVE_MARKERS: tuple[str, ...] = (
    "实盘", "真实下单", "连接资金账户", "自动买入", "自动卖出", "真实委托",
    "真实交易", "资金账户", "实盘交易", "真实账户", "下真单",
)

#: English/API markers. These are matched case-insensitively.
ENGLISH_LIVE_MARKERS: tuple[str, ...] = (
    "live trading", "live_trading", "real order", "real_order",
    "real money", "real-money", "real account", "real_account",
    "broker login", "broker_login", "account login", "place real",
    "enable_live", "enable live", "go live", "production trading",
)

#: Broker SDK entry points that would transmit an order. Their appearance in a
#: normal user/job request is treated as intent.  The audited adapter may name
#: them in source; that is a separate static-code boundary enforced by tests.
FORBIDDEN_API_MARKERS: tuple[str, ...] = (
    "order_stock_async", "order_stock", "xtquanttrader", "xt_trader",
    "order_send", "ordersendasync", "cancel_order_stock",
    "connect_broker", "brokerapi", "tradeapi.buy", "tradeapi.sell",
)

ALL_LIVE_MARKERS: tuple[str, ...] = (
    CHINESE_LIVE_MARKERS + ENGLISH_LIVE_MARKERS + FORBIDDEN_API_MARKERS
)

#: Read-only surfaces whose names legitimately contain a forbidden substring.
#: Kept deliberately small and explicit -- an over-broad exemption list is how a
#: real order path eventually slips through.
EXEMPT_CONTEXTS: frozenset[str] = frozenset({
    "l2order", "l2orderqueue", "order_event", "order_book", "orderbook",
    "order_flow", "orders.parquet", "order_ledger", "simulated_order",
    "paper_order", "order_intent",
})


def _haystacks(text: str) -> tuple[str, ...]:
    """Normalised variants of ``text`` to match markers against."""
    lowered = re.sub(r"\s+", " ", str(text)).strip().lower()
    flattened = re.sub(r"[-_./]+", " ", lowered)
    flattened = re.sub(r"\s+", " ", flattened).strip()
    return (lowered, flattened) if flattened != lowered else (lowered,)


def scan_for_live_intent(payload: Any, *, where: str = "request") -> str | None:
    """Return the first live-intent marker found in ``payload``, or ``None``.

    Scans strings, and recursively the keys and values of mappings and
    sequences, because intent can arrive in a nested job parameter just as
    easily as in a top-level prompt.
    """
    def _scan(value: Any) -> str | None:
        if isinstance(value, str):
            for haystack in _haystacks(value):
                for exempt in EXEMPT_CONTEXTS:
                    haystack = haystack.replace(exempt, "")
                    haystack = haystack.replace(exempt.replace("_", " "), "")
                for marker in ALL_LIVE_MARKERS:
                    if marker.lower() in haystack:
                        return marker
            return None
        if isinstance(value, Mapping):
            for key, item in value.items():
                hit = _scan(str(key)) or _scan(item)
                if hit:
                    return hit
            return None
        if isinstance(value, (list, tuple, set)):
            for item in value:
                hit = _scan(item)
                if hit:
                    return hit
            return None
        return None

    return _scan(payload)


def reject_live_intent(payload: Any, *, where: str = "request") -> None:
    """Refuse a product-surface request carrying live-order intent.

    Called at every normal entry point *before* anything else happens -- before
    agent consultation, broker routing, or a job record is created.  Controlled
    trading-host certification code is intentionally not exposed through this
    free-form request surface.
    """
    matched = scan_for_live_intent(payload, where=where)
    if matched is None:
        return
    raise LiveTradingRejected(
        matched, where,
        f"refusing {where}: it references {matched!r}, which denotes real-money "
        f"trading. This build operates in {LIVE_DISABLED}: no executable product "
        "mode, web job or agent route is armed for broker transmission. An audited "
        "QMT adapter may exist for controlled certification, but it is not an "
        "end-user live-trading capability.",
    )


# ---------------------------------------------------------------------------
# mode state
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OperatingModeState:
    """The system's current mode plus the invariants that always hold."""

    mode: str
    declared_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    live_trading_available: bool = False
    banner: str = POLICY_BANNER

    def __post_init__(self) -> None:
        if self.mode not in OPERATING_MODES:
            raise ValueError(
                f"unknown operating mode {self.mode!r}; known: {list(OPERATING_MODES)}"
            )
        if self.live_trading_available:
            raise ValueError(
                "live_trading_available cannot be True: this build has no "
                "executable LIVE operating mode or armed product route. The "
                "presence of a controlled broker adapter does not satisfy that "
                "production-readiness contract."
            )

    @property
    def executable(self) -> bool:
        return self.mode in EXECUTABLE_MODES

    @property
    def simulates_orders(self) -> bool:
        return self.mode in ORDER_SIMULATING_MODES

    def require_executable(self) -> None:
        if not self.executable:
            raise ModeViolation(
                f"{self.mode} is a terminal policy state, not an executable mode; "
                "no ordinary job may run in it"
            )

    def require_order_simulation(self) -> None:
        self.require_executable()
        if not self.simulates_orders:
            raise ModeViolation(
                f"{self.mode} does not simulate orders; use one of "
                f"{sorted(ORDER_SIMULATING_MODES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "executable": self.executable,
            "simulatesOrders": self.simulates_orders,
            "paperBannerLines": list(PAPER_BANNER_LINES),
        }


def policy_state() -> OperatingModeState:
    """The terminal state proving product-surface live trading is unavailable."""
    return OperatingModeState(mode=LIVE_DISABLED)


def describe_policy() -> dict[str, Any]:
    """Machine-readable statement of the safety/arming boundary for UI/reports."""
    return {
        "banner": POLICY_BANNER,
        "paperBannerLines": list(PAPER_BANNER_LINES),
        "operatingModes": list(OPERATING_MODES),
        "executableModes": sorted(EXECUTABLE_MODES),
        "orderSimulatingModes": sorted(ORDER_SIMULATING_MODES),
        "liveTradingAvailable": False,
        "liveTradingCertificate": "NOT_IMPLEMENTED_BY_POLICY",
        "controlledBrokerAdapterPresent": True,
        "controlledBrokerAdapterArmed": False,
        "forbiddenApis": list(FORBIDDEN_API_MARKERS),
        "rejectionPoint": "before agent consultation, broker routing and job creation",
        "guarantees": [
            "no real-money transmission reachable from executable operating modes",
            "no web job or agent route may arm the controlled broker adapter",
            "no broker credentials accepted from free-form product requests",
            "no automatic transition from paper/shadow to live",
            "no free-form shell command from the web UI",
            "current live model trust certificate remains independently fail-closed",
        ],
    }
