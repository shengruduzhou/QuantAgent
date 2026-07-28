"""Global operating mode and the live-order rejection boundary.

This module is the single place that answers "may this system send a real
order?" and the answer is structurally no. Not "no by configuration", not "no
unless an environment variable is set" -- there is no mode in which a live order
path becomes reachable, and :data:`LIVE_DISABLED` exists precisely to make that
a stated policy rather than an omission.

Why a terminal state rather than a missing feature: an absent capability is
indistinguishable from an unfinished one. A named, asserted, non-executable
state is auditable -- tests can prove the system is in it, and the UI can
display it.

The rejection happens **before** agent consultation, broker routing or job
creation. Checking later would mean an intent to trade real money had already
been reasoned about, queued, or partially executed by the time it was refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

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
#: Terminal policy state. **Not executable.** Its only purpose is to be
#: assertable: the system can prove live trading is unavailable.
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
    "NO BROKER CONNECTION",
    "NO REAL ORDERS",
)


class LiveTradingRejected(RuntimeError):
    """Raised when a request carries live-order intent. Never catchable-and-continue."""

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

#: Broker SDK entry points that would transmit an order. Their mere appearance
#: in a request is treated as intent, because there is no benign reason for a
#: research workstation request to name them.
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


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def scan_for_live_intent(payload: Any, *, where: str = "request") -> str | None:
    """Return the first live-intent marker found in ``payload``, or ``None``.

    Scans strings, and recursively the keys and values of mappings and
    sequences, because intent can arrive in a nested job parameter just as
    easily as in a top-level prompt.
    """
    def _scan(value: Any) -> str | None:
        if isinstance(value, str):
            haystack = _normalise(value)
            for exempt in EXEMPT_CONTEXTS:
                haystack = haystack.replace(exempt, "")
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
    """Refuse a request carrying live-order intent.

    Called at every entry point *before* anything else happens -- before agent
    consultation, before broker routing, before a job record is created.
    """
    matched = scan_for_live_intent(payload, where=where)
    if matched is None:
        return
    raise LiveTradingRejected(
        matched, where,
        f"refusing {where}: it references {matched!r}, which denotes real-money "
        f"trading. This system operates in {LIVE_DISABLED} and has no live order "
        "path; simulated orders exist only inside the local paper broker.",
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
                "live_trading_available cannot be True: this build has no live "
                "order path, and a flag claiming otherwise would be a lie the UI "
                "would faithfully display"
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
                "no job may run in it"
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
    """The terminal state proving live trading is unavailable."""
    return OperatingModeState(mode=LIVE_DISABLED)


def describe_policy() -> dict[str, Any]:
    """Machine-readable statement of the safety boundary, for UI and reports."""
    return {
        "banner": POLICY_BANNER,
        "paperBannerLines": list(PAPER_BANNER_LINES),
        "operatingModes": list(OPERATING_MODES),
        "executableModes": sorted(EXECUTABLE_MODES),
        "orderSimulatingModes": sorted(ORDER_SIMULATING_MODES),
        "liveTradingAvailable": False,
        "liveTradingCertificate": "NOT_IMPLEMENTED_BY_POLICY",
        "forbiddenApis": list(FORBIDDEN_API_MARKERS),
        "rejectionPoint": "before agent consultation, broker routing and job creation",
        "guarantees": [
            "no live broker order transmission",
            "no real account identifiers",
            "no broker credentials",
            "no account login workflow",
            "no automatic transition from paper to live",
            "no free-form shell command from the web UI",
        ],
    }
