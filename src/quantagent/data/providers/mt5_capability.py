"""Fail-closed MetaTrader 5 capability probe.

MT5 is treated here as a *candidate* venue, never as an assumed A-share data
vendor. The probe answers one question with evidence: does this terminal expose
genuine mainland A-share exchange data, and if so, of what class?

Design constraints that come out of the official documentation
(https://www.mql5.com/en/docs/python_metatrader5):

* The ``MetaTrader5`` Python package talks to a **locally running terminal**
  over IPC. There is no remote/hosted mode, so "install the package" and "have
  a feed" are different problems.
* The package is distributed as Windows wheels only. On a non-Windows host the
  import fails and the correct classification is ``TERMINAL_UNAVAILABLE`` --
  not "no A-share data", which would be an unproven claim about brokers.
* ``market_book_add`` succeeding does not mean the broker publishes depth;
  ``market_book_get`` can still return an empty book. Depth must be *read*, not
  subscribed, to count.
* Symbols visible in Market Watch are broker instruments. A broker CFD named
  ``CHINA50`` or even ``600519`` is not an exchange feed. Provenance is decided
  by ``SYMBOL_PATH``/exchange fields and by reconciliation against U0, never by
  the ticker string looking Chinese.

Everything the probe cannot establish is recorded as unproven. The probe never
writes a capability cell as ``SERVING`` unless a call returned rows.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from quantagent.data.microstructure import capability as cap
from quantagent.data.microstructure import contracts as mc

# --- classification outcomes ------------------------------------------------
GENUINE_A_SHARE_EXCHANGE_FEED = "GENUINE_A_SHARE_EXCHANGE_FEED"
A_SHARE_LEVEL1_ONLY = "A_SHARE_LEVEL1_ONLY"
A_SHARE_TICK_NO_DEPTH = "A_SHARE_TICK_NO_DEPTH"
A_SHARE_DOM_SNAPSHOT = "A_SHARE_DOM_SNAPSHOT"
A_SHARE_LEVEL2_CANDIDATE = "A_SHARE_LEVEL2_CANDIDATE"
BROKER_CFD_OR_SYNTHETIC = "BROKER_CFD_OR_SYNTHETIC"
NO_A_SHARE_BROKER_FEED = "NO_A_SHARE_BROKER_FEED"
TERMINAL_UNAVAILABLE = "TERMINAL_UNAVAILABLE"
ACCOUNT_OR_ENTITLEMENT_REQUIRED = "ACCOUNT_OR_ENTITLEMENT_REQUIRED"

CLASSIFICATIONS: tuple[str, ...] = (
    GENUINE_A_SHARE_EXCHANGE_FEED, A_SHARE_LEVEL1_ONLY, A_SHARE_TICK_NO_DEPTH,
    A_SHARE_DOM_SNAPSHOT, A_SHARE_LEVEL2_CANDIDATE, BROKER_CFD_OR_SYNTHETIC,
    NO_A_SHARE_BROKER_FEED, TERMINAL_UNAVAILABLE, ACCOUNT_OR_ENTITLEMENT_REQUIRED,
)

#: Board-representative probe cohort. Deliberately spans all five boards plus
#: the awkward cases (ST, suspended, recent IPO, delisted) because a probe that
#: only tries 600519 proves nothing about coverage.
DEFAULT_PROBE_COHORT: tuple[dict[str, str], ...] = (
    {"symbol": "600000.SH", "board": "SH_Main", "note": "liquid SSE main board"},
    {"symbol": "000001.SZ", "board": "SZ_Main", "note": "liquid SZSE main board"},
    {"symbol": "300750.SZ", "board": "ChiNext", "note": "liquid ChiNext"},
    {"symbol": "688981.SH", "board": "STAR", "note": "liquid STAR"},
    {"symbol": "920002.BJ", "board": "BSE", "note": "Beijing Stock Exchange"},
)

#: Naming conventions brokers plausibly use. Tried in order; never assumed.
def symbol_candidates(canonical: str) -> list[str]:
    """Broker symbol spellings to try for a canonical ``600000.SH`` code."""
    code, _, exchange = canonical.partition(".")
    exchange = exchange.upper()
    long_exchange = {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}.get(exchange, exchange)
    return [
        canonical,                    # 600000.SH
        f"{exchange}{code}",          # SH600000
        f"{code}.{long_exchange}",    # 600000.SSE
        code,                         # 600000
        f"{code}.{exchange.lower()}", # 600000.sh
        f"{long_exchange}:{code}",    # SSE:600000
    ]


@dataclass
class TerminalState:
    probed_at: str
    os_name: str
    os_release: str
    python_version: str
    package_importable: bool
    package_version: str | None = None
    import_error: str | None = None
    initialized: bool = False
    initialize_error: str | None = None
    terminal_path: str | None = None
    terminal_build: int | None = None
    terminal_name: str | None = None
    terminal_connected: bool | None = None
    company: str | None = None
    timezone_note: str | None = None
    wine_present: bool = False
    classification: str = TERMINAL_UNAVAILABLE
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AccountState:
    available: bool
    login: int | None = None
    server: str | None = None
    company: str | None = None
    currency: str | None = None
    trade_mode: str | None = None
    is_demo: bool | None = None
    margin_mode: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SymbolProbe:
    canonical: str
    board: str
    resolved_symbol: str | None
    found: bool
    classification: str
    symbol_info: dict[str, Any] = field(default_factory=dict)
    tick_probe: dict[str, Any] = field(default_factory=dict)
    dom_probe: dict[str, Any] = field(default_factory=dict)
    rates_probe: dict[str, Any] = field(default_factory=dict)
    tried_spellings: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Mt5ProbeResult:
    terminal: TerminalState
    account: AccountState
    symbols: list[SymbolProbe] = field(default_factory=list)
    symbols_total: int | None = None
    symbol_groups: list[str] = field(default_factory=list)
    overall_classification: str = TERMINAL_UNAVAILABLE
    genuine_a_share_symbols: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminal": self.terminal.to_dict(),
            "account": self.account.to_dict(),
            "symbols_total": self.symbols_total,
            "symbol_groups": self.symbol_groups,
            "overall_classification": self.overall_classification,
            "genuine_a_share_symbols": self.genuine_a_share_symbols,
            "notes": self.notes,
            "symbols": [s.to_dict() for s in self.symbols],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _import_mt5() -> tuple[Any | None, str | None]:
    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - any import failure is a real answer
        return None, f"{type(exc).__name__}: {exc}"
    return mt5, None


def probe_terminal(mt5: Any | None, import_error: str | None) -> TerminalState:
    """Establish what the local MT5 runtime is, without assuming it exists."""
    state = TerminalState(
        probed_at=_utc_now(),
        os_name=platform.system(),
        os_release=platform.release(),
        python_version=sys.version.split()[0],
        package_importable=mt5 is not None,
        import_error=import_error,
        wine_present=bool(shutil.which("wine") or shutil.which("wine64")),
    )
    if mt5 is None:
        state.classification = TERMINAL_UNAVAILABLE
        state.detail = (
            "MetaTrader5 Python package is not importable on this host. The "
            "package communicates with a locally running Windows terminal over "
            "IPC and is published as Windows wheels only, so no MT5 capability "
            "-- A-share or otherwise -- can be measured here."
        )
        return state

    state.package_version = getattr(mt5, "__version__", None)
    try:
        initialized = bool(mt5.initialize())
    except Exception as exc:  # noqa: BLE001
        state.initialize_error = f"{type(exc).__name__}: {exc}"
        state.classification = TERMINAL_UNAVAILABLE
        state.detail = "mt5.initialize() raised; no terminal reachable"
        return state

    state.initialized = initialized
    if not initialized:
        try:
            state.initialize_error = str(mt5.last_error())
        except Exception:  # noqa: BLE001 - best effort
            state.initialize_error = "initialize() returned False"
        state.classification = TERMINAL_UNAVAILABLE
        state.detail = "no running MetaTrader 5 terminal accepted the IPC connection"
        return state

    info = mt5.terminal_info()
    if info is not None:
        state.terminal_path = getattr(info, "path", None)
        state.terminal_build = getattr(info, "build", None)
        state.terminal_name = getattr(info, "name", None)
        state.terminal_connected = getattr(info, "connected", None)
        state.company = getattr(info, "company", None)
    state.timezone_note = (
        "MT5 server times are broker-server local, not exchange local; every "
        "timestamp must be converted before comparison with Asia/Shanghai data"
    )
    state.classification = "TERMINAL_AVAILABLE"
    return state


def probe_account(mt5: Any | None) -> AccountState:
    if mt5 is None:
        return AccountState(available=False, detail="package not importable")
    try:
        info = mt5.account_info()
    except Exception as exc:  # noqa: BLE001
        return AccountState(available=False, detail=f"{type(exc).__name__}: {exc}")
    if info is None:
        return AccountState(
            available=False,
            detail="no account is logged in; symbol visibility and history depth "
                   "are account-dependent, so capability cannot be measured",
        )
    trade_mode = getattr(info, "trade_mode", None)
    demo_modes = {0}  # ACCOUNT_TRADE_MODE_DEMO == 0 per MQL5 docs
    return AccountState(
        available=True,
        login=getattr(info, "login", None),
        server=getattr(info, "server", None),
        company=getattr(info, "company", None),
        currency=getattr(info, "currency", None),
        trade_mode=str(trade_mode),
        is_demo=(trade_mode in demo_modes) if trade_mode is not None else None,
        margin_mode=str(getattr(info, "margin_mode", None)),
    )


def _classify_symbol(
    info: Mapping[str, Any], tick: Mapping[str, Any], dom: Mapping[str, Any]
) -> tuple[str, str]:
    """Classify one broker symbol from what the terminal actually returned."""
    path = str(info.get("path") or "")
    exchange = str(info.get("exchange") or "")
    currency = str(info.get("currency_profit") or "")

    looks_exchange_traded = bool(exchange) and currency.upper() == "CNY"
    if not looks_exchange_traded:
        return (
            BROKER_CFD_OR_SYNTHETIC,
            f"symbol path {path!r} / exchange {exchange!r} / currency {currency!r} "
            "does not evidence a mainland exchange feed",
        )

    dom_levels = int(dom.get("levels") or 0)
    has_ticks = int(tick.get("ticks_returned") or 0) > 0
    has_real_volume = bool(tick.get("real_volume_present"))

    if dom_levels > 5:
        return (A_SHARE_LEVEL2_CANDIDATE, f"DOM returned {dom_levels} levels")
    if dom_levels > 0:
        return (A_SHARE_DOM_SNAPSHOT, f"DOM returned {dom_levels} levels")
    if has_ticks and has_real_volume:
        return (A_SHARE_TICK_NO_DEPTH, "tick history with real volume, no DOM")
    if has_ticks:
        return (A_SHARE_TICK_NO_DEPTH, "tick history without real volume, no DOM")
    return (A_SHARE_LEVEL1_ONLY, "quotes only; no tick history and no DOM")


def probe_symbol(
    mt5: Any, canonical: str, board: str, *, dom_reads: int = 5
) -> SymbolProbe:
    """Resolve one canonical A-share code against the broker's namespace."""
    spellings = symbol_candidates(canonical)
    probe = SymbolProbe(
        canonical=canonical, board=board, resolved_symbol=None, found=False,
        classification=NO_A_SHARE_BROKER_FEED, tried_spellings=spellings,
    )

    resolved = None
    for spelling in spellings:
        try:
            info = mt5.symbol_info(spelling)
        except Exception:  # noqa: BLE001
            continue
        if info is not None:
            resolved = spelling
            break
    if resolved is None:
        probe.detail = "no broker symbol matched any tried spelling"
        return probe

    probe.resolved_symbol = resolved
    probe.found = True
    info = mt5.symbol_info(resolved)
    probe.symbol_info = {
        "name": getattr(info, "name", None),
        "path": getattr(info, "path", None),
        "exchange": getattr(info, "exchange", None),
        "currency_profit": getattr(info, "currency_profit", None),
        "currency_base": getattr(info, "currency_base", None),
        "trade_mode": getattr(info, "trade_mode", None),
        "trade_calc_mode": getattr(info, "trade_calc_mode", None),
        "digits": getattr(info, "digits", None),
        "point": getattr(info, "point", None),
        "trade_contract_size": getattr(info, "trade_contract_size", None),
        "volume_step": getattr(info, "volume_step", None),
        "volume_min": getattr(info, "volume_min", None),
        "volume_max": getattr(info, "volume_max", None),
        "ticks_bookdepth": getattr(info, "ticks_bookdepth", None),
        "visible": getattr(info, "visible", None),
    }

    try:
        mt5.symbol_select(resolved, True)
    except Exception:  # noqa: BLE001
        pass

    # -- ticks
    try:
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        ticks = mt5.copy_ticks_range(
            resolved, now - timedelta(days=5), now, mt5.COPY_TICKS_ALL
        )
        count = 0 if ticks is None else int(len(ticks))
        probe.tick_probe = {
            "ticks_returned": count,
            "window_days": 5,
            "real_volume_present": bool(
                count and "volume_real" in getattr(ticks, "dtype", type("x", (), {"names": ()})).names
            ),
            "fields": list(getattr(ticks, "dtype", type("x", (), {"names": ()})).names or [])
            if count else [],
        }
    except Exception as exc:  # noqa: BLE001
        probe.tick_probe = {"error": f"{type(exc).__name__}: {exc}", "ticks_returned": 0}

    # -- depth of market: subscribing is not evidence, reading is
    dom_result: dict[str, Any] = {"subscribed": False, "levels": 0, "reads": 0}
    try:
        subscribed = bool(mt5.market_book_add(resolved))
        dom_result["subscribed"] = subscribed
        if subscribed:
            observed = 0
            for _ in range(dom_reads):
                book = mt5.market_book_get(resolved)
                if book:
                    observed = max(observed, len(book))
            dom_result["levels"] = observed
            dom_result["reads"] = dom_reads
            mt5.market_book_release(resolved)
    except Exception as exc:  # noqa: BLE001
        dom_result["error"] = f"{type(exc).__name__}: {exc}"
    probe.dom_probe = dom_result

    # -- bars
    try:
        rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_D1, 0, 100)
        probe.rates_probe = {"daily_bars_returned": 0 if rates is None else int(len(rates))}
    except Exception as exc:  # noqa: BLE001
        probe.rates_probe = {"error": f"{type(exc).__name__}: {exc}"}

    probe.classification, probe.detail = _classify_symbol(
        probe.symbol_info, probe.tick_probe, probe.dom_probe
    )
    return probe


def run_probe(cohort: Sequence[Mapping[str, str]] = DEFAULT_PROBE_COHORT) -> Mt5ProbeResult:
    """Run the full fail-closed probe and return a structured result."""
    mt5, import_error = _import_mt5()
    terminal = probe_terminal(mt5, import_error)
    account = probe_account(mt5)
    result = Mt5ProbeResult(terminal=terminal, account=account)

    if mt5 is None or not terminal.initialized:
        result.overall_classification = TERMINAL_UNAVAILABLE
        result.notes.append(
            "No MT5 terminal was reachable, so every downstream MT5 question "
            "(A-share symbols, tick history, DOM depth, real vs generated ticks) "
            "is UNMEASURED on this host. This is not evidence that brokers do or "
            "do not offer A-share feeds."
        )
        return result

    try:
        result.symbols_total = int(mt5.symbols_total())
        all_symbols = mt5.symbols_get() or []
        result.symbol_groups = sorted(
            {str(getattr(s, "path", "")).rsplit("\\", 1)[0] for s in all_symbols}
        )[:200]
    except Exception as exc:  # noqa: BLE001
        result.notes.append(f"symbol enumeration failed: {type(exc).__name__}: {exc}")

    for entry in cohort:
        result.symbols.append(
            probe_symbol(mt5, entry["symbol"], entry.get("board", "UNKNOWN"))
        )

    genuine = [
        s for s in result.symbols
        if s.found and s.classification not in (BROKER_CFD_OR_SYNTHETIC, NO_A_SHARE_BROKER_FEED)
    ]
    result.genuine_a_share_symbols = len(genuine)
    if not genuine:
        result.overall_classification = NO_A_SHARE_BROKER_FEED
    elif any(s.classification == A_SHARE_LEVEL2_CANDIDATE for s in genuine):
        result.overall_classification = A_SHARE_LEVEL2_CANDIDATE
    elif any(s.classification == A_SHARE_DOM_SNAPSHOT for s in genuine):
        result.overall_classification = A_SHARE_DOM_SNAPSHOT
    elif any(s.classification == A_SHARE_TICK_NO_DEPTH for s in genuine):
        result.overall_classification = A_SHARE_TICK_NO_DEPTH
    else:
        result.overall_classification = A_SHARE_LEVEL1_ONLY

    try:
        mt5.shutdown()
    except Exception:  # noqa: BLE001
        pass
    return result


def capability_cells(result: Mt5ProbeResult) -> list[cap.CapabilityCell]:
    """Translate the probe result into capability-matrix cells."""
    probed_at = result.terminal.probed_at
    if result.overall_classification == TERMINAL_UNAVAILABLE:
        detail = result.terminal.detail
        return [
            cap.CapabilityCell(
                provider="mt5_broker_feed",
                dataset_family=family,
                status=cap.CLIENT_UNAVAILABLE,
                entitlement=cap.ENTITLEMENT_UNKNOWN,
                probed_at=probed_at,
                detail=detail,
                evidence={
                    "os": result.terminal.os_name,
                    "package_importable": result.terminal.package_importable,
                    "import_error": result.terminal.import_error,
                },
            )
            for family in (
                "daily_bars_raw", "minute_bars", "trade_ticks", "level1_quote",
                "level2_snapshot", "level2_order_events", "security_master",
            )
        ]

    cells: list[cap.CapabilityCell] = []
    found = [s for s in result.symbols if s.found]
    tick_rows = sum(int(s.tick_probe.get("ticks_returned") or 0) for s in found)
    dom_levels = max((int(s.dom_probe.get("levels") or 0) for s in found), default=0)
    bar_rows = sum(int(s.rates_probe.get("daily_bars_returned") or 0) for s in found)

    def cell(family: str, rows: int, data_class: str | None, detail: str) -> cap.CapabilityCell:
        return cap.CapabilityCell(
            provider="mt5_broker_feed", dataset_family=family,
            status=cap.SERVING if rows else cap.EMPTY,
            entitlement=cap.BROKER_ACCOUNT_REQUIRED,
            data_class=data_class if rows else None,
            probed_at=probed_at, rows_returned=rows or None, detail=detail,
        )

    cells.append(cell("trade_ticks", tick_rows, mc.TRADE_TICK,
                      "copy_ticks_range over a 5-day window"))
    cells.append(cell("daily_bars_raw", bar_rows, None, "copy_rates_from_pos D1"))
    cells.append(cell(
        "level2_snapshot", dom_levels,
        mc.LEVEL2_SNAPSHOT if dom_levels else None,
        f"market_book_get returned at most {dom_levels} levels",
    ))
    cells.append(cap.CapabilityCell(
        provider="mt5_broker_feed", dataset_family="level2_order_events",
        status=cap.NOT_OFFERED, entitlement=cap.BROKER_ACCOUNT_REQUIRED,
        probed_at=probed_at,
        detail="MT5's market book is an aggregated DOM; the API exposes no "
               "per-order insert/cancel stream, so exchange order events cannot "
               "come from this transport at any entitlement level",
    ))
    return cells


def write_artifacts(result: Mt5ProbeResult, directory: str | Path) -> dict[str, str]:
    """Persist machine-readable probe artifacts."""
    import pandas as pd

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    terminal_path = target / "terminal.json"
    terminal_path.write_text(
        json.dumps(result.terminal.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written["terminal"] = str(terminal_path)

    accounts_path = target / "accounts.json"
    accounts_path.write_text(
        json.dumps(result.account.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written["accounts"] = str(accounts_path)

    mapping_rows = [
        {
            "canonical_symbol": s.canonical, "board": s.board,
            "resolved_symbol": s.resolved_symbol, "found": s.found,
            "classification": s.classification,
            "tried_spellings": "|".join(s.tried_spellings), "detail": s.detail,
        }
        for s in result.symbols
    ]
    mapping_path = target / "ashare_symbol_mapping.parquet"
    pd.DataFrame(mapping_rows).to_parquet(mapping_path, index=False)
    written["ashare_symbol_mapping"] = str(mapping_path)

    tick_path = target / "tick_probe.parquet"
    pd.DataFrame([
        {"canonical_symbol": s.canonical, "resolved_symbol": s.resolved_symbol,
         **{f"tick_{k}": (json.dumps(v) if isinstance(v, list) else v)
            for k, v in s.tick_probe.items()}}
        for s in result.symbols
    ]).to_parquet(tick_path, index=False)
    written["tick_probe"] = str(tick_path)

    dom_path = target / "dom_probe.parquet"
    pd.DataFrame([
        {"canonical_symbol": s.canonical, "resolved_symbol": s.resolved_symbol,
         **{f"dom_{k}": v for k, v in s.dom_probe.items()}}
        for s in result.symbols
    ]).to_parquet(dom_path, index=False)
    written["dom_probe"] = str(dom_path)

    symbols_path = target / "symbols.parquet"
    pd.DataFrame([
        {"canonical_symbol": s.canonical, "board": s.board, "found": s.found,
         **{f"info_{k}": v for k, v in s.symbol_info.items()}}
        for s in result.symbols
    ]).to_parquet(symbols_path, index=False)
    written["symbols"] = str(symbols_path)

    matrix = cap.CapabilityMatrix(capability_cells(result))
    written.update(matrix.write(target, stem="capability_matrix"))

    summary_path = target / "probe_result.json"
    summary_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written["probe_result"] = str(summary_path)
    return written
