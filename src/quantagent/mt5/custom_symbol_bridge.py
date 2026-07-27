"""Export canonical A-share events into MetaTrader 5 custom symbols.

MT5 is used here as a *workstation*, never as a source of truth. The direction
of travel is one-way by design:

    authoritative journal  ->  MT5 custom symbol  ->  charts / Strategy Tester

and never back. :func:`build_import_plan` produces files plus a manifest; there
is no function in this module that reads ticks out of MT5 and writes them into
the journal, because MT5 cannot distinguish what it was given from what it
generated once both are inside the terminal.

Three honesty requirements shape the design.

**Imported data changes class.** Once events live in a custom symbol they are
``CUSTOM_SYMBOL_REPLAY``, whatever they were on the way in. The manifest records
the *origin* class too, so a Strategy Tester result can always be traced back to
whether it rested on exchange trades or on 3-second aggregates.

**Generated ticks are not real ticks.** MT5's Strategy Tester can synthesise
ticks from bars. :func:`classify_tester_ticks` names the modelling mode, and
:data:`REAL_TICK_MODES` is the only set that may be described as real. A tester
run in any other mode is reported as ``GENERATED_TESTER_TICK``.

**Custom symbols must be unmistakable.** Names are prefixed ``QA_`` so a custom
symbol can never be confused with a broker-traded instrument in the same
terminal.

Nothing in this module imports MetaTrader5 at module scope, so it stays
importable -- and testable -- on a host with no terminal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quantagent.data.microstructure import contracts as mc
from quantagent.data.microstructure.store import frame_hash

SYMBOL_PREFIX = "QA_"

# --- MT5 tick flags (MQL5 ENUM_TICK_FLAG) -----------------------------------
TICK_FLAG_BID = 0x02
TICK_FLAG_ASK = 0x04
TICK_FLAG_LAST = 0x08
TICK_FLAG_VOLUME = 0x10
TICK_FLAG_BUY = 0x20
TICK_FLAG_SELL = 0x40

# --- Strategy Tester modelling modes ----------------------------------------
TESTER_EVERY_TICK = "EVERY_TICK"
TESTER_ONE_MINUTE_OHLC = "ONE_MINUTE_OHLC"
TESTER_OPEN_PRICES = "OPEN_PRICES_ONLY"
TESTER_EVERY_TICK_REAL = "EVERY_TICK_BASED_ON_REAL_TICKS"
TESTER_MATH_CALC = "MATH_CALCULATIONS"

#: The only mode in which the tester consumes the ticks it was given rather
#: than manufacturing them from bars.
REAL_TICK_MODES: frozenset[str] = frozenset({TESTER_EVERY_TICK_REAL})


def custom_symbol_name(canonical: str) -> str:
    """``600000.SH`` -> ``QA_600000_SH``."""
    code, _, exchange = canonical.partition(".")
    return f"{SYMBOL_PREFIX}{code}_{exchange.upper()}"


def canonical_from_custom(name: str) -> str:
    """Inverse of :func:`custom_symbol_name`."""
    if not name.startswith(SYMBOL_PREFIX):
        raise ValueError(f"{name!r} is not a QuantAgent custom symbol")
    body = name[len(SYMBOL_PREFIX):]
    code, _, exchange = body.rpartition("_")
    return f"{code}.{exchange}"


@dataclass(frozen=True)
class CustomSymbolSpec:
    """Properties to set on the MT5 side before loading any history.

    Contract size is 1 share and the volume step is the board's lot, because an
    A-share custom symbol configured with an FX default (100,000 units, 0.01
    lots) would make every position size wrong by orders of magnitude.
    """

    name: str
    canonical_symbol: str
    board: str
    currency: str = "CNY"
    digits: int = 2
    tick_size: float = 0.01
    contract_size: float = 1.0
    volume_min: float = 100.0
    volume_step: float = 100.0
    #: China equity sessions, Asia/Shanghai. Index 0 is Sunday per MQL5.
    quote_sessions: tuple[tuple[str, str], ...] = (("09:15", "11:30"), ("13:00", "15:00"))
    trade_sessions: tuple[tuple[str, str], ...] = (("09:30", "11:30"), ("13:00", "15:00"))
    trading_weekdays: tuple[int, ...] = (1, 2, 3, 4, 5)  # Monday..Friday
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Board -> (minimum volume, volume step) in shares, mirroring the exchange
#: lot rules rather than an FX convention.
BOARD_VOLUME_RULES: dict[str, tuple[float, float]] = {
    "SH_Main": (100.0, 100.0), "SZ_Main": (100.0, 100.0), "ChiNext": (100.0, 100.0),
    "STAR": (200.0, 1.0), "BSE": (100.0, 1.0),
}


def build_symbol_spec(canonical: str, *, board: str | None = None,
                      description: str = "") -> CustomSymbolSpec:
    board = board or mc.board_of(canonical)
    volume_min, volume_step = BOARD_VOLUME_RULES.get(board, (100.0, 100.0))
    star_sessions = (("09:30", "11:30"), ("13:00", "15:00"), ("15:05", "15:30"))
    return CustomSymbolSpec(
        name=custom_symbol_name(canonical),
        canonical_symbol=canonical,
        board=board,
        volume_min=volume_min,
        volume_step=volume_step,
        trade_sessions=star_sessions if board == "STAR" else CustomSymbolSpec.trade_sessions,
        description=description or f"QuantAgent replay of {canonical} ({board})",
    )


def events_to_mt5_ticks(events: pd.DataFrame) -> pd.DataFrame:
    """Map canonical trade events onto the MT5 tick layout.

    Only the columns MT5 actually stores are produced. Bid and ask are left at
    zero when the source is a trade-only feed rather than being back-filled from
    the last price -- a fabricated spread would make every spread-based
    indicator in the terminal read a number this repository invented.
    """
    if events.empty:
        return pd.DataFrame(
            columns=["time", "bid", "ask", "last", "volume", "time_msc",
                     "flags", "volume_real"]
        )

    times = pd.to_datetime(events["exchange_time"], errors="coerce")
    last = pd.to_numeric(events["price"], errors="coerce")
    volume = pd.to_numeric(events["volume_shares"], errors="coerce").fillna(0.0)

    flags = np.full(len(events), TICK_FLAG_LAST | TICK_FLAG_VOLUME, dtype=np.int64)
    if "side" in events.columns:
        side = events["side"].astype("object")
        flags = np.where(side == "BUY", flags | TICK_FLAG_BUY, flags)
        flags = np.where(side == "SELL", flags | TICK_FLAG_SELL, flags)

    has_quotes = {"bid_price", "ask_price"}.issubset(events.columns)
    bid = pd.to_numeric(events["bid_price"], errors="coerce").fillna(0.0) if has_quotes \
        else pd.Series(0.0, index=events.index)
    ask = pd.to_numeric(events["ask_price"], errors="coerce").fillna(0.0) if has_quotes \
        else pd.Series(0.0, index=events.index)
    if has_quotes:
        flags = flags | TICK_FLAG_BID | TICK_FLAG_ASK

    return pd.DataFrame({
        "time": (times.astype("int64") // 1_000_000_000).astype("int64"),
        "bid": bid.to_numpy(),
        "ask": ask.to_numpy(),
        "last": last.to_numpy(),
        # MT5's integer `volume` is a tick count proxy; `volume_real` carries
        # the actual share quantity, so shares go in the field that keeps them.
        "volume": volume.round().astype("int64").to_numpy(),
        "time_msc": (times.astype("int64") // 1_000_000).astype("int64"),
        "flags": flags,
        "volume_real": volume.to_numpy(),
    })


def bars_to_mt5_rates(bars: pd.DataFrame) -> pd.DataFrame:
    """Map U0 daily bars onto the MT5 rates layout."""
    if bars.empty:
        return pd.DataFrame(
            columns=["time", "open", "high", "low", "close", "tick_volume",
                     "spread", "real_volume"]
        )
    times = pd.to_datetime(bars["trade_date"], errors="coerce")
    volume = pd.to_numeric(bars["volume"], errors="coerce").fillna(0.0)
    return pd.DataFrame({
        "time": (times.astype("int64") // 1_000_000_000).astype("int64"),
        "open": pd.to_numeric(bars["open"], errors="coerce").to_numpy(),
        "high": pd.to_numeric(bars["high"], errors="coerce").to_numpy(),
        "low": pd.to_numeric(bars["low"], errors="coerce").to_numpy(),
        "close": pd.to_numeric(bars["close"], errors="coerce").to_numpy(),
        "tick_volume": volume.round().astype("int64").to_numpy(),
        # Spread is unknown for a bar-only source. Zero means "not supplied",
        # and the manifest says so rather than implying a tight market.
        "spread": np.zeros(len(bars), dtype=np.int64),
        "real_volume": volume.to_numpy(),
    })


@dataclass
class ImportManifest:
    """What was exported, from what, and with what integrity guarantees."""

    custom_symbol: str
    canonical_symbol: str
    board: str
    generated_at: str
    origin_data_class: str
    imported_data_class: str = mc.CUSTOM_SYMBOL_REPLAY
    tick_rows: int = 0
    bar_rows: int = 0
    source_content_hash: str = ""
    export_content_hash: str = ""
    date_range: tuple[str, str] | None = None
    files: dict[str, str] = field(default_factory=dict)
    spec: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_import_plan(
    canonical_symbol: str,
    *,
    events: pd.DataFrame | None = None,
    bars: pd.DataFrame | None = None,
    origin_data_class: str = mc.UNKNOWN_SEMANTICS,
    output_dir: str | Path,
    board: str | None = None,
) -> ImportManifest:
    """Write the MT5-side files and the manifest that proves what they contain.

    The manifest is the audit artifact: it hashes both the source frame and the
    exported frame, so a later terminal-side row count can be checked against a
    number that was recorded before the data left this process.
    """
    spec = build_symbol_spec(canonical_symbol, board=board)
    target = Path(output_dir) / spec.name
    target.mkdir(parents=True, exist_ok=True)

    manifest = ImportManifest(
        custom_symbol=spec.name,
        canonical_symbol=canonical_symbol,
        board=spec.board,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        origin_data_class=origin_data_class,
        spec=spec.to_dict(),
    )

    if origin_data_class in (mc.GENERATED_TESTER_TICK, mc.BAR_DERIVED_TICK):
        manifest.warnings.append(
            "origin data is manufactured; a Strategy Tester result built on it "
            "measures the generator, not the market"
        )
    if origin_data_class == mc.SNAPSHOT_DERIVED_TRADE_AGGREGATE:
        manifest.warnings.append(
            "origin data is 3-second snapshot aggregates: the terminal will "
            "display them as ticks, but intra-bucket sequencing and per-trade "
            "size are not present in the underlying data"
        )
    if origin_data_class == mc.UNKNOWN_SEMANTICS:
        manifest.warnings.append(
            "origin data class was not declared; the import cannot be audited"
        )

    if events is not None and len(events):
        ticks = events_to_mt5_ticks(events)
        tick_path = target / "ticks.parquet"
        ticks.to_parquet(tick_path, index=False)
        manifest.tick_rows = len(ticks)
        manifest.source_content_hash = frame_hash(events)
        manifest.export_content_hash = frame_hash(ticks)
        manifest.files["ticks"] = str(tick_path)
        times = pd.to_datetime(events["exchange_time"], errors="coerce").dropna()
        if len(times):
            manifest.date_range = (str(times.min()), str(times.max()))
        if not {"bid_price", "ask_price"}.issubset(events.columns):
            manifest.warnings.append(
                "source carries no quotes; bid/ask exported as 0 rather than "
                "being synthesised from last price, so spread-based indicators "
                "in the terminal will be meaningless"
            )

    if bars is not None and len(bars):
        rates = bars_to_mt5_rates(bars)
        rates_path = target / "rates.parquet"
        rates.to_parquet(rates_path, index=False)
        manifest.bar_rows = len(rates)
        manifest.files["rates"] = str(rates_path)

    manifest_path = target / "import_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    manifest.files["manifest"] = str(manifest_path)
    return manifest


@dataclass
class ImportVerification:
    custom_symbol: str
    expected_ticks: int
    observed_ticks: int
    expected_bars: int
    observed_bars: int
    verdict: str
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_import(
    manifest: ImportManifest, *, observed_ticks: int, observed_bars: int
) -> ImportVerification:
    """Compare terminal-side counts against the manifest recorded at export.

    A row-count match is necessary, not sufficient: it proves nothing was
    dropped, and says nothing about whether prices survived the round trip. The
    verdict wording is chosen to keep that distinction visible.
    """
    problems: list[str] = []
    if observed_ticks != manifest.tick_rows:
        problems.append(
            f"tick count mismatch: exported {manifest.tick_rows}, "
            f"terminal holds {observed_ticks}"
        )
    if observed_bars != manifest.bar_rows:
        problems.append(
            f"bar count mismatch: exported {manifest.bar_rows}, "
            f"terminal holds {observed_bars}"
        )
    return ImportVerification(
        custom_symbol=manifest.custom_symbol,
        expected_ticks=manifest.tick_rows, observed_ticks=observed_ticks,
        expected_bars=manifest.bar_rows, observed_bars=observed_bars,
        verdict="COUNTS_MATCH" if not problems else "COUNT_MISMATCH",
        problems=problems,
    )


def classify_tester_ticks(modelling_mode: str) -> dict[str, Any]:
    """Say whether a Strategy Tester run used real or generated ticks."""
    real = modelling_mode in REAL_TICK_MODES
    return {
        "modelling_mode": modelling_mode,
        "data_class": mc.CUSTOM_SYMBOL_REPLAY if real else mc.GENERATED_TESTER_TICK,
        "uses_real_ticks": real,
        "reportable_as_real_ticks": real,
        "note": (
            "the tester consumed the ticks supplied to the custom symbol"
            if real else
            "the tester synthesised ticks from bars; results describe the tick "
            "generator's behaviour and must not be reported as tick-level results"
        ),
    }
