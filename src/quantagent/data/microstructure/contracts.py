"""Provider-neutral canonical contracts for A-share microstructure events.

The daily U0 panel already has a provenance spine (see
:mod:`quantagent.data.ashare.contracts`). Tick and order-book data needs a
stricter one, because the failure mode is different: a daily bar that is wrong
is usually *visibly* wrong, while a book snapshot relabelled as an exchange
order book silently invalidates every microstructure conclusion built on it.

Two rules drive this module.

**Semantics are declared, never inferred.** Every event frame carries a
``data_class`` drawn from :data:`DATA_CLASSES`. A five-level aggregated quote
is ``LEVEL2_SNAPSHOT``, never ``LEVEL2_ORDER_BOOK``. A tick synthesised from a
bar is ``BAR_DERIVED_TICK``. A Strategy Tester tick is ``GENERATED_TESTER_TICK``
and may never enter the authoritative store at all. When a provider will not
say what it is returning, the answer is ``UNKNOWN_SEMANTICS`` and the Data
Quality agent vetoes the dataset.

**Nothing is manufactured.** If the source does not publish an exchange
sequence number, ``sequence`` stays null and only ``ingest_sequence`` -- a
storage-ordering counter owned by this repository, never by the exchange -- is
populated. Same for ``trade_id``, ``order_id`` and ``side``: an inferred trade
direction must be written to ``side`` only together with a ``side_method`` that
names the inference rule, so downstream code can tell a published side from a
tick-rule guess.

单位约定 / unit conventions:

``price``            CNY per share
``volume_shares``    股 (shares). Vendors reporting 手 are converted at the
                     adapter boundary, never downstream.
``amount_cny``       CNY
``*_time_ns``        Unix epoch nanoseconds, UTC. ``exchange_time`` keeps the
                     vendor's own wall-clock string for forensics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# ---------------------------------------------------------------------------
# Data-class taxonomy
# ---------------------------------------------------------------------------
#: Individual order insert/cancel messages carrying exchange order identifiers.
EXCHANGE_ORDER_EVENT = "EXCHANGE_ORDER_EVENT"
#: Individual matched trades carrying exchange trade identifiers.
EXCHANGE_TRADE_EVENT = "EXCHANGE_TRADE_EVENT"
#: A book reconstructed from, or delivered as, per-order queue state.
LEVEL2_ORDER_BOOK = "LEVEL2_ORDER_BOOK"
#: Periodic aggregated depth (the common "10档/千档" vendor product).
LEVEL2_SNAPSHOT = "LEVEL2_SNAPSHOT"
#: Best bid/offer plus the vendor's aggregated 5-level display.
LEVEL1_QUOTE = "LEVEL1_QUOTE"
#: Last-trade prints without a matching order stream.
TRADE_TICK = "TRADE_TICK"
#: Trade activity aggregated between two vendor snapshots, published as if it
#: were a tick stream. **Measured, not assumed**: the public Chinese "分笔"
#: products (Tencent, and the akshare wrappers over it) emit one record per
#: 3-second exchange snapshot interval, and ``amount != price * volume`` on
#: 48%-82% of records because each record buckets several trades at different
#: prices. Calling that ``TRADE_TICK`` would overstate it by exactly the margin
#: this taxonomy exists to prevent, so it gets its own class.
#:
#: This class extends the classes named in the mission brief. It was added
#: because the brief's list had no honest slot for a 3-second aggregate, and
#: the alternative was to mislabel real data.
SNAPSHOT_DERIVED_TRADE_AGGREGATE = "SNAPSHOT_DERIVED_TRADE_AGGREGATE"
#: Broker-manufactured quotes for an OTC/CFD instrument.
BROKER_SYNTHETIC_QUOTE = "BROKER_SYNTHETIC_QUOTE"
#: MT5 Strategy Tester tick generation. Never authoritative.
GENERATED_TESTER_TICK = "GENERATED_TESTER_TICK"
#: Ticks interpolated from OHLC bars. Never authoritative.
BAR_DERIVED_TICK = "BAR_DERIVED_TICK"
#: Events replayed out of a custom symbol rather than read from a feed.
CUSTOM_SYMBOL_REPLAY = "CUSTOM_SYMBOL_REPLAY"
#: The provider would not, or could not, state what the payload means.
UNKNOWN_SEMANTICS = "UNKNOWN_SEMANTICS"

DATA_CLASSES: tuple[str, ...] = (
    EXCHANGE_ORDER_EVENT,
    EXCHANGE_TRADE_EVENT,
    LEVEL2_ORDER_BOOK,
    LEVEL2_SNAPSHOT,
    LEVEL1_QUOTE,
    TRADE_TICK,
    SNAPSHOT_DERIVED_TRADE_AGGREGATE,
    BROKER_SYNTHETIC_QUOTE,
    GENERATED_TESTER_TICK,
    BAR_DERIVED_TICK,
    CUSTOM_SYMBOL_REPLAY,
    UNKNOWN_SEMANTICS,
)

#: Classes that may never be written into the authoritative raw event store.
#: They are legitimate research inputs but they are *manufactured*, so mixing
#: them into the immutable journal destroys its meaning as a record of what the
#: market actually did.
NON_AUTHORITATIVE_CLASSES: frozenset[str] = frozenset(
    {GENERATED_TESTER_TICK, BAR_DERIVED_TICK, CUSTOM_SYMBOL_REPLAY, UNKNOWN_SEMANTICS}
)

#: Classes that genuinely evidence per-order exchange messages. Only these
#: license a Level-A (queue-position) simulation.
ORDER_EVENT_CLASSES: frozenset[str] = frozenset(
    {EXCHANGE_ORDER_EVENT, LEVEL2_ORDER_BOOK}
)


# ---------------------------------------------------------------------------
# Trade-direction provenance
# ---------------------------------------------------------------------------
#: The exchange published the aggressor side directly.
SIDE_EXCHANGE_PUBLISHED = "EXCHANGE_PUBLISHED"
#: Derived by matching the trade against the resting order ids.
SIDE_ORDER_MATCHED = "ORDER_ID_MATCHED"
#: Lee-Ready / tick-rule style inference. A guess, and labelled as one.
SIDE_TICK_RULE = "TICK_RULE_INFERRED"
#: Quote-midpoint comparison inference.
SIDE_QUOTE_RULE = "QUOTE_RULE_INFERRED"
#: Direction genuinely unknown; ``side`` must be null.
SIDE_UNKNOWN = "UNKNOWN"

SIDE_METHODS: tuple[str, ...] = (
    SIDE_EXCHANGE_PUBLISHED,
    SIDE_ORDER_MATCHED,
    SIDE_TICK_RULE,
    SIDE_QUOTE_RULE,
    SIDE_UNKNOWN,
)

#: Methods that produce an *observed* rather than *inferred* side.
OBSERVED_SIDE_METHODS: frozenset[str] = frozenset(
    {SIDE_EXCHANGE_PUBLISHED, SIDE_ORDER_MATCHED}
)


# ---------------------------------------------------------------------------
# Book semantics
# ---------------------------------------------------------------------------
#: Depth aggregated per price level (what almost every vendor ships).
BOOK_PRICE_AGGREGATED = "PRICE_LEVEL_AGGREGATED"
#: Per-order queue visible at each level.
BOOK_ORDER_QUEUE = "ORDER_QUEUE_RESOLVED"
#: Broker's own display book for a synthetic instrument.
BOOK_BROKER_DISPLAY = "BROKER_DISPLAY"
BOOK_SEMANTICS: tuple[str, ...] = (
    BOOK_PRICE_AGGREGATED,
    BOOK_ORDER_QUEUE,
    BOOK_BROKER_DISPLAY,
)


# ---------------------------------------------------------------------------
# Event families
# ---------------------------------------------------------------------------
FAMILY_TRADE = "trade_event"
FAMILY_ORDER = "order_event"
FAMILY_BOOK = "book_snapshot"
FAMILY_QUOTE = "quote_event"

#: Columns every canonical event frame carries, whatever the family.
#: ``sequence`` is the *exchange's* number and is null when the vendor does not
#: publish one; ``ingest_sequence`` is ours and is always populated.
COMMON_COLUMNS: tuple[str, ...] = (
    "symbol",
    "exchange",
    "trade_date",
    "exchange_time",
    "event_time_ns",
    "receive_time_ns",
    "ingest_sequence",
    "sequence",
    "source_provider",
    "source_channel",
    "data_class",
    "raw_partition",
    "available_at",
)


@dataclass(frozen=True)
class EventContract:
    """Declared schema and semantics for one canonical event family."""

    family: str
    value_columns: tuple[str, ...]
    #: Columns that must never be silently synthesised by the pipeline.
    never_manufactured: tuple[str, ...] = ()
    units: Mapping[str, str] = field(default_factory=dict)
    description: str = ""

    @property
    def columns(self) -> tuple[str, ...]:
        return COMMON_COLUMNS + self.value_columns

    def missing_columns(self, columns: object) -> list[str]:
        present = set(columns)  # type: ignore[arg-type]
        return [c for c in self.columns if c not in present]


TRADE_EVENT = EventContract(
    family=FAMILY_TRADE,
    value_columns=(
        "trade_id",
        "price",
        "volume_shares",
        "amount_cny",
        "side",
        "side_method",
        "buy_order_id",
        "sell_order_id",
        "trade_kind",
    ),
    never_manufactured=("trade_id", "sequence", "side", "buy_order_id", "sell_order_id"),
    units={"price": "CNY", "volume_shares": "shares", "amount_cny": "CNY"},
    description="逐笔成交 / individual matched trades",
)

ORDER_EVENT = EventContract(
    family=FAMILY_ORDER,
    value_columns=(
        "order_id",
        "order_type",
        "side",
        "price",
        "volume_shares",
        "event_action",
    ),
    never_manufactured=("order_id", "sequence", "side"),
    units={"price": "CNY", "volume_shares": "shares"},
    description="逐笔委托 / individual order insert & cancel messages",
)

BOOK_SNAPSHOT = EventContract(
    family=FAMILY_BOOK,
    value_columns=(
        "snapshot_sequence",
        "level",
        "bid_price",
        "bid_volume_shares",
        "ask_price",
        "ask_volume_shares",
        "book_semantics",
        "depth_levels",
    ),
    never_manufactured=("snapshot_sequence", "sequence"),
    units={
        "bid_price": "CNY",
        "ask_price": "CNY",
        "bid_volume_shares": "shares",
        "ask_volume_shares": "shares",
    },
    description="盘口快照 / periodic aggregated depth snapshots",
)

QUOTE_EVENT = EventContract(
    family=FAMILY_QUOTE,
    value_columns=(
        "bid_price",
        "bid_volume_shares",
        "ask_price",
        "ask_volume_shares",
        "last_price",
        "last_volume_shares",
        "cum_volume_shares",
        "cum_amount_cny",
    ),
    never_manufactured=("sequence",),
    units={
        "bid_price": "CNY",
        "ask_price": "CNY",
        "last_price": "CNY",
        "bid_volume_shares": "shares",
        "ask_volume_shares": "shares",
        "last_volume_shares": "shares",
        "cum_volume_shares": "shares",
        "cum_amount_cny": "CNY",
    },
    description="Level-1 行情快照 / best bid-offer plus session cumulatives",
)

EVENT_CONTRACTS: Mapping[str, EventContract] = {
    FAMILY_TRADE: TRADE_EVENT,
    FAMILY_ORDER: ORDER_EVENT,
    FAMILY_BOOK: BOOK_SNAPSHOT,
    FAMILY_QUOTE: QUOTE_EVENT,
}


def contract_for(family: str) -> EventContract:
    try:
        return EVENT_CONTRACTS[family]
    except KeyError:  # pragma: no cover - guarded by callers
        raise KeyError(
            f"unknown event family {family!r}; known families: "
            f"{sorted(EVENT_CONTRACTS)}"
        ) from None


# ---------------------------------------------------------------------------
# A-share trading sessions
# ---------------------------------------------------------------------------
#: Session phases, in the order they occur on a normal SSE/SZSE/BSE trading day.
#: Times are Asia/Shanghai. Sources are recorded in
#: ``docs/research/A股高频回测模型说明.md``; the simulator imports these rather
#: than hard-coding minutes at the call site.
PHASE_OPENING_AUCTION = "OPENING_CALL_AUCTION"
PHASE_PRE_OPEN_QUIET = "PRE_OPEN_NO_CANCEL"
PHASE_CONTINUOUS_AM = "CONTINUOUS_MORNING"
PHASE_LUNCH_BREAK = "LUNCH_BREAK"
PHASE_CONTINUOUS_PM = "CONTINUOUS_AFTERNOON"
PHASE_CLOSING_AUCTION = "CLOSING_CALL_AUCTION"
PHASE_AFTER_HOURS = "AFTER_HOURS_FIXED_PRICE"
#: Prints that arrive after the closing auction settles but before the vendor
#: stops publishing. Measured 2026-07-24 across all five boards: 235 records
#: between 15:05 and 15:23 carrying 0.047% of the day's volume, none of which
#: appear in the exchange daily bar (the reconciliation matches exactly without
#: them). Their nature -- block-trade reporting, odd-lot settlement, or vendor
#: republication -- is **not established**, so they get a phase of their own
#: rather than being folded into a session or dismissed as noise.
PHASE_POST_CLOSE = "POST_CLOSE_UNCLASSIFIED"
PHASE_CLOSED = "CLOSED"

SESSION_PHASES: tuple[str, ...] = (
    PHASE_OPENING_AUCTION,
    PHASE_PRE_OPEN_QUIET,
    PHASE_CONTINUOUS_AM,
    PHASE_LUNCH_BREAK,
    PHASE_CONTINUOUS_PM,
    PHASE_CLOSING_AUCTION,
    PHASE_AFTER_HOURS,
    PHASE_POST_CLOSE,
    PHASE_CLOSED,
)

#: ``(start, end)`` half-open windows as "HH:MM" in Asia/Shanghai.
#: 09:15-09:20 accepts and cancels orders; 09:20-09:25 accepts but does **not**
#: allow cancellation; 09:25-09:30 is order-accepting quiet time.
SESSION_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("09:15", "09:20", PHASE_OPENING_AUCTION),
    ("09:20", "09:25", PHASE_PRE_OPEN_QUIET),
    ("09:25", "09:30", PHASE_PRE_OPEN_QUIET),
    # The morning session closes at 11:30:00 and the closing snapshot carries
    # that exact stamp. Measured 2026-07-24: five of five cohort symbols print
    # their morning-close aggregate at 11:30:00-11:30:01. A half-open window
    # ending at 11:30 pushed those into the lunch break, so the window runs to
    # 11:31 and the lunch break starts after the close print.
    ("09:30", "11:31", PHASE_CONTINUOUS_AM),
    ("11:31", "13:00", PHASE_LUNCH_BREAK),
    ("13:00", "14:57", PHASE_CONTINUOUS_PM),
    # The closing call auction runs 14:57-15:00, but its *result* is
    # disseminated on the snapshot that follows the close. Measured on
    # 600000.SH 2026-07-24: the auction print (486,400 shares at 9.04) carries
    # an exchange timestamp of 15:00:03, and excluding it left the reconciled
    # day short by exactly that volume and turnover. The window therefore runs
    # to 15:01 so the auction result lands in the auction phase, while genuine
    # post-close block prints (observed at 15:08 and 15:11 the same day, and
    # correctly absent from the U0 daily bar) stay CLOSED.
    ("14:57", "15:01", PHASE_CLOSING_AUCTION),
)

#: STAR market (科创板) after-hours fixed-price trading, 15:05-15:30.
STAR_AFTER_HOURS = ("15:05", "15:30")

#: Phases in which a marketable order can actually trade against the book.
CONTINUOUS_PHASES: frozenset[str] = frozenset(
    {PHASE_CONTINUOUS_AM, PHASE_CONTINUOUS_PM}
)
AUCTION_PHASES: frozenset[str] = frozenset(
    {PHASE_OPENING_AUCTION, PHASE_PRE_OPEN_QUIET, PHASE_CLOSING_AUCTION}
)


def _to_minutes(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


#: Window in which vendors keep publishing prints after the close. Records here
#: are quarantined into :data:`PHASE_POST_CLOSE`, not silently accepted.
POST_CLOSE_WINDOW = ("15:01", "15:35")


def board_of(symbol: str) -> str:
    """Board for a canonical ``<code>.<EX>`` symbol, for session selection.

    Delegates to :func:`quantagent.data.ashare.symbols.board_of`, which is the
    single authoritative code-prefix classifier in this repository. Keeping a
    second copy of those prefix rules here would guarantee the two drift --
    B-share ranges, the 002/003 SME merger and the BSE legacy codes are exactly
    the kind of detail one copy would quietly miss.

    Used only to pick the right *session calendar*. Anything needing
    authoritative board membership must read the U0 security master.
    """
    from quantagent.data.ashare.symbols import SymbolError
    from quantagent.data.ashare.symbols import board_of as _board_of

    try:
        return _board_of(symbol)
    except (SymbolError, ValueError):
        # A malformed symbol must not crash a session classification; the
        # caller gets the default (non-STAR) calendar.
        return "UNKNOWN"


def session_phase(hhmm: str, *, board: str | None = None) -> str:
    """Classify an ``HH:MM`` Asia/Shanghai wall-clock into a session phase.

    ``board`` selects the STAR after-hours fixed-price window (15:05-15:30),
    which no other board has. Without it, a STAR after-hours print would be
    indistinguishable from an unexplained post-close record.
    """
    minute = _to_minutes(hhmm)
    for start, end, phase in SESSION_WINDOWS:
        if _to_minutes(start) <= minute < _to_minutes(end):
            return phase
    if board == "STAR":
        start, end = STAR_AFTER_HOURS
        if _to_minutes(start) <= minute < _to_minutes(end):
            return PHASE_AFTER_HOURS
    start, end = POST_CLOSE_WINDOW
    if _to_minutes(start) <= minute < _to_minutes(end):
        return PHASE_POST_CLOSE
    return PHASE_CLOSED
