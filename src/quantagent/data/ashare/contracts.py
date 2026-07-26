"""Canonical dataset contracts for the A-share data foundation.

Every dataset family the pipeline persists carries the same provenance spine so
that any row can be traced back to the exact provider call that produced it and
audited for point-in-time validity:

``source``            provider family (tickflow / tencent / eastmoney / sina / exchange)
``source_endpoint``   the concrete endpoint or vendor method that answered
``retrieved_at``      UTC timestamp of the retrieval
``available_at``      the earliest wall-clock time a decision maker could have used the row
``quality_status``    OK / SUSPECT / DERIVED — never silently upgraded

Unit and scale semantics are declared explicitly per family rather than being
implied by a column name, because A-share vendors disagree: some report volume
in lots (手, 100 shares), some in shares; amount is CNY for every source we use
but scale differs between raw feeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# --- provenance spine -------------------------------------------------------
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "source",
    "source_endpoint",
    "retrieved_at",
    "available_at",
    "quality_status",
)

QUALITY_OK = "OK"
QUALITY_SUSPECT = "SUSPECT"
QUALITY_DERIVED = "DERIVED"

# --- unit vocabulary --------------------------------------------------------
VOLUME_SHARES = "shares"
VOLUME_LOTS = "lots"          # 手 == 100 shares for A-shares
AMOUNT_CNY = "CNY"
PRICE_CNY = "CNY"

ADJUST_NONE = "none"          # raw traded price
ADJUST_QFQ = "qfq"            # forward adjusted (前复权)
ADJUST_HFQ = "hfq"            # backward adjusted (后复权)

TIMEZONE_CST = "Asia/Shanghai"


@dataclass(frozen=True)
class DatasetContract:
    """Declared schema + semantics for one dataset family."""

    name: str
    key_columns: tuple[str, ...]
    value_columns: tuple[str, ...]
    price_unit: str | None = None
    volume_unit: str | None = None
    amount_unit: str | None = None
    adjustment: str | None = None
    timezone: str = TIMEZONE_CST
    point_in_time: bool = True
    notes: str = ""

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.key_columns) + tuple(self.value_columns) + PROVENANCE_COLUMNS


DAILY_BARS = DatasetContract(
    name="daily_bars",
    key_columns=("symbol", "trade_date"),
    value_columns=("open", "high", "low", "close", "volume", "amount"),
    price_unit=PRICE_CNY,
    volume_unit=VOLUME_SHARES,
    amount_unit=AMOUNT_CNY,
    adjustment=ADJUST_NONE,
    notes="raw traded prices; volume normalised to shares at ingestion (vendor lots x100)",
)

MINUTE_BARS = DatasetContract(
    name="minute_bars",
    key_columns=("symbol", "bar_time"),
    value_columns=("open", "high", "low", "close", "volume", "amount", "frequency"),
    price_unit=PRICE_CNY,
    volume_unit=VOLUME_SHARES,
    amount_unit=AMOUNT_CNY,
    adjustment=ADJUST_NONE,
    notes="bar_time is the bar CLOSE stamp in Asia/Shanghai; frequency in minutes",
)

ADJUST_FACTORS = DatasetContract(
    name="adjust_factors",
    key_columns=("symbol", "effective_date"),
    value_columns=("hfq_factor", "adjustment_method"),
    adjustment=ADJUST_HFQ,
    notes="cumulative backward-adjustment factor effective from effective_date (ex-rights date)",
)

CORPORATE_ACTIONS = DatasetContract(
    name="corporate_actions",
    key_columns=("symbol", "ex_date", "action_type"),
    value_columns=("cash_dividend_per_share", "stock_dividend_ratio", "rights_ratio",
                   "announce_date", "record_date"),
    amount_unit=AMOUNT_CNY,
    notes="per-share cash dividend is pre-tax CNY; ratios are shares per share",
)

SUSPENSION_INTERVALS = DatasetContract(
    name="suspension_intervals",
    key_columns=("symbol", "effective_start"),
    value_columns=("effective_end", "suspension_reason", "evidence"),
    notes="a trading day inside an interval is a halted session, not missing data",
)

ST_INTERVALS = DatasetContract(
    name="st_intervals",
    key_columns=("symbol", "effective_start"),
    value_columns=("effective_end", "security_name", "st_flag", "st_kind"),
    notes="derived from the exchange security-name history; ST/*ST names imply the tighter limit",
)

SECURITY_MASTER = DatasetContract(
    name="security_master",
    key_columns=("symbol",),
    value_columns=("code", "exchange", "board", "security_type", "name",
                   "listing_date", "delisting_date", "status"),
    point_in_time=False,
    notes="current-snapshot identity; historical name/ST/suspension live in interval tables",
)

QUOTES = DatasetContract(
    name="quotes",
    key_columns=("symbol", "quote_time"),
    value_columns=("last_price", "prev_close", "open", "high", "low", "volume", "amount",
                   "bid_prices", "bid_volumes", "ask_prices", "ask_volumes", "depth_levels"),
    price_unit=PRICE_CNY,
    volume_unit=VOLUME_SHARES,
    amount_unit=AMOUNT_CNY,
    point_in_time=False,
    notes="Level-1 snapshot with 5-level aggregated depth; NOT order-by-order Level-2",
)

MONEY_FLOW = DatasetContract(
    name="money_flow",
    key_columns=("symbol", "trade_date"),
    value_columns=("main_net", "extra_large_net", "large_net", "medium_net", "small_net"),
    amount_unit=AMOUNT_CNY,
    notes="order-size bucketed net inflow in CNY; vendor bucket thresholds are vendor-defined",
)

TRADING_CALENDAR = DatasetContract(
    name="trading_calendar",
    key_columns=("trade_date",),
    value_columns=("exchange", "is_open"),
    notes="exchange session calendar used for missing-session detection",
)

CONTRACTS: Mapping[str, DatasetContract] = {
    c.name: c
    for c in (
        DAILY_BARS, MINUTE_BARS, ADJUST_FACTORS, CORPORATE_ACTIONS,
        SUSPENSION_INTERVALS, ST_INTERVALS, SECURITY_MASTER, QUOTES,
        MONEY_FLOW, TRADING_CALENDAR,
    )
}


@dataclass(frozen=True)
class SourceBoundary:
    """Records that a symbol's history changes provider at a date.

    Vendors are never blended into one continuous series without this record;
    the boundary makes the seam explicit and auditable downstream.
    """

    symbol: str
    boundary_date: str
    provider_before: str
    provider_after: str
    reason: str
    metrics: dict = field(default_factory=dict)
