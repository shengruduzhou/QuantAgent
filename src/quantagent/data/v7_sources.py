"""V7 data-source registry.

Single source of truth for provider metadata (Qlib, AkShare, TuShare,
local snapshots). Downstream bootstrap code, CLI commands, and manifest
emitters look here so the contract between V7 silver/gold artefacts and
their upstream providers stays explicit.

The registry is intentionally light: it does not hold credentials or
endpoints that vary by deployment. It documents:

* logical provider name (``qlib``, ``akshare``, ``tushare``, ``local``)
* the kind of data it produces (``market``, ``valuation``,
  ``fundamentals``, ``universe``, ``sector``, ``tradability``,
  ``disclosure``)
* the requested behaviour when network is disabled (``offline`` vs
  ``fail_loud`` vs ``require_local_snapshot``)
* the canonical V7 PIT columns each provider must populate

If a provider's actual response drifts from the canonical schema the
bootstrap layer will fail loudly; that policy lives in this file's
``REQUIRED_COLUMNS`` map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


V7DataKind = Literal[
    "market",
    "valuation",
    "fundamentals",
    "universe",
    "sector",
    "tradability",
    "disclosure",
    "macro",
    "flow",
    "index",
]

V7OfflineBehaviour = Literal["offline", "fail_loud", "require_local_snapshot"]


@dataclass(frozen=True)
class V7DataSource:
    """Static metadata describing one V7 provider × data-kind pair."""

    name: str
    kind: V7DataKind
    provider: str
    description: str
    offline_behaviour: V7OfflineBehaviour
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()
    requires_network: bool = False
    pit_policy: str = "available_at <= trade_date"
    notes: str = ""


REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "market_panel": (
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "available_at",
    ),
    "valuation": (
        "symbol",
        "trade_date",
        "available_at",
        "pe_ttm",
        "pb",
        "market_cap",
    ),
    "fundamentals": (
        "symbol",
        "report_period",
        "ann_date",
        "available_at",
    ),
    "universe": (
        "symbol",
        "name",
        "exchange",
        "available_at",
    ),
    "sector": (
        "symbol",
        "sector_level_1",
        "sector_level_2",
        "source",
        "source_version",
        "effective_date",
        "fetched_at",
        "available_at",
        "coverage_status",
    ),
    "tradability": (
        "symbol",
        "trade_date",
        "available_at",
        "is_suspended",
        "is_st",
        "is_limit_up",
        "is_limit_down",
    ),
    "disclosure": (
        "symbol",
        "ann_date",
        "available_at",
    ),
    "macro": (
        "observation_date",
        "available_at",
        "source",
    ),
    "flow": (
        "observation_date",
        "available_at",
        "source",
    ),
    "index": (
        "observation_date",
        "symbol",
        "close",
        "available_at",
        "source",
    ),
}


V7_DATA_SOURCES: tuple[V7DataSource, ...] = (
    V7DataSource(
        name="qlib_cn_daily",
        kind="market",
        provider="qlib",
        description="Local Qlib CN data provider; daily OHLCV + PIT next-day available_at.",
        offline_behaviour="offline",
        required_columns=REQUIRED_COLUMNS["market_panel"],
        optional_columns=("is_suspended", "is_st", "is_limit_up", "is_limit_down"),
        pit_policy="close-derived features available from the next trading row",
        notes="run scripts/get_data.py qlib_data --region cn before first use",
    ),
    V7DataSource(
        name="akshare_valuation_snapshot",
        kind="valuation",
        provider="akshare",
        description="Daily PE/PB/PS/market-cap spot quote snapshot.",
        offline_behaviour="require_local_snapshot",
        required_columns=REQUIRED_COLUMNS["valuation"],
        optional_columns=("ps_ttm", "free_float_market_cap", "dividend_yield", "turnover_rate"),
        requires_network=True,
    ),
    V7DataSource(
        name="akshare_financial_statements",
        kind="fundamentals",
        provider="akshare",
        description="Sina/EastMoney financial statements (income / balance / cashflow).",
        offline_behaviour="fail_loud",
        required_columns=REQUIRED_COLUMNS["fundamentals"],
        requires_network=True,
    ),
    V7DataSource(
        name="tushare_financial_statements",
        kind="fundamentals",
        provider="tushare",
        description="TuShare PRO financial statements API; PIT ann_date + trading-calendar available_at.",
        offline_behaviour="fail_loud",
        required_columns=REQUIRED_COLUMNS["fundamentals"],
        requires_network=True,
        notes="requires TUSHARE_TOKEN env",
    ),
    V7DataSource(
        name="akshare_universe",
        kind="universe",
        provider="akshare",
        description="A-share universe (symbol, name, exchange, list_date).",
        offline_behaviour="require_local_snapshot",
        required_columns=REQUIRED_COLUMNS["universe"],
        requires_network=True,
    ),
    V7DataSource(
        name="local_sector_mapping",
        kind="sector",
        provider="local",
        description="Local symbol-level industry/sector mapping CSV (preferred over AkShare board scraping).",
        offline_behaviour="offline",
        required_columns=REQUIRED_COLUMNS["sector"],
    ),
    V7DataSource(
        name="akshare_sector",
        kind="sector",
        provider="akshare",
        description="AkShare industry board endpoints; symbol-level resolution requires per-board membership API.",
        offline_behaviour="fail_loud",
        required_columns=REQUIRED_COLUMNS["sector"],
        requires_network=True,
        notes="AkShare's industry-summary endpoints are board-level only and must NOT be cross-joined onto symbols.",
    ),
    V7DataSource(
        name="local_tradability",
        kind="tradability",
        provider="local",
        description="Symbol-level tradability flags (suspension, ST, limit-up/down) derived from market panel.",
        offline_behaviour="offline",
        required_columns=REQUIRED_COLUMNS["tradability"],
    ),
    V7DataSource(
        name="local_disclosure",
        kind="disclosure",
        provider="local",
        description="Local pre-collected announcement / disclosure snapshots.",
        offline_behaviour="offline",
        required_columns=REQUIRED_COLUMNS["disclosure"],
    ),
    V7DataSource(
        name="akshare_macro",
        kind="macro",
        provider="akshare",
        description="China macro: yield curve, Shibor, repo, central-bank OMO, social financing, M0/M1/M2, CPI, PPI.",
        offline_behaviour="require_local_snapshot",
        required_columns=REQUIRED_COLUMNS["macro"],
        optional_columns=("maturity", "tenor", "yield_pct", "rate_pct",
                          "inject_amount_cny", "expire_amount_cny", "net_amount_cny",
                          "aggregate_financing_cny", "m0_cny", "m1_cny", "m2_cny",
                          "cpi_yoy_pct", "ppi_yoy_pct"),
        requires_network=True,
        notes="Tracks the national-team money-flow thesis (bond curve + central-bank stance + money supply).",
    ),
    V7DataSource(
        name="akshare_flow",
        kind="flow",
        provider="akshare",
        description="China capital flow: northbound, margin balance, ETF fund flow, sector fund flow.",
        offline_behaviour="require_local_snapshot",
        required_columns=REQUIRED_COLUMNS["flow"],
        optional_columns=("channel", "market", "symbol", "sector",
                          "net_inflow_cny", "margin_balance_cny", "short_balance_cny",
                          "net_flow", "main_net_inflow_cny"),
        requires_network=True,
    ),
    V7DataSource(
        name="akshare_index",
        kind="index",
        provider="akshare",
        description="Major Chinese equity indices, commodity main-continuous, treasury futures.",
        offline_behaviour="require_local_snapshot",
        required_columns=REQUIRED_COLUMNS["index"],
        optional_columns=("label", "kind", "open", "high", "low", "volume", "amount"),
        requires_network=True,
    ),
)


_SOURCES_BY_NAME = {source.name: source for source in V7_DATA_SOURCES}


@dataclass(frozen=True)
class V7SchemaReport:
    """Result of validating a frame against a registered V7 data source."""

    source: str
    kind: V7DataKind
    row_count: int
    missing_columns: tuple[str, ...]
    extra_columns: tuple[str, ...] = field(default_factory=tuple)
    pit_violation_count: int = 0
    status: str = "passed"

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "kind": self.kind,
            "row_count": self.row_count,
            "missing_columns": list(self.missing_columns),
            "extra_columns": list(self.extra_columns),
            "pit_violation_count": self.pit_violation_count,
            "status": self.status,
        }


def get_v7_data_source(name: str) -> V7DataSource:
    """Return the registry entry for ``name`` or raise ``KeyError``."""

    if name not in _SOURCES_BY_NAME:
        raise KeyError(f"unknown V7 data source: {name}; known: {sorted(_SOURCES_BY_NAME)}")
    return _SOURCES_BY_NAME[name]


def list_v7_data_sources(kind: V7DataKind | None = None) -> tuple[V7DataSource, ...]:
    """Return the registered sources, optionally filtered by kind."""

    if kind is None:
        return V7_DATA_SOURCES
    return tuple(source for source in V7_DATA_SOURCES if source.kind == kind)


def validate_frame_against_source(frame, source_name: str) -> V7SchemaReport:
    """Validate a frame against the canonical schema for ``source_name``."""

    source = get_v7_data_source(source_name)
    columns = set(getattr(frame, "columns", ()))
    missing = tuple(c for c in source.required_columns if c not in columns)
    extra = tuple(sorted(columns - set(source.required_columns) - set(source.optional_columns)))
    pit_violations = 0
    if frame is not None and not getattr(frame, "empty", True) and "available_at" in columns:
        try:
            import pandas as pd

            parsed = pd.to_datetime(frame["available_at"], errors="coerce")
            pit_violations = int(parsed.isna().sum())
        except Exception:  # pragma: no cover - defensive
            pit_violations = 0
    status = "passed" if not missing and pit_violations == 0 else "failed"
    return V7SchemaReport(
        source=source_name,
        kind=source.kind,
        row_count=int(0 if frame is None else len(frame)),
        missing_columns=missing,
        extra_columns=extra,
        pit_violation_count=pit_violations,
        status=status,
    )


__all__ = [
    "V7DataSource",
    "V7DataKind",
    "V7OfflineBehaviour",
    "V7SchemaReport",
    "V7_DATA_SOURCES",
    "REQUIRED_COLUMNS",
    "get_v7_data_source",
    "list_v7_data_sources",
    "validate_frame_against_source",
]
