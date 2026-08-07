"""Resumable, auditable full Fuyao acquisition orchestration.

"Full" means every *currently documented and callable* Fuyao data capability is
represented by an explicit acquisition strategy.  It does not pretend that the
vendor exposes data outside its documented retention windows, entitlements or
coming-soon modules.  Every failed/upstream-limited request is recorded in the
run report; nothing is silently replaced with synthetic data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from quantagent.data.fuyao_catalog import (
    DUMP_CAPABILITIES,
    PLANNED_CAPABILITIES,
    REST_CAPABILITIES,
    coverage_summary,
)
from quantagent.data.fuyao_dump import download_fuyao_market_dump
from quantagent.data.providers.base import ProviderUnavailable
from quantagent.data.providers.fuyao_provider import FuyaoProvider


# Every live REST capability has exactly one declared acquisition strategy.
# Tests intentionally compare these keys with REST_CAPABILITIES so docs/adapter
# drift cannot become a silent omission.
SYNC_STRATEGIES: dict[str, str] = {
    "meta.ticker_search": "covered_by_complete_ticker_universe",
    "meta.ticker_list": "paginate_all_asset_types",
    "a_share.prices_snapshot": "batch_all_a_shares",
    "a_share.prices_historical": "bulk_dump_raw_plus_adjustment_factors",
    "a_share.valuation_snapshot": "batch_all_a_shares",
    "a_share.adjustment_factors": "bulk_dump_adjustment_factors",
    "a_share.income_statements": "per_a_share_annual_and_quarterly",
    "a_share.balance_sheets": "per_a_share_annual_and_quarterly",
    "a_share.cash_flow_statements": "per_a_share_annual_and_quarterly",
    "a_share.financial_indicators": "per_a_share_per_quarter_non_pit",
    "a_share.trading_calendar": "fetch_once_recent_year",
    "a_share.limit_up_pool": "per_trading_day_paginated",
    "a_share.limit_up_ladder": "fetch_once_fixed_30d_window",
    "a_share.skyrocket_list": "fetch_day_and_hour",
    "a_share.hot_stock_list": "fetch_day_and_hour",
    "a_share.hot_stock_history": "per_natural_day_recent_year",
    "a_share.hot_stock_rank_trend": "per_a_share_recent_year",
    "a_share.anomaly_list": "fetch_current_rest_only",
    "a_share.anomaly_stock": "batch_all_a_shares_50",
    "a_share.dragon_tiger": "per_trading_day_all_three_boards",
    "index.catalog": "fetch_all_four_tags",
    "index.constituents": "per_current_index",
    "index.prices_snapshot": "batch_all_current_indexes",
    "index.prices_historical": "per_current_index_10y",
    "fund.profile": "per_enumerated_fund",
    "fund.holdings": "per_enumerated_fund_latest_disclosure",
    "fund.nav": "per_enumerated_fund_fyear",
    "fund.returns": "per_enumerated_fund",
    "fund.holders": "per_enumerated_fund_latest_scopes",
    "fund.market_snapshot": "per_etf_only",
    "fund.market_historical": "per_etf_only_5y",
}

DUMP_STRATEGIES: dict[str, str] = {
    "dump.daily_k": "download_validated_parquet",
    "dump.daily_k_10d": "download_validated_parquet",
    "dump.adjustment_factors": "download_validated_parquet",
}


@dataclass(frozen=True, slots=True)
class SyncEvent:
    capability_id: str
    endpoint: str | None
    status: str
    output: str | None = None
    params: dict[str, Any] | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class FuyaoUniverse:
    a_shares: tuple[str, ...]
    indexes: tuple[str, ...]
    fund_otc: tuple[str, ...]
    fund_etf: tuple[str, ...]
    fund_lof: tuple[str, ...]
    fund_reits: tuple[str, ...]

    @property
    def exchange_funds(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.fund_etf, *self.fund_lof)))


def validate_sync_coverage() -> None:
    rest_ids = {cap.id for cap in REST_CAPABILITIES}
    strategy_ids = set(SYNC_STRATEGIES)
    if rest_ids != strategy_ids:
        missing = sorted(rest_ids - strategy_ids)
        extra = sorted(strategy_ids - rest_ids)
        raise RuntimeError(f"Fuyao full-sync strategy drift: missing={missing} extra={extra}")
    dump_ids = {cap.id for cap in DUMP_CAPABILITIES}
    if dump_ids != set(DUMP_STRATEGIES):
        raise RuntimeError("Fuyao market-dump strategy drift")


validate_sync_coverage()


def build_coverage_audit() -> dict[str, Any]:
    """Return a static, serialisable proof that every documented capability is classified."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_index": "https://fuyao.aicubes.cn/llms.txt",
        "official_full_contract": "https://fuyao.aicubes.cn/llms-full.txt",
        "counts": coverage_summary(),
        "rest": [
            {**cap.as_dict(), "sync_strategy": SYNC_STRATEGIES[cap.id]}
            for cap in REST_CAPABILITIES
        ],
        "dumps": [
            {**cap.as_dict(), "sync_strategy": DUMP_STRATEGIES[cap.id]}
            for cap in DUMP_CAPABILITIES
        ],
        "coming_soon": [cap.as_dict() for cap in PLANNED_CAPABILITIES],
        "hard_limits": [
            "A-share historical K endpoint currently supports 1d and a maximum 10-year query window.",
            "Full-market daily-K dump is unadjusted; the adjustment-factor dump is stored separately.",
            "A-share trading calendar exposes only the recent one year.",
            "Hot-stock history/rank trend and dragon-tiger history are limited to one year.",
            "Limit-up ladder is a fixed recent-30-trading-day view.",
            "A-share valuation is latest snapshot only; no historical valuation endpoint is documented.",
            "Fund NAV history is bounded to fyear (5y); fund exchange market history is ETF-only and max 5 natural years.",
            "Meta ticker enumeration documents fund-otc/fund-etf/fund-lof but no fund-reit leaf type; REIT symbols must be supplied explicitly until Fuyao publishes a REIT enumeration path.",
            "Financial indicators do not document report_date_ms and therefore are archived as non-PIT until joined to a disclosure timestamp.",
            "Coming-soon stock basics, historical index constituents/weights and stock-to-THS-index membership cannot be fetched yet.",
        ],
    }


class FuyaoFullSynchronizer:
    """Archive all documented Fuyao datasets within their published boundaries.

    The synchronizer is intentionally sequential and resumable.  Exhaustive mode
    can generate many vendor calls (especially financial indicators and fund
    data), so an interrupted run can safely continue without re-fetching already
    archived requests.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        provider: FuyaoProvider | None = None,
        resume: bool = True,
        stop_on_error: bool = False,
    ) -> None:
        self.root = Path(root)
        self.provider = provider or FuyaoProvider(allow_network=True)
        self.resume = resume
        self.stop_on_error = stop_on_error
        self.events: list[SyncEvent] = []

    def run(
        self,
        *,
        deep: bool = True,
        extra_reits: Iterable[str] = (),
        as_of: date | None = None,
        include_dumps: bool = True,
        dump_timeout: float = 180.0,
    ) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        today = as_of or date.today()
        one_year_start = today - timedelta(days=365)
        ten_year_start = today - timedelta(days=3652)
        five_year_start = today - timedelta(days=1826)

        universe = self._sync_universes(tuple(extra_reits))
        trading_days = self._sync_calendar()

        if include_dumps:
            self._sync_dumps(timeout=dump_timeout)

        self._sync_a_share_current(universe)
        self._sync_special_current(universe)
        self._sync_index_catalogs()

        if deep:
            self._sync_a_share_financials(universe, ten_year_start, today)
            self._sync_financial_indicators(universe, ten_year_start, today)
            self._sync_special_history(universe, trading_days, one_year_start, today)
            self._sync_indexes(universe, ten_year_start, today)
            self._sync_funds(universe, five_year_start, today)

        return self._write_report(universe=universe, deep=deep, as_of=today)

    # ------------------------------------------------------------------
    # Universe and generic persistence
    # ------------------------------------------------------------------
    def _sync_universes(self, extra_reits: tuple[str, ...]) -> FuyaoUniverse:
        asset_types = ("a-share", "a-share-index", "fund-otc", "fund-etf", "fund-lof")
        resolved: dict[str, tuple[str, ...]] = {}
        for asset_type in asset_types:
            rows: list[dict[str, Any]] = []
            offset = 0
            limit = 10_000
            while True:
                data = self._fetch(
                    "meta.ticker_list",
                    "/api/meta/tickers/list",
                    {"asset_type": asset_type, "limit": limit, "offset": offset},
                    qualifier=f"{asset_type}/offset-{offset}",
                )
                if data is None:
                    break
                page = _items(data)
                rows.extend(page)
                if len(page) < limit:
                    break
                offset += limit
            resolved[asset_type] = tuple(
                dict.fromkeys(str(row.get("thscode") or "").upper() for row in rows if row.get("thscode"))
            )
        reits = tuple(dict.fromkeys(code.strip().upper() for code in extra_reits if code.strip()))
        if not reits:
            self.events.append(
                SyncEvent(
                    "fund.profile",
                    None,
                    "upstream_enumeration_gap",
                    message="Fuyao meta ticker-list has no documented fund-reit asset_type; pass explicit REIT thscodes to cover REIT fund endpoints.",
                )
            )
        return FuyaoUniverse(
            a_shares=resolved.get("a-share", ()),
            indexes=resolved.get("a-share-index", ()),
            fund_otc=resolved.get("fund-otc", ()),
            fund_etf=resolved.get("fund-etf", ()),
            fund_lof=resolved.get("fund-lof", ()),
            fund_reits=reits,
        )

    def _fetch(
        self,
        capability_id: str,
        endpoint: str,
        params: Mapping[str, Any] | None = None,
        *,
        qualifier: str = "default",
    ) -> dict[str, Any] | None:
        clean_params = dict(params or {})
        target = self._artifact_path(capability_id, qualifier, clean_params)
        if self.resume and target.exists():
            try:
                artifact = json.loads(target.read_text(encoding="utf-8"))
                data = artifact.get("data")
                if isinstance(data, dict):
                    self.events.append(SyncEvent(capability_id, endpoint, "resumed", str(target), clean_params))
                    return data
            except (OSError, json.JSONDecodeError):
                pass
        try:
            data = self.provider.get_capability(endpoint, params=clean_params)
            artifact = {
                "source": "hithink_fuyao",
                "capability_id": capability_id,
                "endpoint": endpoint,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "request_params": clean_params,
                "data": data,
            }
            _atomic_json(target, artifact)
            self.events.append(SyncEvent(capability_id, endpoint, "success", str(target), clean_params))
            return data
        except (ProviderUnavailable, ValueError, RuntimeError) as exc:
            self.events.append(SyncEvent(capability_id, endpoint, "failed", None, clean_params, str(exc)))
            if self.stop_on_error:
                raise
            return None

    def _artifact_path(self, capability_id: str, qualifier: str, params: Mapping[str, Any]) -> Path:
        digest = hashlib.sha256(
            json.dumps(params, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:12]
        safe_qualifier = qualifier.replace("..", "_").replace("\\", "_").strip("/") or "default"
        return self.root / "raw" / capability_id.replace(".", "/") / safe_qualifier / f"{digest}.json"

    # ------------------------------------------------------------------
    # Bulk / current data
    # ------------------------------------------------------------------
    def _sync_dumps(self, *, timeout: float) -> None:
        for dataset, capability_id in (
            ("daily-k", "dump.daily_k"),
            ("daily-k-10d", "dump.daily_k_10d"),
            ("adjustment-factors", "dump.adjustment_factors"),
        ):
            target = self.root / "dumps" / f"{dataset}.parquet"
            manifest = target.with_suffix(target.suffix + ".manifest.json")
            if self.resume and target.exists() and manifest.exists():
                self.events.append(SyncEvent(capability_id, None, "resumed", str(target)))
                continue
            try:
                result = download_fuyao_market_dump(
                    dataset,
                    target,
                    allow_network=True,
                    timeout=timeout,
                    provider=self.provider,
                )
                self.events.append(SyncEvent(capability_id, None, "success", str(result.output)))
            except Exception as exc:  # download/Parquet errors must be preserved in the audit report
                self.events.append(SyncEvent(capability_id, None, "failed", None, None, str(exc)))
                if self.stop_on_error:
                    raise

    def _sync_calendar(self) -> tuple[str, ...]:
        data = self._fetch("a_share.trading_calendar", "/api/a-share/calendar/trading-days")
        days: list[str] = []
        if data:
            for row in _items(data):
                value = row.get("date") or row.get("trade_date")
                if isinstance(value, str):
                    days.append(value[:10])
                elif row.get("date_ms") is not None:
                    days.append(datetime.fromtimestamp(int(row["date_ms"]) / 1000, timezone.utc).date().isoformat())
        return tuple(dict.fromkeys(days))

    def _sync_a_share_current(self, universe: FuyaoUniverse) -> None:
        for batch_no, batch in enumerate(_batches(universe.a_shares, 50)):
            token = ",".join(batch)
            self._fetch("a_share.prices_snapshot", "/api/a-share/prices/snapshot", {"thscodes": token}, qualifier=f"batch-{batch_no:04d}")
        for batch_no, batch in enumerate(_batches(universe.a_shares, 100)):
            self._fetch("a_share.valuation_snapshot", "/api/a-share/valuations/snapshot", {"thscodes": ",".join(batch)}, qualifier=f"batch-{batch_no:04d}")

    def _sync_special_current(self, universe: FuyaoUniverse) -> None:
        for capability, endpoint in (
            ("a_share.skyrocket_list", "/api/a-share/special-data/skyrocket-list"),
            ("a_share.hot_stock_list", "/api/a-share/special-data/hot-stock-list"),
        ):
            for period in ("day", "hour"):
                self._fetch(capability, endpoint, {"period": period}, qualifier=period)
        self._fetch("a_share.limit_up_ladder", "/api/a-share/special-data/limit-up-ladder")
        self._fetch("a_share.anomaly_list", "/api/a-share/special-data/anomaly-analysis-list")
        for batch_no, batch in enumerate(_batches(universe.a_shares, 50)):
            self._fetch(
                "a_share.anomaly_stock",
                "/api/a-share/special-data/anomaly-analysis-stock",
                {"thscodes": ",".join(batch)},
                qualifier=f"batch-{batch_no:04d}",
            )
        # Current limit-pool pagination.
        self._sync_limit_pool_for_date(None)

    def _sync_index_catalogs(self) -> None:
        for tag in ("industry", "cn_concept", "region", "tszs"):
            self._fetch("index.catalog", "/api/a-share-index/catalog/ths-index-list", {"tag": tag}, qualifier=tag)

    # ------------------------------------------------------------------
    # Exhaustive/deep data
    # ------------------------------------------------------------------
    def _sync_a_share_financials(self, universe: FuyaoUniverse, start: date, end: date) -> None:
        start_ms, end_ms = _date_ms(start), _date_ms(end)
        endpoints = (
            ("a_share.income_statements", "/api/a-share/financials/income-statements"),
            ("a_share.balance_sheets", "/api/a-share/financials/balance-sheets"),
            ("a_share.cash_flow_statements", "/api/a-share/financials/cash-flow-statements"),
        )
        for symbol in universe.a_shares:
            for capability, endpoint in endpoints:
                for period in ("annual", "quarterly"):
                    self._fetch(
                        capability,
                        endpoint,
                        {"thscode": symbol, "period": period, "start": start_ms, "end": end_ms},
                        qualifier=f"{symbol}/{period}",
                    )

    def _sync_financial_indicators(self, universe: FuyaoUniverse, start: date, end: date) -> None:
        for symbol in universe.a_shares:
            for year in range(start.year, end.year + 1):
                for quarter in range(1, 5):
                    self._fetch(
                        "a_share.financial_indicators",
                        "/api/a-share/financials/indicators",
                        {"thscode": symbol, "report": f"{year}-{quarter}"},
                        qualifier=f"{symbol}/{year}-{quarter}",
                    )

    def _sync_special_history(
        self,
        universe: FuyaoUniverse,
        trading_days: tuple[str, ...],
        start: date,
        end: date,
    ) -> None:
        # History endpoint is natural-day based, so archive all 366 possible days.
        day = start
        while day <= end:
            self._fetch(
                "a_share.hot_stock_history",
                "/api/a-share/special-data/hot-stock-list-history",
                {"date": day.isoformat()},
                qualifier=day.isoformat(),
            )
            day += timedelta(days=1)

        start_text, end_text = start.isoformat(), end.isoformat()
        for symbol in universe.a_shares:
            self._fetch(
                "a_share.hot_stock_rank_trend",
                "/api/a-share/special-data/hot-stock-rank-trend",
                {"thscode": symbol, "start_date": start_text, "end_date": end_text},
                qualifier=symbol,
            )

        for trade_date in trading_days:
            parsed = _parse_date(trade_date)
            if parsed is None or parsed < start or parsed > end:
                continue
            self._sync_limit_pool_for_date(trade_date)
            for board in ("all", "org", "hot_money"):
                self._fetch(
                    "a_share.dragon_tiger",
                    "/api/a-share/special-data/dragon-tiger-list",
                    {"board_type": board, "date": trade_date},
                    qualifier=f"{trade_date}/{board}",
                )

    def _sync_limit_pool_for_date(self, trade_date: str | None) -> None:
        page = 1
        while True:
            params: dict[str, Any] = {"page": page, "size": 200, "sort_field": "continue_day_cnt", "sort_dir": "desc"}
            qualifier = "current"
            if trade_date:
                parsed = _parse_date(trade_date)
                if parsed is None:
                    return
                params["date_ms"] = _date_ms(parsed)
                qualifier = trade_date
            data = self._fetch(
                "a_share.limit_up_pool",
                "/api/a-share/special-data/limit-up-pool",
                params,
                qualifier=f"{qualifier}/page-{page:03d}",
            )
            if not data:
                break
            pagination = data.get("pagination")
            pages = int(pagination.get("pages", 1)) if isinstance(pagination, Mapping) else 1
            if page >= pages:
                break
            page += 1

    def _sync_indexes(self, universe: FuyaoUniverse, start: date, end: date) -> None:
        for batch_no, batch in enumerate(_batches(universe.indexes, 50)):
            self._fetch(
                "index.prices_snapshot",
                "/api/a-share-index/prices/snapshot",
                {"thscodes": ",".join(batch)},
                qualifier=f"batch-{batch_no:04d}",
            )
        start_ms, end_ms = _date_ms(start), _date_ms(end)
        for symbol in universe.indexes:
            self._fetch("index.constituents", "/api/a-share-index/constituents/ths-stock-list", {"thscode": symbol}, qualifier=symbol)
            self._fetch(
                "index.prices_historical",
                "/api/a-share-index/prices/historical",
                {"thscode": symbol, "interval": "1d", "start": start_ms, "end": end_ms},
                qualifier=symbol,
            )

    def _sync_funds(self, universe: FuyaoUniverse, five_year_start: date, end: date) -> None:
        groups = (
            ("otc", universe.fund_otc),
            ("exchange", universe.exchange_funds),
            ("reits", universe.fund_reits),
        )
        for fund_type, symbols in groups:
            for symbol in symbols:
                common = {"fund_type": fund_type, "thscode": symbol}
                self._fetch("fund.profile", "/api/fund/profile/detail", common, qualifier=f"{fund_type}/{symbol}")
                self._fetch("fund.holdings", "/api/fund/portfolio/holdings", common, qualifier=f"{fund_type}/{symbol}")
                self._fetch("fund.nav", "/api/fund/performance/nav", {**common, "range": "fyear", "nav_type": "unit,adj"}, qualifier=f"{fund_type}/{symbol}")
                self._fetch("fund.returns", "/api/fund/performance/returns", common, qualifier=f"{fund_type}/{symbol}")
                self._fetch("fund.holders", "/api/fund/holders/detail", {**common, "merge_scope": "all"}, qualifier=f"{fund_type}/{symbol}")

        start_ms, end_ms = _date_ms(five_year_start), _date_ms(end)
        for symbol in universe.fund_etf:
            self._fetch("fund.market_snapshot", "/api/fund/market/snapshot", {"thscode": symbol}, qualifier=symbol)
            self._fetch(
                "fund.market_historical",
                "/api/fund/market/historical",
                {"thscode": symbol, "interval": "1d", "start": start_ms, "end": end_ms},
                qualifier=symbol,
            )

    # ------------------------------------------------------------------
    # Final audit report
    # ------------------------------------------------------------------
    def _write_report(self, *, universe: FuyaoUniverse, deep: bool, as_of: date) -> Path:
        statuses: dict[str, set[str]] = {}
        for event in self.events:
            statuses.setdefault(event.capability_id, set()).add(event.status)
        capability_status = {}
        for cap in REST_CAPABILITIES:
            capability_status[cap.id] = {
                "strategy": SYNC_STRATEGIES[cap.id],
                "observed_statuses": sorted(statuses.get(cap.id, {"not_exercised"})),
            }
        for cap in DUMP_CAPABILITIES:
            capability_status[cap.id] = {
                "strategy": DUMP_STRATEGIES[cap.id],
                "observed_statuses": sorted(statuses.get(cap.id, {"not_exercised"})),
            }
        for cap in PLANNED_CAPABILITIES:
            capability_status[cap.id] = {
                "strategy": "unavailable_upstream",
                "observed_statuses": ["coming_soon"],
            }

        report = {
            "source": "hithink_fuyao",
            "as_of": as_of.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "deep": deep,
            "coverage_contract": build_coverage_audit(),
            "universe_counts": {
                "a_share": len(universe.a_shares),
                "index": len(universe.indexes),
                "fund_otc": len(universe.fund_otc),
                "fund_etf": len(universe.fund_etf),
                "fund_lof": len(universe.fund_lof),
                "fund_reits_explicit": len(universe.fund_reits),
            },
            "capability_status": capability_status,
            "event_counts": _count_statuses(self.events),
            "events": [asdict(event) for event in self.events],
            "completeness_rule": "No silent gaps: unavailable permissions, retention limits, upstream errors and coming-soon modules remain explicit in this report.",
        }
        target = self.root / "fuyao_full_sync_report.json"
        _atomic_json(target, report)
        return target


def _items(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = data.get("item", [])
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _batches(values: tuple[str, ...], size: int) -> Iterable[tuple[str, ...]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _date_ms(value: date) -> int:
    return int(datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp() * 1000)


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _count_statuses(events: list[SyncEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        counts[event.status] = counts.get(event.status, 0) + 1
    return counts


__all__ = [
    "DUMP_STRATEGIES",
    "SYNC_STRATEGIES",
    "FuyaoFullSynchronizer",
    "FuyaoUniverse",
    "SyncEvent",
    "build_coverage_audit",
    "validate_sync_coverage",
]
