from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Query, Request

from quantagent.data.providers.base import ProviderUnavailable
from services.quant_api.services.fuyao_analytics import attention_price_resonance, cashflow_quality
from services.quant_api.services.fuyao_best_practices import best_practice_payload


router = APIRouter(prefix="/api/market", tags=["market"])
SHANGHAI = ZoneInfo("Asia/Shanghai")


def _response(
    data: Any,
    *,
    status: str = "ready",
    issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {"status": status, "data": data, "issues": issues or []}


def _unavailable(exc: ProviderUnavailable, *, empty: Any) -> dict[str, Any]:
    return _response(
        empty,
        status="unavailable",
        issues=[
            {
                "code": "market_provider_unavailable",
                "message": str(exc),
                "recoverable": True,
            }
        ],
    )


def _items(data: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not data:
        return []
    rows = data.get("item", [])
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


@router.get("/best-practices")
def market_best_practices() -> dict[str, Any]:
    """Return the machine-readable Fuyao/Financial-API product contract."""
    return _response(best_practice_payload())


@router.get("/search")
def market_search(
    request: Request,
    q: str = Query(min_length=1, max_length=80),
    limit: int = Query(12, ge=1, le=50),
) -> dict[str, Any]:
    try:
        data = request.app.state.services.market.search(q, limit=limit)
    except ProviderUnavailable as exc:
        return _unavailable(exc, empty=[])
    return _response(data, status="ready" if data else "empty")


@router.get("/stocks/{symbol}/overview")
def market_stock_overview(
    request: Request,
    symbol: str,
    calendar_days: int = Query(420, alias="calendarDays", ge=370, le=3650),
) -> dict[str, Any]:
    try:
        data = request.app.state.services.market.stock_overview(
            symbol,
            calendar_days=calendar_days,
        )
    except ProviderUnavailable as exc:
        return _unavailable(exc, empty=None)
    return _response(data, status="ready" if data.get("bars") else "partial")


@router.get("/stocks/{symbol}/financial-health")
def market_stock_financial_health(
    request: Request,
    symbol: str,
    limit: int = Query(5, ge=1, le=10),
) -> dict[str, Any]:
    try:
        data = request.app.state.services.market.financial_health(symbol, limit=limit)
    except ProviderUnavailable as exc:
        return _unavailable(exc, empty=None)
    statements = data.get("statements", {})
    has_rows = any(bool(rows) for rows in statements.values())
    return _response(data, status="ready" if has_rows else "empty")


@router.get("/stocks/{symbol}/cashflow-quality")
def market_stock_cashflow_quality(
    request: Request,
    symbol: str,
    limit: int = Query(5, ge=1, le=10),
) -> dict[str, Any]:
    """Fuyao inspiration 10: PIT-aligned cash conversion/FCF/accrual audit."""
    try:
        financial = request.app.state.services.market.financial_health(symbol, limit=limit)
        data = cashflow_quality(financial)
    except ProviderUnavailable as exc:
        return _unavailable(exc, empty=None)
    return _response(data, status="ready" if data.get("rows") else "empty")


@router.get("/stocks/{symbol}/attention-price")
def market_stock_attention_price(
    request: Request,
    symbol: str,
    days: int = Query(90, ge=10, le=365),
    benchmark: str = Query("000300.SH", min_length=8, max_length=16),
) -> dict[str, Any]:
    """Fuyao inspiration 11: align raw hot-list rank, stock and benchmark daily bars."""
    normalized_symbol = symbol.strip().upper()
    normalized_benchmark = benchmark.strip().upper()
    end = datetime.now(SHANGHAI).date()
    start = end - timedelta(days=days)
    try:
        market = request.app.state.services.market
        provider = market._provider()  # shared authenticated/read-only provider; key remains server-side
        rank_payload = provider.get_capability(
            "/api/a-share/special-data/hot-stock-rank-trend",
            {
                "thscode": normalized_symbol,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            },
        )
        stock = market.stock_overview(normalized_symbol, calendar_days=max(420, days + 120))
        benchmark_view = market.index_overview(normalized_benchmark, calendar_days=max(180, min(730, days + 60)))
        data = attention_price_resonance(
            symbol=normalized_symbol,
            benchmark=normalized_benchmark,
            rank_rows=_items(rank_payload),
            stock_bars=stock.get("bars", []),
            benchmark_bars=benchmark_view.get("bars", []),
        )
        data["provenance"] = {
            "rank": "/api/a-share/special-data/hot-stock-rank-trend",
            "stock": "/api/a-share/prices/historical?adjust=forward",
            "benchmark": "/api/a-share-index/prices/historical",
        }
    except ProviderUnavailable as exc:
        return _unavailable(exc, empty=None)
    return _response(data, status="ready" if data.get("rows") else "empty")


@router.get("/heat-radar")
def market_heat_radar(
    request: Request,
    trend_days: int = Query(30, alias="trendDays", ge=5, le=365),
) -> dict[str, Any]:
    """Fuyao inspiration 07: current day/hour hot/skyrocket views + top-3 rank trends."""
    market = request.app.state.services.market
    end = datetime.now(SHANGHAI).date()
    start = end - timedelta(days=trend_days)
    try:
        current = market.market_intelligence()
        hot_day = _items((current.get("panels") or {}).get("hotDay"))
        provider = market._provider()
        trends: dict[str, Any] = {}
        trend_issues: list[dict[str, str]] = []
        for row in hot_day[:3]:
            symbol = str(row.get("thscode") or "").strip().upper()
            if not symbol:
                continue
            try:
                trends[symbol] = provider.get_capability(
                    "/api/a-share/special-data/hot-stock-rank-trend",
                    {"thscode": symbol, "start_date": start.isoformat(), "end_date": end.isoformat()},
                )
            except ProviderUnavailable as exc:
                trends[symbol] = None
                trend_issues.append({"symbol": symbol, "message": str(exc)})
        data = {
            "source": "hithink_fuyao",
            "current": {
                key: (current.get("panels") or {}).get(key)
                for key in ("hotDay", "hotHour", "skyrocketDay", "skyrocketHour")
            },
            "top3RankTrends": trends,
            "issues": [*(current.get("issues") or []), *trend_issues],
            "window": {"start": start.isoformat(), "end": end.isoformat(), "days": trend_days},
            "boundary": "Hot list and skyrocket list retain separate semantics; attention is not a trading signal.",
            "provenance": {
                "hot": "/api/a-share/special-data/hot-stock-list",
                "skyrocket": "/api/a-share/special-data/skyrocket-list",
                "rankTrend": "/api/a-share/special-data/hot-stock-rank-trend",
            },
        }
    except ProviderUnavailable as exc:
        return _unavailable(exc, empty=None)
    return _response(data, status="partial" if data["issues"] else "ready", issues=data["issues"])


@router.get("/indexes")
def market_index_catalog(
    request: Request,
    tag: str = Query("industry"),
) -> dict[str, Any]:
    try:
        data = request.app.state.services.market.index_catalog(tag)
    except ProviderUnavailable as exc:
        return _unavailable(exc, empty=None)
    except ValueError as exc:
        return _response(
            None,
            status="unavailable",
            issues=[{"code": "invalid_index_tag", "message": str(exc), "recoverable": True}],
        )
    return _response(data, status="ready" if data.get("items") else "empty")


@router.get("/indexes/{symbol}/overview")
def market_index_overview(
    request: Request,
    symbol: str,
    calendar_days: int = Query(180, alias="calendarDays", ge=90, le=730),
) -> dict[str, Any]:
    try:
        data = request.app.state.services.market.index_overview(
            symbol,
            calendar_days=calendar_days,
        )
    except ProviderUnavailable as exc:
        return _unavailable(exc, empty=None)
    return _response(data, status="ready" if data.get("bars") else "partial")


@router.get("/intelligence")
def market_intelligence(request: Request) -> dict[str, Any]:
    try:
        data = request.app.state.services.market.market_intelligence()
    except ProviderUnavailable as exc:
        return _unavailable(exc, empty=None)
    issues = data.get("issues", [])
    panels = data.get("panels", {})
    ready_count = sum(value is not None for value in panels.values())
    if ready_count == 0:
        status = "unavailable"
    elif issues:
        status = "partial"
    else:
        status = "ready"
    return _response(data, status=status, issues=issues)
