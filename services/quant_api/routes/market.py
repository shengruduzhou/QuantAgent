from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from quantagent.data.providers.base import ProviderUnavailable
from services.quant_api.services.fund_research import FundResearchService
from services.quant_api.services.market_playbooks_v2 import MarketPlaybookService


router = APIRouter(prefix="/api/market", tags=["market"])


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


@router.get("/funds/overview")
def market_fund_overview(
    request: Request,
    symbol: str = Query("510300.SH"),
    fund_type: str = Query("exchange", alias="fundType"),
) -> dict[str, Any]:
    try:
        data = FundResearchService(request.app.state.services.market).overview(symbol, fund_type=fund_type)
    except ProviderUnavailable as exc:
        return _unavailable(exc, empty=None)
    except ValueError as exc:
        return _response(None, status="unavailable", issues=[{"code": "invalid_fund_request", "message": str(exc), "recoverable": True}])
    status = "partial" if data.get("issues") else "ready"
    return _response(data, status=status, issues=data.get("issues", []))


@router.get("/playbooks")
def market_playbooks(request: Request) -> dict[str, Any]:
    return _response(MarketPlaybookService(request.app.state.services.market).catalog())


@router.get("/playbooks/{playbook_id}")
def market_playbook_run(
    request: Request,
    playbook_id: str,
    symbol: str = Query("600519.SH"),
    benchmark: str = Query("000300.SH"),
    index_symbol: str = Query("881101.TI", alias="indexSymbol"),
    cost_bps: float = Query(8.0, alias="costBps", ge=0.0, le=500.0),
) -> dict[str, Any]:
    service = MarketPlaybookService(request.app.state.services.market)
    try:
        data = service.run(
            playbook_id,
            symbol=symbol,
            benchmark=benchmark,
            index_symbol=index_symbol,
            cost_bps=cost_bps,
        )
    except ProviderUnavailable as exc:
        return _unavailable(exc, empty=None)
    except ValueError as exc:
        return _response(None, status="unavailable", issues=[{"code": "playbook_input_or_data_unavailable", "message": str(exc), "recoverable": True}])
    return _response(data, status="ready")
