from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from quantagent.data.providers.base import ProviderUnavailable
from services.quant_api.services.fuyao_best_practices import best_practice_payload


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


@router.get("/best-practices")
def market_best_practices() -> dict[str, Any]:
    """Return the machine-readable Fuyao/Financial-API product contract.

    This endpoint contains no market observations and needs no API key.  It lets
    the UI expose the exact analytical/output boundary of every upstream best
    practice while live data continues to come from the dedicated endpoints.
    """
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
