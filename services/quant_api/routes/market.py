from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from quantagent.data.providers.base import ProviderUnavailable


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
