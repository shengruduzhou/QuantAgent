from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from quantagent.data.providers.base import ProviderUnavailable
from services.quant_api.services.fund_research import FundResearchService
from services.quant_api.services.market_playbooks_v3 import MarketPlaybookService


router = APIRouter(prefix="/api/market", tags=["market", "fuyao-playbooks"])


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


@router.get("/funds/overview")
def market_fund_overview(
    request: Request,
    symbol: str = Query("510300.SH"),
    fund_type: str = Query("exchange", alias="fundType"),
) -> dict[str, Any]:
    """Fuyao fund/ETF/REITs research aggregation with disclosure boundaries."""
    try:
        data = FundResearchService(request.app.state.services.market).overview(
            symbol,
            fund_type=fund_type,
        )
    except ProviderUnavailable as exc:
        return _unavailable(exc, empty=None)
    except ValueError as exc:
        return _response(
            None,
            status="unavailable",
            issues=[
                {
                    "code": "invalid_fund_request",
                    "message": str(exc),
                    "recoverable": True,
                }
            ],
        )
    status = "partial" if data.get("issues") else "ready"
    return _response(data, status=status, issues=data.get("issues", []))


@router.get("/playbooks")
def market_playbooks(request: Request) -> dict[str, Any]:
    """Return the executable 01-16 playbook catalog, not only static contracts."""
    return _response(MarketPlaybookService(request.app.state.services.market).catalog())


@router.get("/playbooks/{playbook_id}")
def market_playbook_run(
    request: Request,
    playbook_id: str,
    symbol: str = Query("600519.SH"),
    benchmark: str = Query("000300.SH"),
    index_symbol: str = Query("000300.SH", alias="indexSymbol"),
    cost_bps: float = Query(8.0, alias="costBps", ge=0.0, le=500.0),
) -> dict[str, Any]:
    """Execute one Fuyao-inspired workbench using real server-side data only."""
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
    except (KeyError, ValueError) as exc:
        return _response(
            None,
            status="unavailable",
            issues=[
                {
                    "code": "playbook_input_or_data_unavailable",
                    "message": str(exc),
                    "recoverable": True,
                }
            ],
        )
    return _response(data, status="ready")


__all__ = ["router"]
