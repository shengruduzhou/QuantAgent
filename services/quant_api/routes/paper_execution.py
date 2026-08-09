from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from services.quant_api.services.paper_execution_evidence import (
    PaperExecutionEvidenceService,
)


router = APIRouter(prefix="/api/paper", tags=["paper", "execution-evidence"])


@router.get("/execution-evidence")
def execution_evidence(
    request: Request,
    limit: int = Query(default=30, ge=1, le=200),
) -> dict[str, Any]:
    """Return verified, read-only paper execution evidence for the operator UI."""

    service = PaperExecutionEvidenceService(request.app.state.services.settings)
    return {
        "status": "ready",
        "data": service.status(limit=limit),
        "issues": [],
    }


__all__ = ["router"]
