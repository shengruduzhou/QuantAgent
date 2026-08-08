from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from services.quant_api.services.production_readiness import ProductionReadinessService


router = APIRouter(prefix="/api/governance", tags=["governance", "production-readiness"])


@router.get("/production-readiness")
def production_readiness(request: Request) -> dict[str, Any]:
    """Return side-effect-free operator truth from fixed machine evidence only."""
    settings = request.app.state.services.settings
    service = ProductionReadinessService(settings.project_root, settings.runtime_root)
    return {
        "status": "ready",
        "data": service.status(),
        "issues": [],
    }


__all__ = ["router"]
