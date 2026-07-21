from __future__ import annotations

import json

from fastapi import APIRouter, Query, Request
from pydantic import ValidationError

from services.quant_api.schemas.parity import VnpyParityStatus


router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/vnpy-parity")
async def vnpy_parity(
    request: Request,
    category: str | None = None,
    status: VnpyParityStatus | None = None,
    query: str | None = None,
    refresh: bool = Query(False),
) -> dict:
    try:
        view = request.app.state.services.vnpy_parity.view(
            category=category,
            status=status,
            query=query,
            refresh=refresh,
        )
    except (OSError, ValueError, json.JSONDecodeError, ValidationError) as exc:
        return {
            "status": "error",
            "data": None,
            "issues": [{
                "code": "vnpy_parity_registry_invalid",
                "message": str(exc),
                "path": None,
                "recoverable": True,
            }],
        }

    data = view.model_dump(by_alias=True)
    return {
        "status": "ready" if data["capabilities"] else "empty",
        "data": data,
        "issues": [],
        "provenance": {
            "sourcePath": "services/quant_api/resources/vnpy_capability_parity.v1.json",
            "sourceType": "validated_registry",
            "parser": "pydantic",
        },
    }
