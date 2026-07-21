from __future__ import annotations

import asyncio

import httpx

from services.quant_api.app import create_app
from services.quant_api.services.vnpy_parity import VnpyParityService


def request(app, method: str, url: str, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def test_registry_is_valid_versioned_and_has_unique_capabilities() -> None:
    registry = VnpyParityService().load()

    assert registry.schema_version == "quantagent.vnpy-parity.v1"
    assert registry.source_baseline.release == "4.4.0"
    assert registry.completeness == "partial"
    ids = [item.id for item in registry.capabilities]
    assert len(ids) == len(set(ids))
    assert {"core", "data", "risk", "visualization", "service"}.issubset(
        {item.category for item in registry.capabilities}
    )


def test_parity_api_filters_and_exposes_honest_summary(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)

    result = request(
        app,
        "GET",
        "/api/system/vnpy-parity",
        params={"category": "risk", "status": "partial"},
    )

    assert result.status_code == 200
    payload = result.json()
    assert payload["status"] == "ready"
    assert payload["data"]["schemaVersion"] == "quantagent.vnpy-parity.v1"
    assert payload["data"]["summary"]["total"] == 1
    assert payload["data"]["capabilities"][0]["id"] == "risk.risk_manager"
    assert payload["provenance"]["sourceType"] == "validated_registry"


def test_parity_api_returns_empty_for_unmatched_query(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)

    result = request(
        app,
        "GET",
        "/api/system/vnpy-parity",
        params={"query": "capability-that-does-not-exist"},
    )

    assert result.status_code == 200
    payload = result.json()
    assert payload["status"] == "empty"
    assert payload["data"]["summary"]["total"] == 0
    assert payload["data"]["capabilities"] == []
