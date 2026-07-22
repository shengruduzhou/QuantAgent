from __future__ import annotations

import asyncio

import httpx
import pytest

from services.quant_api.app import create_app
from services.quant_api.services.jobs import JobManager, JobRecord


def request(app, method: str, url: str, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def test_data_provider_registry_is_explicit_and_never_exposes_credentials(quant_ui_settings, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "do-not-expose-this-token")

    result = request(create_app(quant_ui_settings), "GET", "/api/data/providers")

    assert result.status_code == 200
    payload = result.json()["data"]
    assert payload["supportsCancellation"] is True
    assert any(provider["id"] == "runtime_catalog" for provider in payload["providers"])
    tushare = next(provider for provider in payload["providers"] if provider["id"] == "tushare_fundamentals")
    assert tushare["configured"] is True
    assert tushare["missingRequirements"] == []
    assert "do-not-expose-this-token" not in result.text
    assert "tokenValue" not in result.text


def test_data_job_requires_allowlisted_command_and_explicit_universe(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    arbitrary = request(
        app,
        "POST",
        "/api/jobs/data",
        json={"commandId": "bash", "parameters": {}},
    )
    assert arbitrary.status_code == 422

    no_universe = request(
        app,
        "POST",
        "/api/jobs/data",
        json={
            "commandId": "build-akshare-market-panel-v7",
            "parameters": {
                "start_date": "2026-01-01",
                "end_date": "2026-01-31",
                "output": "runtime/data/web/market.parquet",
                "allow_network": True,
            },
        },
    )
    assert no_universe.status_code == 422
    assert "one of" in no_universe.json()["detail"]


def test_queued_job_can_be_cancelled_without_starting_a_process(quant_ui_settings) -> None:
    manager = JobManager(quant_ui_settings)
    manager._jobs["job_cancel"] = JobRecord(
        id="job_cancel",
        type="data",
        status="queued",
        commandId="build-akshare-market-panel-v7",
        createdAt="2026-07-22T00:00:00+00:00",
    )

    cancelled = manager.cancel("job_cancel")

    assert cancelled["status"] == "cancelled"
    assert cancelled["finishedAt"] is not None


def test_terminal_job_cannot_be_cancelled_again(quant_ui_settings) -> None:
    manager = JobManager(quant_ui_settings)
    manager._jobs["job_done"] = JobRecord(
        id="job_done",
        type="data",
        status="succeeded",
        commandId="build-akshare-market-panel-v7",
        createdAt="2026-07-22T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="already finished"):
        manager.cancel("job_done")
