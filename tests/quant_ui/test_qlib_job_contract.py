"""Qlib jobs must validate explicit raw contracts before recording or spawning."""
from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from services.quant_api.app import create_app
from services.quant_api.services import jobs


@pytest.fixture
def qlib_job(empty_quant_ui_settings):
    settings = empty_quant_ui_settings
    bundle = settings.runtime_root / "data" / "qlib_fixture"
    bundle.mkdir(parents=True)
    client = TestClient(create_app(settings))
    parameters = {
        "provider_uri": str(bundle), "symbols": "600000.SH",
        "start_date": "2024-01-02", "end_date": "2024-01-11",
        "output_root": "runtime/data/qlib_output",
        "raw_amount_field": "$turnover_raw_cny", "volume_scale_to_shares": 100,
    }
    return client, parameters


def test_qlib_job_accepts_and_forwards_explicit_contract(qlib_job, monkeypatch):
    client, parameters = qlib_job
    commands = []

    def capture_popen(command, **kwargs):
        commands.append(command)
        raise OSError("synthetic test stops before launching provider")

    monkeypatch.setattr(jobs.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(jobs, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: kwargs["target"](*kwargs["args"])))
    payload = {"commandId": "build-market-panel-v7", "parameters": parameters}
    validation = client.post("/api/jobs/data/validate", json=payload)
    assert validation.status_code == 200, validation.text
    assert client.get("/api/jobs").json()["data"] == []
    response = client.post("/api/jobs/data", json=payload)
    assert response.status_code == 200, response.text
    record = response.json()["data"]
    assert record["parameters"]["raw_amount_field"] == "turnover_raw_cny"
    assert record["parameters"]["volume_scale_to_shares"] == 100.
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--raw-amount-field") + 1] == "turnover_raw_cny"
    assert command[command.index("--volume-scale-to-shares") + 1] == "100.0"


@pytest.mark.parametrize("key, value", [
    ("raw_amount_field", None), ("raw_amount_field", " "),
    ("raw_amount_field", "$close"), ("raw_amount_field", "$$amount"),
    ("raw_amount_field", "Mean($amount, 5)"),
    ("volume_scale_to_shares", None), ("volume_scale_to_shares", "nan"),
    ("volume_scale_to_shares", "inf"), ("volume_scale_to_shares", 0),
    ("volume_scale_to_shares", -1), ("volume_scale_to_shares", True),
    ("volume_scale_to_shares", []),
])
def test_qlib_invalid_contract_never_queues(qlib_job, monkeypatch, key, value):
    client, parameters = qlib_job
    parameters[key] = value
    monkeypatch.setattr(jobs, "Thread", lambda **kwargs: pytest.fail("invalid contract queued a worker"))
    payload = {"commandId": "build-market-panel-v7", "parameters": parameters}
    for endpoint in ("/api/jobs/data/validate", "/api/jobs/data"):
        response = client.post(endpoint, json=payload)
        assert response.status_code == 422, response.text
        assert key in response.text
    assert client.get("/api/jobs").json()["data"] == []
