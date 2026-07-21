from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import httpx

from services.quant_api.app import create_app
from services.quant_api.config import ApiSettings


def request(app, method: str, url: str, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def test_api_smoke_and_required_routes(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    health = request(app, "GET", "/health")
    assert health.status_code == 200

    backtests = request(app, "GET", "/api/backtests").json()
    assert backtests["status"] == "ready"
    backtest_id = backtests["data"][0]["id"]

    urls = [
        "/api/system/overview",
        "/api/system/runtime-index?pageSize=5",
        "/api/system/runtime-cleanup",
        f"/api/backtests/{backtest_id}",
        f"/api/backtests/{backtest_id}/equity",
        f"/api/backtests/{backtest_id}/trades",
        f"/api/backtests/{backtest_id}/stocks/000001.SZ",
        f"/api/backtests/{backtest_id}/stocks/000001.SZ/kline",
        f"/api/backtests/{backtest_id}/stocks/000001.SZ/signals",
        f"/api/backtests/{backtest_id}/stocks/000001.SZ/trades",
        f"/api/backtests/{backtest_id}/stocks/000001.SZ/t-analysis",
        "/api/factors",
        "/api/factors/alpha001/explanation",
        "/api/factors/alpha001/backtest",
        "/api/factors/alpha001/stocks/000001.SZ/trades",
        "/api/selection/runs",
        "/api/models",
        "/api/risk/overview",
        "/api/risk/events",
        "/api/risk/stocks",
        "/api/risk/rules",
        "/api/do-t/sources",
        "/api/jobs",
    ]
    for url in urls:
        result = request(app, "GET", url)
        assert result.status_code == 200, (url, result.text)
        assert result.json()["status"] in {"ready", "partial", "empty"}

    models = request(app, "GET", "/api/models").json()["data"]
    model_id = models[0]["id"]
    for url in (
        f"/api/models/{model_id}/observability",
        f"/api/models/{model_id}/training-metrics",
        f"/api/models/{model_id}/feature-importance",
        f"/api/models/{model_id}/predictions",
    ):
        result = request(app, "GET", url)
        assert result.status_code == 200, (url, result.text)

    compare = request(
        app,
        "GET",
        "/api/models/compare",
        params={"ids": ",".join(item["id"] for item in models[:2])},
    )
    assert compare.status_code == 200
    assert len(compare.json()["data"]["models"]) == 2


def test_factor_placeholder_returns_empty_not_error(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    result = request(app, "GET", "/api/factors/alpha001/stocks/000001.SZ/trades")
    assert result.status_code == 200
    assert result.json()["status"] == "empty"
    assert result.json()["issues"][0]["code"] == "independent_factor_trades_missing"


def test_missing_selection_stock_returns_empty_not_404(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_id = request(app, "GET", "/api/selection/runs").json()["data"][0]["id"]

    result = request(app, "GET", f"/api/selection/runs/{run_id}/stocks/999999.SZ/decision-chain")

    assert result.status_code == 200
    assert result.json()["status"] == "empty"
    assert result.json()["issues"][0]["code"] == "selection_stock_not_found"


def test_job_api_rejects_arbitrary_command_and_path_escape(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    arbitrary = request(
        app,
        "POST",
        "/api/jobs/train",
        json={"commandId": "bash", "parameters": {}},
    )
    assert arbitrary.status_code == 422

    escaped = request(
        app,
        "POST",
        "/api/jobs/infer",
        json={
            "commandId": "predict-alpha-v7",
            "parameters": {
                "model_dir": "../../outside",
                "feature_dataset": "runtime/data/missing.parquet",
                "output": "runtime/predictions/test.parquet",
            },
        },
    )
    assert escaped.status_code == 422

    missing = request(
        app,
        "POST",
        "/api/jobs/infer",
        json={"commandId": "predict-alpha-v7", "parameters": {}},
    )
    assert missing.status_code == 422
    assert "missing required parameters" in missing.json()["detail"]


def test_empty_runtime_api(empty_quant_ui_settings) -> None:
    app = create_app(empty_quant_ui_settings)
    for url in ("/api/backtests", "/api/models", "/api/selection/runs", "/api/do-t/sources"):
        result = request(app, "GET", url)
        assert result.status_code == 200
        assert result.json()["status"] == "empty"


def test_log_endpoint_returns_lines_without_nested_parser_envelope(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    index = request(app, "GET", "/api/system/runtime-index?kind=log").json()
    artifact_id = index["data"]["items"][0]["id"]

    result = request(app, "GET", f"/api/system/logs?artifactId={artifact_id}").json()

    assert result["status"] == "ready"
    assert result["data"] == ["first line", "last line"]
    assert result["provenance"]["parser"] == "log"


def test_runtime_index_and_backtest_expose_verified_trust_contract(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)

    index = request(
        app,
        "GET",
        "/api/system/runtime-index",
        params={"query": "metrics.json", "refresh": True},
    ).json()
    artifact = next(item for item in index["data"]["items"] if item["name"] == "metrics.json")

    assert artifact["trustClass"] == "production_ready"
    assert artifact["validationStatus"] == "verified"
    assert artifact["schemaVersion"] == "quantagent.backtest.metrics.1"
    assert artifact["sourceTime"] == "2026-01-06T00:00:00+00:00"
    assert "production_display" in artifact["capabilities"]

    catalog = request(app, "GET", "/api/system/runtime-catalog").json()
    assert catalog["status"] == "ready"
    assert catalog["data"]["summary"]["runCount"] >= 1
    assert catalog["data"]["summary"]["byTrust"]["production_ready"] >= 1

    lineage = request(
        app,
        "GET",
        f"/api/system/runtime-index/{artifact['id']}/lineage",
    ).json()
    assert lineage["status"] == "ready"
    assert lineage["data"]["upstream"][0]["artifact"]["name"] == "nav.csv"

    backtest = request(app, "GET", "/api/backtests").json()["data"][0]
    assert backtest["trustClass"] == "production_ready"
    assert backtest["validationStatus"] == "verified"
    assert backtest["capabilities"]["productionDisplay"] is True


def test_api_reads_runtime_outside_repository(
    quant_ui_settings,
    tmp_path: Path,
) -> None:
    external_runtime = tmp_path / "external-runtime"
    shutil.copytree(quant_ui_settings.runtime_root, external_runtime)
    project_root = tmp_path / "checkout-without-runtime"
    project_root.mkdir()
    settings = ApiSettings(
        project_root=project_root,
        runtime_root=external_runtime,
        cache_root=external_runtime / "cache" / "quant_ui_external",
        jobs_root=external_runtime / "jobs" / "quant_ui_external",
        index_ttl_seconds=0,
    ).ensure()

    app = create_app(settings)
    backtests = request(app, "GET", "/api/backtests").json()
    runtime_index = request(
        app,
        "GET",
        "/api/system/runtime-index",
        params={"query": "metrics.json", "refresh": True},
    ).json()

    assert backtests["status"] == "ready"
    assert backtests["data"][0]["path"].startswith("runtime/")
    artifact = next(item for item in runtime_index["data"]["items"] if item["name"] == "metrics.json")
    assert artifact["path"].startswith("runtime/")
    assert artifact["validationStatus"] == "verified"
