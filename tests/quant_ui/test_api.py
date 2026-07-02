from __future__ import annotations

import asyncio

import httpx

from services.quant_api.app import create_app


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
