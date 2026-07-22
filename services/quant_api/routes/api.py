from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from services.quant_api.adapters.utils import page_slice
from services.quant_api.config import safe_project_path
from services.quant_api.runtime_indexer.parsers import parser_for
from services.quant_api.schemas.models import CleanupRequest, JobRequest


router = APIRouter(prefix="/api")


def services(request: Request):
    return request.app.state.services


def response(data: Any, *, issues: list[dict] | None = None, status: str | None = None, provenance: dict | None = None) -> dict:
    resolved_status = status or ("empty" if data in (None, [], {}) else "ready")
    payload = {"status": resolved_status, "data": data, "issues": issues or []}
    if provenance is not None:
        payload["provenance"] = provenance
    return payload



@router.get("/data/providers")
async def data_providers(request: Request) -> dict:
    data = services(request).data_manager.overview()
    return response(data, status="ready")


@router.get("/data/quarantine")
async def data_quarantine(request: Request) -> dict:
    data = services(request).data_manager.quarantine_files()
    return response(data, status="ready" if data else "empty")


@router.get("/data/coverage")
async def data_coverage(
    request: Request,
    path: str,
    date_column: str = Query("trade_date", alias="dateColumn"),
    symbol_column: str = Query("symbol", alias="symbolColumn"),
    deep: bool = False,
) -> dict:
    try:
        data = services(request).data_manager.inspect_dataset(
            path,
            date_column=date_column,
            symbol_column=symbol_column,
            deep=deep,
        )
        status = "partial" if data["duplicateKeys"] or data["missingBusinessDayCount"] else "ready"
        return response(data, status=status)
    except (OSError, ValueError) as exc:
        raise HTTPException(422, str(exc))


@router.get("/system/overview")
async def system_overview(request: Request) -> dict:
    svc = services(request)
    backtests = svc.backtests.list()
    models = svc.models.list()
    selections = svc.selections.list()
    latest_backtest = next(
        (
            item for item in backtests
            if item.get("capabilities", {}).get("equity")
            and item.get("capabilities", {}).get("trades")
        ),
        next(
            (
                item for item in backtests
                if item.get("capabilities", {}).get("equity")
                or item.get("capabilities", {}).get("trades")
            ),
            backtests[0] if backtests else None,
        ),
    )
    latest_model = models[0] if models else None
    latest_selection = selections[0] if selections else None
    trades = (
        svc.backtests.trades(latest_backtest["id"], page=1, page_size=1_000)["items"]
        if latest_backtest else []
    )
    buy_count = sum(item["action"] == "BUY" and item.get("success") for item in trades)
    sell_count = sum(item["action"] == "SELL" and item.get("success") for item in trades)
    do_t_sources = svc.do_t.list_sources()
    risk = svc.risk.overview(latest_backtest["id"] if latest_backtest else None)
    data = {
        "modelStatus": "ready" if latest_model else "unavailable",
        "latestModel": latest_model,
        "latestBacktest": latest_backtest,
        "latestSelection": latest_selection,
        "stockPoolCount": latest_selection.get("finalCount") if latest_selection else None,
        "candidateCount": latest_selection.get("candidateCount") if latest_selection else None,
        "signalCount": len(trades),
        "buySignalCount": buy_count,
        "sellSignalCount": sell_count,
        "doTSignalCount": (do_t_sources[0].get("metrics", {}).get("n_legs") if do_t_sources else None),
        "riskStatus": "warning" if risk.get("eventCounts") else "normal",
        "risk": risk,
        "runtime": svc.indexer.stats(),
    }
    return response(data, status="ready" if latest_backtest or latest_model else "partial")


@router.get("/system/runtime-index")
async def runtime_index(
    request: Request,
    kind: str | None = None,
    query: str | None = None,
    extension: str | None = None,
    run_id: str | None = Query(None, alias="runId"),
    horizon: str | None = None,
    modified_after: str | None = Query(None, alias="modifiedAfter"),
    modified_before: str | None = Query(None, alias="modifiedBefore"),
    strategy: str | None = None,
    model: str | None = None,
    symbol: str | None = None,
    trust_class: str | None = Query(None, alias="trustClass"),
    validation_status: str | None = Query(None, alias="validationStatus"),
    freshness_status: str | None = Query(None, alias="freshnessStatus"),
    capability: str | None = None,
    sort_by: str = Query("modifiedAt", alias="sortBy"),
    sort_direction: str = Query("desc", alias="sortDirection"),
    page: int = 1,
    page_size: int = Query(100, alias="pageSize", le=1_000),
    refresh: bool = False,
) -> dict:
    svc = services(request)
    if refresh:
        svc.indexer.scan(force=True)
    items = svc.indexer.filter(
        kind=kind,
        query=query,
        extension=extension,
        run_id=run_id,
        horizon=horizon,
        modified_after=modified_after,
        modified_before=modified_before,
        strategy=strategy,
        model=model,
        symbol=symbol,
        trust_class=trust_class,
        validation_status=validation_status,
        freshness_status=freshness_status,
        capability=capability,
        sort_by=sort_by,
        sort_direction=sort_direction,
    )
    return response(page_slice(items, page, page_size), status="ready" if items else "empty")


@router.get("/system/runtime-catalog")
async def runtime_catalog(request: Request, refresh: bool = False) -> dict:
    svc = services(request)
    if refresh:
        svc.indexer.scan(force=True)
    data = svc.indexer.catalog()
    return response(data, status="ready" if data["summary"]["artifactCount"] else "empty")


@router.get("/system/runtime-index/{artifact_id}/lineage")
async def runtime_lineage(request: Request, artifact_id: str) -> dict:
    data = services(request).indexer.lineage(artifact_id)
    if data is None:
        raise HTTPException(404, "artifact not found")
    status = "ready" if data["status"] == "complete" else "partial"
    return response(data, status=status, issues=data["issues"])


@router.get("/system/runtime-index/{artifact_id}/preview")
async def runtime_preview(request: Request, artifact_id: str, limit: int = Query(100, le=1_000)) -> dict:
    svc = services(request)
    artifact = svc.indexer.get(artifact_id)
    if artifact is None:
        raise HTTPException(404, "artifact not found")
    path = safe_project_path(svc.settings, artifact["path"])
    parser_name, parser = parser_for(path)
    try:
        preview = parser(path, limit)
        return response(
            preview,
            status=preview.get("status", "ready"),
            provenance={"sourcePath": artifact["path"], "parser": parser_name},
        )
    except Exception as exc:
        return response(
            None,
            status="error",
            issues=[{"code": "preview_error", "message": str(exc), "path": artifact["path"], "recoverable": True}],
        )


@router.get("/system/logs")
async def system_logs(
    request: Request,
    artifact_id: str | None = Query(None, alias="artifactId"),
    query: str | None = None,
    limit: int = Query(200, le=1_000),
) -> dict:
    svc = services(request)
    if artifact_id:
        artifact = svc.indexer.get(artifact_id)
        if artifact is None or artifact["kind"] != "log":
            raise HTTPException(404, "log artifact not found")
        path = safe_project_path(svc.settings, artifact["path"])
        _, parser = parser_for(path)
        parsed = parser(path, limit)
        return response(
            parsed.get("data"),
            status=parsed.get("status", "ready"),
            provenance={"sourcePath": artifact["path"], "parser": "log"},
        )
    logs = svc.indexer.filter(kind="log", query=query)
    return response(logs[:limit], status="ready" if logs else "empty")


@router.get("/system/runtime-cleanup")
async def runtime_cleanup_analysis(request: Request) -> dict:
    data = services(request).cleanup.analyze()
    return response(data, status="ready" if data["candidates"] else "empty")


@router.post("/system/runtime-cleanup")
async def runtime_cleanup_execute(request: Request, body: CleanupRequest) -> dict:
    try:
        data = services(request).cleanup.execute(body.candidate_ids, body.confirmation)
        services(request).indexer.invalidate()
        return response(data, status="partial" if data["errors"] else "ready")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("/backtests")
async def list_backtests(request: Request) -> dict:
    items = services(request).backtests.list()
    return response(items, status="ready" if items else "empty")


@router.get("/backtests/{backtest_id}")
async def get_backtest(request: Request, backtest_id: str) -> dict:
    item = services(request).backtests.get(backtest_id)
    if item is None:
        raise HTTPException(404, "backtest not found")
    return response(item)


@router.get("/backtests/{backtest_id}/equity")
async def backtest_equity(request: Request, backtest_id: str) -> dict:
    return _backtest_call(request, backtest_id, "equity")


@router.get("/backtests/{backtest_id}/drawdown")
async def backtest_drawdown(request: Request, backtest_id: str) -> dict:
    payload = _backtest_call(request, backtest_id, "equity")
    if payload["status"] == "error":
        return payload
    return response([
        {"datetime": item["datetime"], "drawdown": item["drawdown"]}
        for item in payload["data"]
    ], status=payload["status"])


@router.get("/backtests/{backtest_id}/trades")
async def backtest_trades(
    request: Request,
    backtest_id: str,
    symbol: str | None = None,
    page: int = 1,
    page_size: int = Query(100, alias="pageSize", le=1_000),
) -> dict:
    try:
        data = services(request).backtests.trades(backtest_id, symbol=symbol, page=page, page_size=page_size)
        return response(
            data,
            status="ready" if data["items"] else "empty",
            issues=data.get("issues", []),
        )
    except KeyError:
        raise HTTPException(404, "backtest not found")


@router.get("/backtests/{backtest_id}/positions")
async def backtest_positions(request: Request, backtest_id: str, symbol: str | None = None) -> dict:
    try:
        data = services(request).backtests.positions(backtest_id, symbol=symbol)
        return response(data, status="ready" if data else "empty")
    except KeyError:
        raise HTTPException(404, "backtest not found")


@router.get("/backtests/{backtest_id}/risk")
async def backtest_risk(request: Request, backtest_id: str, page: int = 1, page_size: int = Query(100, alias="pageSize", le=1_000)) -> dict:
    try:
        data = services(request).backtests.risk_events(backtest_id, page=page, page_size=page_size)
        return response(data, status="ready" if data["items"] else "empty")
    except KeyError:
        raise HTTPException(404, "backtest not found")


@router.get("/backtests/{backtest_id}/stocks/{symbol}")
async def stock_replay(request: Request, backtest_id: str, symbol: str) -> dict:
    try:
        data = services(request).backtests.stock_replay(backtest_id, symbol)
        return response(data, status="ready" if data["availability"]["bars"] else "partial")
    except KeyError:
        raise HTTPException(404, "backtest not found")


@router.get("/backtests/{backtest_id}/stocks/{symbol}/kline")
async def stock_kline(
    request: Request,
    backtest_id: str,
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    limit: int = Query(2_000, le=10_000),
) -> dict:
    if services(request).backtests.get(backtest_id) is None:
        raise HTTPException(404, "backtest not found")
    data = services(request).backtests.kline(symbol, start=start, end=end, limit=limit)
    return response(data, status="ready" if data["bars"] else "empty")


@router.get("/backtests/{backtest_id}/stocks/{symbol}/signals")
async def stock_signals(request: Request, backtest_id: str, symbol: str) -> dict:
    try:
        data = services(request).backtests.signals(backtest_id, symbol)
        return response(data, status="ready" if data else "empty")
    except KeyError:
        raise HTTPException(404, "backtest not found")


@router.get("/backtests/{backtest_id}/stocks/{symbol}/trades")
async def stock_trades(request: Request, backtest_id: str, symbol: str, page: int = 1, page_size: int = Query(100, alias="pageSize", le=1_000)) -> dict:
    return await backtest_trades(request, backtest_id, symbol, page, page_size)


@router.get("/backtests/{backtest_id}/stocks/{symbol}/t-analysis")
async def stock_t_analysis(request: Request, backtest_id: str, symbol: str, source_id: str | None = Query(None, alias="sourceId")) -> dict:
    if services(request).backtests.get(backtest_id) is None:
        raise HTTPException(404, "backtest not found")
    data = services(request).do_t.analyze(source_id=source_id, symbol=symbol)
    return response(
        data,
        status="ready" if data["pairs"] else "empty",
        issues=[] if data["pairs"] else [{
            "code": "no_linked_do_t_pairs",
            "message": "该股票在当前可用 Do-T artifact 中没有可映射交易对。",
            "recoverable": True,
        }],
    )


@router.get("/factors")
async def list_factors(request: Request, query: str | None = None) -> dict:
    data = services(request).factors.list(query)
    return response(data, status="ready" if data else "empty")


@router.get("/factors/{factor_name}")
async def get_factor(request: Request, factor_name: str) -> dict:
    data = services(request).factors.get(factor_name)
    if data is None:
        raise HTTPException(404, "factor not found")
    return response(data)


@router.get("/factors/{factor_name}/explanation")
async def factor_explanation(request: Request, factor_name: str) -> dict:
    data = services(request).factors.explanation(factor_name)
    if data is None:
        raise HTTPException(404, "factor not found")
    return response(data)


@router.get("/factors/{factor_name}/backtest")
async def factor_backtest(request: Request, factor_name: str) -> dict:
    if services(request).factors.get(factor_name) is None:
        raise HTTPException(404, "factor not found")
    data = services(request).factors.backtest(factor_name)
    return response(data, status="ready" if data["availability"]["summaryMetrics"] else "partial")


@router.get("/factors/{factor_name}/stocks/{symbol}/signals")
async def factor_stock_signals(request: Request, factor_name: str, symbol: str) -> dict:
    if services(request).factors.get(factor_name) is None:
        raise HTTPException(404, "factor not found")
    return response([], status="empty", issues=[{
        "code": "independent_factor_signals_missing",
        "message": "未发现该因子的独立 signal artifact。",
        "recoverable": True,
    }])


@router.get("/factors/{factor_name}/stocks/{symbol}/trades")
async def factor_stock_trades(request: Request, factor_name: str, symbol: str) -> dict:
    if services(request).factors.get(factor_name) is None:
        raise HTTPException(404, "factor not found")
    return response([], status="empty", issues=[{
        "code": "independent_factor_trades_missing",
        "message": "未发现该因子的独立 trade artifact；未使用 multi-factor trades 冒充。",
        "recoverable": True,
    }])


@router.get("/factors/{factor_name}/ic")
async def factor_ic(request: Request, factor_name: str) -> dict:
    if services(request).factors.get(factor_name) is None:
        raise HTTPException(404, "factor not found")
    return response(services(request).factors.ic(factor_name))


@router.get("/factors/{factor_name}/quantile-returns")
async def factor_quantiles(request: Request, factor_name: str) -> dict:
    if services(request).factors.get(factor_name) is None:
        raise HTTPException(404, "factor not found")
    data = services(request).factors.quantile_returns(factor_name)
    return response(data, status="ready" if data else "empty")


@router.get("/selection/runs")
async def selection_runs(request: Request) -> dict:
    data = services(request).selections.list()
    return response(data, status="ready" if data else "empty")


@router.get("/selection/runs/{run_id}")
async def selection_run(request: Request, run_id: str) -> dict:
    data = services(request).selections.get(run_id)
    if data is None:
        raise HTTPException(404, "selection run not found")
    return response(data)


@router.get("/selection/runs/{run_id}/funnel")
async def selection_funnel(request: Request, run_id: str) -> dict:
    try:
        return response(services(request).selections.funnel(run_id))
    except KeyError:
        raise HTTPException(404, "selection run not found")


@router.get("/selection/runs/{run_id}/ranking")
async def selection_ranking(request: Request, run_id: str, limit: int = Query(500, le=1_000)) -> dict:
    try:
        data = services(request).selections.ranking(run_id, limit)
        return response(data, status="ready" if data else "empty")
    except KeyError:
        raise HTTPException(404, "selection run not found")


@router.get("/selection/runs/{run_id}/stocks/{symbol}/decision-chain")
async def selection_decision_chain(request: Request, run_id: str, symbol: str) -> dict:
    data = services(request).selections.decision_chain(run_id, symbol)
    if data is None:
        return response(
            {"gates": [], "finalDecision": None, "issues": [{
                "code": "selection_stock_not_found",
                "message": "该股票不在此 persisted selection run 中。",
                "recoverable": True,
            }]},
            status="empty",
            issues=[{
                "code": "selection_stock_not_found",
                "message": "该股票不在此 persisted selection run 中。",
                "recoverable": True,
            }],
        )
    return response(data, status="partial" if data.get("issues") else "ready")


@router.get("/models")
async def list_models(request: Request) -> dict:
    data = services(request).models.list()
    return response(data, status="ready" if data else "empty")


@router.get("/models/compare")
async def compare_models(request: Request, ids: str = Query("")) -> dict:
    model_ids = [value for value in ids.split(",") if value]
    if not model_ids:
        return response({"models": [], "metricKeys": []}, status="empty")
    try:
        return response(services(request).models.compare(model_ids))
    except KeyError as exc:
        raise HTTPException(404, f"model not found: {exc.args[0]}")


@router.get("/models/{model_id}")
async def get_model(request: Request, model_id: str) -> dict:
    data = services(request).models.get(model_id)
    if data is None:
        raise HTTPException(404, "model not found")
    return response(data)


@router.get("/models/{model_id}/observability")
async def model_observability(request: Request, model_id: str) -> dict:
    try:
        return response(services(request).models.observability(model_id))
    except KeyError:
        raise HTTPException(404, "model not found")


@router.get("/models/{model_id}/training-metrics")
async def model_training_metrics(request: Request, model_id: str) -> dict:
    try:
        data = services(request).models.training_metrics(model_id)
        return response(data, status="ready" if data else "empty")
    except KeyError:
        raise HTTPException(404, "model not found")


@router.get("/models/{model_id}/feature-importance")
async def model_feature_importance(request: Request, model_id: str) -> dict:
    try:
        data = services(request).models.feature_importance(model_id)
        return response(data, status="ready" if data else "empty")
    except KeyError:
        raise HTTPException(404, "model not found")


@router.get("/models/{model_id}/predictions")
async def model_predictions(request: Request, model_id: str, symbol: str | None = None, limit: int = Query(2_000, le=10_000)) -> dict:
    try:
        data = services(request).models.predictions(model_id, symbol=symbol, limit=limit)
        return response(data, status="ready" if data else "empty")
    except KeyError:
        raise HTTPException(404, "model not found")


@router.get("/models/{model_id}/stocks/{symbol}/prediction-history")
async def model_stock_predictions(request: Request, model_id: str, symbol: str, limit: int = Query(2_000, le=10_000)) -> dict:
    return await model_predictions(request, model_id, symbol, limit)


@router.get("/risk/overview")
async def risk_overview(request: Request, backtest_id: str | None = Query(None, alias="backtestId")) -> dict:
    return response(services(request).risk.overview(backtest_id))


@router.get("/risk/events")
async def risk_events(request: Request, backtest_id: str | None = Query(None, alias="backtestId"), page: int = 1, page_size: int = Query(100, alias="pageSize", le=1_000)) -> dict:
    data = services(request).risk.events(backtest_id, page, page_size)
    return response(data, status="ready" if data["items"] else "empty")


@router.get("/risk/stocks")
async def risk_stocks(request: Request, backtest_id: str | None = Query(None, alias="backtestId")) -> dict:
    data = services(request).risk.stocks(backtest_id)
    return response(data, status="ready" if data else "empty")


@router.get("/risk/rules")
async def risk_rules(request: Request) -> dict:
    return response(services(request).risk.rules())


@router.get("/risk/backtests/{backtest_id}")
async def risk_backtest(request: Request, backtest_id: str) -> dict:
    if services(request).backtests.get(backtest_id) is None:
        raise HTTPException(404, "backtest not found")
    return response({
        "overview": services(request).risk.overview(backtest_id),
        "events": services(request).risk.events(backtest_id, 1, 200),
        "stocks": services(request).risk.stocks(backtest_id),
    })


@router.get("/do-t/sources")
async def do_t_sources(request: Request) -> dict:
    data = services(request).do_t.list_sources()
    return response(data, status="ready" if data else "empty")


@router.get("/do-t/analysis")
async def do_t_analysis(request: Request, source_id: str | None = Query(None, alias="sourceId"), symbol: str | None = None, limit: int = Query(500, le=1_000)) -> dict:
    try:
        data = services(request).do_t.analyze(source_id, symbol, limit)
        return response(data, status="ready" if data["pairs"] else "empty")
    except KeyError:
        raise HTTPException(404, "Do-T source not found")



@router.post("/jobs/data")
async def create_data_job(request: Request, body: JobRequest) -> dict:
    return _create_job(request, "data", body)


@router.post("/jobs/backtest")
async def create_backtest_job(request: Request, body: JobRequest) -> dict:
    return _create_job(request, "backtest", body)


@router.post("/jobs/train")
async def create_train_job(request: Request, body: JobRequest) -> dict:
    return _create_job(request, "train", body)


@router.post("/jobs/infer")
async def create_infer_job(request: Request, body: JobRequest) -> dict:
    return _create_job(request, "infer", body)


@router.get("/jobs")
async def list_jobs(request: Request) -> dict:
    data = services(request).jobs.list()
    return response(data, status="ready" if data else "empty")



@router.post("/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str) -> dict:
    try:
        return response(services(request).jobs.cancel(job_id))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> dict:
    data = services(request).jobs.get(job_id)
    if data is None:
        raise HTTPException(404, "job not found")
    return response(data)


@router.get("/jobs/{job_id}/logs")
async def job_logs(request: Request, job_id: str, limit: int = Query(500, le=10_000)) -> dict:
    if services(request).jobs.get(job_id) is None:
        raise HTTPException(404, "job not found")
    data = services(request).jobs.logs(job_id, limit)
    return response(data, status="ready" if data else "empty")


@router.get("/jobs/{job_id}/stream")
async def job_stream(request: Request, job_id: str):
    if services(request).jobs.get(job_id) is None:
        raise HTTPException(404, "job not found")
    return StreamingResponse(services(request).jobs.stream(job_id), media_type="text/event-stream")


def _backtest_call(request: Request, backtest_id: str, method: str) -> dict:
    try:
        data = getattr(services(request).backtests, method)(backtest_id)
        return response(data, status="ready" if data else "empty")
    except KeyError:
        raise HTTPException(404, "backtest not found")


def _create_job(request: Request, job_type: str, body: JobRequest) -> dict:
    try:
        data = services(request).jobs.submit(job_type, body.command_id, body.parameters)
        return response(data)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
