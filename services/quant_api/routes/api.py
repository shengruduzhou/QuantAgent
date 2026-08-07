from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from services.quant_api.adapters.utils import page_slice
from services.quant_api.config import safe_project_path
from services.quant_api.runtime_indexer.parsers import parser_for
from services.quant_api.schemas.models import CleanupRequest, CouncilOverrideRequest, FactorReviewRequest, GlobalSearchResult, JobRequest, SearchEntity, SearchGroup
from services.quant_api.schemas.paper import PaperOrderCancellation, PaperOrderSubmission
from services.quant_api.schemas.strategy import ConnectionRequest, StrategyDraft
from services.quant_api.services.paper_orders import SubmissionRejected, WriterLockUnavailable
from quantagent.execution.order_manager import IdempotencyConflict
from quantagent.safety.operating_mode import LiveTradingRejected


router = APIRouter(prefix="/api")


def services(request: Request):
    return request.app.state.services


def response(data: Any, *, issues: list[dict] | None = None, status: str | None = None, provenance: dict | None = None) -> dict:
    resolved_status = status or ("empty" if data in (None, [], {}) else "ready")
    payload = {"status": resolved_status, "data": data, "issues": issues or []}
    if provenance is not None:
        payload["provenance"] = provenance
    return payload


@router.get("/search")
def global_search(request: Request, q: str = Query(min_length=2, max_length=120), limit: int = Query(8, ge=1, le=20)) -> dict:
    svc = services(request)
    needle = q.strip().lower()
    groups: list[SearchGroup] = []

    def matches(*values: object) -> bool:
        return needle in " ".join(str(value or "") for value in values).lower()

    factors = [
        SearchEntity(
            id=item["name"], kind="factor", label=item.get("displayName") or item["name"],
            detail=f"{item.get('category') or 'factor'} · {item.get('lifecycle') or 'unclassified'}",
            path=f"/factors?{urlencode({'factor': item['name']})}", status="ready", source="FactorAdapter",
        )
        for item in svc.factors.list()
        if matches(item.get("name"), item.get("displayName"), item.get("description"))
    ][:limit]
    if factors:
        groups.append(SearchGroup(type="factor", label="Factors", items=factors))

    models = [
        SearchEntity(
            id=item["id"], kind="model", label=item.get("version") or item["id"],
            detail=f"{item.get('modelFamily') or item.get('modelType') or 'model'} · {item.get('verdict') or item.get('status')}",
            path=f"/models?{urlencode({'modelId': item['id']})}", status=item.get("status", "ready"), source="ModelAdapter",
        )
        for item in svc.models.list()
        if matches(item.get("id"), item.get("version"), item.get("modelType"), item.get("modelFamily"), item.get("verdict"))
    ][:limit]
    if models:
        groups.append(SearchGroup(type="model", label="Models", items=models))

    backtests = svc.backtests.list()
    backtest_entities = [
        SearchEntity(
            id=item["id"], kind="backtest", label=item.get("name") or item["id"],
            detail=f"{item.get('horizon') or 'research'} · {item.get('startDate') or '—'} → {item.get('endDate') or '—'}",
            path=f"/backtests?{urlencode({'run': item['id']})}", status=item.get("status", "ready"), source="BacktestAdapter",
        )
        for item in backtests
        if matches(item.get("id"), item.get("name"), item.get("horizon"), item.get("modelVersion"))
    ][:limit]
    if backtest_entities:
        groups.append(SearchGroup(type="backtest", label="Backtests", items=backtest_entities))

    run_entities = [
        SearchEntity(
            id=item["id"], kind="run", label=item["id"],
            detail=f"selection · {item.get('asOfDate') or 'unknown date'} · {item.get('finalCount') or 0} names",
            path=f"/selection?{urlencode({'run': item['id']})}", status=item.get("status", "ready"), source="SelectionAdapter",
        )
        for item in svc.selections.list()
        if matches(item.get("id"), item.get("asOfDate"), item.get("path"))
    ][:limit]
    if run_entities:
        groups.append(SearchGroup(type="run", label="Runs", items=run_entities))

    artifacts = [
        SearchEntity(
            id=item["id"], kind="artifact", label=item["name"],
            detail=f"{item['kind']} · {item.get('runId') or 'no run'} · {item.get('trustClass') or 'unclassified'}",
            path=f"/runtime?{urlencode({'artifact': item['id'], 'query': item['path']})}",
            status=item.get("status", "ready"), source="RuntimeIndexer",
        )
        for item in svc.indexer.filter(query=q)[:limit]
    ]
    if artifacts:
        groups.append(SearchGroup(type="artifact", label="Artifacts", items=artifacts))

    stock_entities: list[SearchEntity] = []
    seen_symbols: set[str] = set()
    for backtest in backtests[:3]:
        if len(stock_entities) >= limit:
            break
        try:
            trades = svc.backtests.trades(backtest["id"], page=1, page_size=1_000)["items"]
        except (KeyError, OSError, ValueError):
            continue
        for trade in trades:
            symbol = str(trade.get("symbol") or "")
            name = str(trade.get("name") or "")
            if symbol in seen_symbols or not matches(symbol, name):
                continue
            seen_symbols.add(symbol)
            stock_entities.append(SearchEntity(
                id=symbol, kind="stock", label=f"{symbol} {name}".strip(),
                detail=f"persisted trade · {backtest.get('name') or backtest['id']}",
                path=f"/stock-replay?{urlencode({'symbol': symbol, 'backtestId': backtest['id']})}",
                status="ready", source="BacktestAdapter",
            ))
            if len(stock_entities) >= limit:
                break
    if stock_entities:
        groups.insert(0, SearchGroup(type="stock", label="Stocks", items=stock_entities))

    data = GlobalSearchResult(query=q, groups=groups).model_dump(by_alias=True)
    return response(data, status="ready" if groups else "empty")



@router.get("/data/providers")
def data_providers(request: Request) -> dict:
    data = services(request).data_manager.overview()
    return response(data, status="ready")


@router.get("/data/quarantine")
def data_quarantine(request: Request) -> dict:
    data = services(request).data_manager.quarantine_files()
    return response(data, status="ready" if data else "empty")


@router.get("/data/coverage")
def data_coverage(
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


@router.get("/governance/status")
def governance_status(request: Request) -> dict:
    """Read-only operational governance surface (H-031): shadow/S4/U0/lineage.

    Returns existence- and gate-level fields only; never candidate performance.
    """
    data = services(request).governance.status()
    ready = data["shadow"]["status"] == "ready" or data["u0"]["status"] == "ready"
    return response(data, status="ready" if ready else "partial")


@router.get("/system/overview")
def system_overview(request: Request) -> dict:
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


@router.get("/system/resources")
def system_resources() -> dict:
    """Return bounded host telemetry; never expose arbitrary shell execution."""

    payload: dict[str, Any] = {
        "cpuPercent": None,
        "memoryPercent": None,
        "memoryUsedBytes": None,
        "memoryTotalBytes": None,
        "gpus": [],
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        payload.update({
            "cpuPercent": float(psutil.cpu_percent(interval=None)),
            "memoryPercent": float(memory.percent),
            "memoryUsedBytes": int(memory.used),
            "memoryTotalBytes": int(memory.total),
        })
    except (ImportError, OSError):
        pass
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                fields = [item.strip() for item in line.split(",")]
                if len(fields) != 7:
                    continue
                payload["gpus"].append({
                    "index": int(fields[0]),
                    "name": fields[1],
                    "utilizationPercent": float(fields[2]),
                    "memoryUsedMiB": float(fields[3]),
                    "memoryTotalMiB": float(fields[4]),
                    "temperatureC": float(fields[5]),
                    "powerDrawW": float(fields[6]),
                })
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return response(
        payload,
        status="ready" if payload["gpus"] or payload["cpuPercent"] is not None else "unavailable",
    )


@router.get("/system/runtime-index")
def runtime_index(
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
def runtime_catalog(request: Request, refresh: bool = False) -> dict:
    svc = services(request)
    if refresh:
        svc.indexer.scan(force=True)
    data = svc.indexer.catalog()
    return response(data, status="ready" if data["summary"]["artifactCount"] else "empty")


@router.get("/system/runtime-index/{artifact_id}/lineage")
def runtime_lineage(request: Request, artifact_id: str) -> dict:
    data = services(request).indexer.lineage(artifact_id)
    if data is None:
        raise HTTPException(404, "artifact not found")
    status = "ready" if data["status"] == "complete" else "partial"
    return response(data, status=status, issues=data["issues"])


@router.get("/system/runtime-index/{artifact_id}/preview")
def runtime_preview(request: Request, artifact_id: str, limit: int = Query(100, le=1_000)) -> dict:
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
def system_logs(
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
def runtime_cleanup_analysis(request: Request) -> dict:
    data = services(request).cleanup.analyze()
    return response(data, status="ready" if data["candidates"] else "empty")


@router.post("/system/runtime-cleanup")
def runtime_cleanup_execute(request: Request, body: CleanupRequest) -> dict:
    try:
        data = services(request).cleanup.execute(body.candidate_ids, body.confirmation)
        services(request).indexer.invalidate()
        return response(data, status="partial" if data["errors"] else "ready")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("/backtests")
def list_backtests(request: Request) -> dict:
    items = services(request).backtests.list()
    return response(items, status="ready" if items else "empty")


@router.get("/backtests/{backtest_id}")
def get_backtest(request: Request, backtest_id: str) -> dict:
    item = services(request).backtests.get(backtest_id)
    if item is None:
        raise HTTPException(404, "backtest not found")
    return response(item)


@router.get("/backtests/{backtest_id}/equity")
def backtest_equity(request: Request, backtest_id: str) -> dict:
    return _backtest_call(request, backtest_id, "equity")


@router.get("/backtests/{backtest_id}/drawdown")
def backtest_drawdown(request: Request, backtest_id: str) -> dict:
    payload = _backtest_call(request, backtest_id, "equity")
    if payload["status"] == "error":
        return payload
    return response([
        {"datetime": item["datetime"], "drawdown": item["drawdown"]}
        for item in payload["data"]
    ], status=payload["status"])


@router.get("/backtests/{backtest_id}/trades")
def backtest_trades(
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
def backtest_positions(request: Request, backtest_id: str, symbol: str | None = None) -> dict:
    try:
        data = services(request).backtests.positions(backtest_id, symbol=symbol)
        return response(data, status="ready" if data else "empty")
    except KeyError:
        raise HTTPException(404, "backtest not found")


@router.get("/backtests/{backtest_id}/risk")
def backtest_risk(request: Request, backtest_id: str, page: int = 1, page_size: int = Query(100, alias="pageSize", le=1_000)) -> dict:
    try:
        data = services(request).backtests.risk_events(backtest_id, page=page, page_size=page_size)
        return response(data, status="ready" if data["items"] else "empty")
    except KeyError:
        raise HTTPException(404, "backtest not found")


@router.get("/backtests/{backtest_id}/stocks/{symbol}")
def stock_replay(request: Request, backtest_id: str, symbol: str) -> dict:
    try:
        data = services(request).backtests.stock_replay(backtest_id, symbol)
        return response(data, status="ready" if data["availability"]["bars"] else "partial")
    except KeyError:
        raise HTTPException(404, "backtest not found")


@router.get("/backtests/{backtest_id}/stocks/{symbol}/kline")
def stock_kline(
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
def stock_signals(request: Request, backtest_id: str, symbol: str) -> dict:
    try:
        data = services(request).backtests.signals(backtest_id, symbol)
        return response(data, status="ready" if data else "empty")
    except KeyError:
        raise HTTPException(404, "backtest not found")


@router.get("/backtests/{backtest_id}/stocks/{symbol}/trades")
def stock_trades(request: Request, backtest_id: str, symbol: str, page: int = 1, page_size: int = Query(100, alias="pageSize", le=1_000)) -> dict:
    return backtest_trades(request, backtest_id, symbol, page, page_size)


@router.get("/backtests/{backtest_id}/stocks/{symbol}/t-analysis")
def stock_t_analysis(request: Request, backtest_id: str, symbol: str, source_id: str | None = Query(None, alias="sourceId")) -> dict:
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


@router.get("/fusion/runs")
def list_fusion_runs(request: Request) -> dict:
    data = services(request).fusion.list()
    return response(
        data,
        status="ready" if data else "empty",
        issues=[] if data else [{
            "code": "no_fusion_runs",
            "message": "尚无因子融合搜索产物。在因子融合工场配置并启动一次搜索后再回到此处。",
            "recoverable": True,
        }],
    )


@router.get("/fusion/runs/{run_id}")
def get_fusion_run(request: Request, run_id: str) -> dict:
    try:
        data = services(request).fusion.detail(run_id)
    except KeyError:
        raise HTTPException(404, "fusion run not found")
    return response(data, status="ready" if data["candidates"] else "empty")


@router.get("/fusion/runs/{run_id}/nav")
def get_fusion_nav(
    request: Request,
    run_id: str,
    limit: int = Query(4_000, le=20_000),
) -> dict:
    try:
        data = services(request).fusion.navs(run_id, limit)
    except KeyError:
        raise HTTPException(404, "fusion run not found")
    return response(data, status="ready" if data else "empty")


@router.get("/fusion/runs/{run_id}/compare")
def compare_fusion_candidates(
    request: Request,
    run_id: str,
    candidates: str = Query("", max_length=500),
) -> dict:
    ids = [item.strip() for item in candidates.split(",") if item.strip()]
    try:
        data = services(request).fusion.compare(run_id, ids)
    except KeyError as exc:
        raise HTTPException(404, f"candidate not found: {exc}")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return response(data)


@router.get("/council/roster")
def council_roster(request: Request) -> dict:
    return response(services(request).council.roster())


@router.get("/council/review/fusion/{run_id}")
def council_review_fusion(
    request: Request,
    run_id: str,
    candidate: str | None = None,
) -> dict:
    try:
        data = services(request).council.review_fusion_run(run_id, candidate)
    except KeyError as exc:
        raise HTTPException(404, f"subject not found: {exc}")
    return response(data)


@router.get("/council/review/run/{run_id}")
def council_review_strategy_run(request: Request, run_id: str) -> dict:
    """Role-scoped council review of a completed strategy-pipeline run."""
    svc = services(request)
    try:
        data = svc.council.review_strategy_run(
            run_id, svc.strategies.results, svc.strategies
        )
    except KeyError as exc:
        raise HTTPException(404, f"run not found: {exc}")
    return response(data)


@router.get("/council/overrides")
def council_overrides(
    request: Request,
    subjectType: str | None = None,
    subjectId: str | None = None,
) -> dict:
    data = services(request).council.overrides(
        subject_type=subjectType, subject_id=subjectId
    )
    return response(data, status="ready" if data else "empty")


@router.post("/council/overrides")
def create_council_override(request: Request, body: CouncilOverrideRequest) -> dict:
    try:
        data = services(request).council.record_override(
            subject_type=body.subject_type,
            subject_id=body.subject_id,
            role_id=body.role_id,
            verdict=body.verdict,
            reason=body.reason,
            author=body.author,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return response(data)


@router.get("/factors")
def list_factors(request: Request, query: str | None = None) -> dict:
    data = services(request).factors.list(query)
    return response(data, status="ready" if data else "empty")


@router.get("/factors/correlation")
def factor_correlation(
    request: Request,
    names: str = Query("", max_length=500),
) -> dict:
    data = services(request).factors.correlation(
        [item.strip() for item in names.split(",") if item.strip()]
    )
    return response(data, status="ready" if data.get("matrix") else "empty")


@router.get("/factors/{factor_name}")
def get_factor(request: Request, factor_name: str) -> dict:
    data = services(request).factors.get(factor_name)
    if data is None:
        raise HTTPException(404, "factor not found")
    return response(data)


@router.get("/factors/{factor_name}/explanation")
def factor_explanation(request: Request, factor_name: str) -> dict:
    data = services(request).factors.explanation(factor_name)
    if data is None:
        raise HTTPException(404, "factor not found")
    return response(data)


@router.get("/factors/{factor_name}/reviews")
def factor_reviews(request: Request, factor_name: str) -> dict:
    if services(request).factors.get(factor_name) is None:
        raise HTTPException(404, "factor not found")
    data = services(request).factors.reviews(factor_name)
    return response(data, status="ready" if data else "empty")


@router.post("/factors/{factor_name}/reviews")
def create_factor_review(request: Request, factor_name: str, body: FactorReviewRequest) -> dict:
    try:
        return response(services(request).factors.record_review(factor_name, body.action, body.note))
    except KeyError:
        raise HTTPException(404, "factor not found")


@router.get("/factors/{factor_name}/backtest")
def factor_backtest(request: Request, factor_name: str) -> dict:
    if services(request).factors.get(factor_name) is None:
        raise HTTPException(404, "factor not found")
    data = services(request).factors.backtest(factor_name)
    return response(data, status="ready" if data["availability"]["summaryMetrics"] else "partial")


@router.get("/factors/{factor_name}/stocks/{symbol}/signals")
def factor_stock_signals(request: Request, factor_name: str, symbol: str) -> dict:
    if services(request).factors.get(factor_name) is None:
        raise HTTPException(404, "factor not found")
    return response([], status="empty", issues=[{
        "code": "independent_factor_signals_missing",
        "message": "未发现该因子的独立 signal artifact。",
        "recoverable": True,
    }])


@router.get("/factors/{factor_name}/stocks/{symbol}/trades")
def factor_stock_trades(request: Request, factor_name: str, symbol: str) -> dict:
    if services(request).factors.get(factor_name) is None:
        raise HTTPException(404, "factor not found")
    return response([], status="empty", issues=[{
        "code": "independent_factor_trades_missing",
        "message": "未发现该因子的独立 trade artifact；未使用 multi-factor trades 冒充。",
        "recoverable": True,
    }])


@router.get("/factors/{factor_name}/ic")
def factor_ic(request: Request, factor_name: str) -> dict:
    if services(request).factors.get(factor_name) is None:
        raise HTTPException(404, "factor not found")
    return response(services(request).factors.ic(factor_name))


@router.get("/factors/{factor_name}/quantile-returns")
def factor_quantiles(request: Request, factor_name: str) -> dict:
    if services(request).factors.get(factor_name) is None:
        raise HTTPException(404, "factor not found")
    data = services(request).factors.quantile_returns(factor_name)
    return response(data, status="ready" if data else "empty")


@router.get("/selection/runs")
def selection_runs(request: Request) -> dict:
    data = services(request).selections.list()
    return response(data, status="ready" if data else "empty")


@router.get("/selection/runs/{run_id}")
def selection_run(request: Request, run_id: str) -> dict:
    data = services(request).selections.get(run_id)
    if data is None:
        raise HTTPException(404, "selection run not found")
    return response(data)


@router.get("/selection/runs/{run_id}/funnel")
def selection_funnel(request: Request, run_id: str) -> dict:
    try:
        return response(services(request).selections.funnel(run_id))
    except KeyError:
        raise HTTPException(404, "selection run not found")


@router.get("/selection/runs/{run_id}/ranking")
def selection_ranking(request: Request, run_id: str, limit: int = Query(500, le=1_000)) -> dict:
    try:
        data = services(request).selections.ranking(run_id, limit)
        return response(data, status="ready" if data else "empty")
    except KeyError:
        raise HTTPException(404, "selection run not found")


@router.get("/selection/runs/{run_id}/stocks/{symbol}/decision-chain")
def selection_decision_chain(request: Request, run_id: str, symbol: str) -> dict:
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
def list_models(request: Request) -> dict:
    data = services(request).models.list()
    return response(data, status="ready" if data else "empty")


@router.get("/models/compare")
def compare_models(request: Request, ids: str = Query("")) -> dict:
    model_ids = [value for value in ids.split(",") if value]
    if not model_ids:
        return response({"models": [], "metricKeys": []}, status="empty")
    try:
        return response(services(request).models.compare(model_ids))
    except KeyError as exc:
        raise HTTPException(404, f"model not found: {exc.args[0]}")


@router.get("/models/{model_id}")
def get_model(request: Request, model_id: str) -> dict:
    data = services(request).models.get(model_id)
    if data is None:
        raise HTTPException(404, "model not found")
    return response(data)


@router.get("/models/{model_id}/observability")
def model_observability(request: Request, model_id: str) -> dict:
    try:
        return response(services(request).models.observability(model_id))
    except KeyError:
        raise HTTPException(404, "model not found")


@router.get("/models/{model_id}/training-metrics")
def model_training_metrics(request: Request, model_id: str) -> dict:
    try:
        data = services(request).models.training_metrics(model_id)
        return response(data, status="ready" if data else "empty")
    except KeyError:
        raise HTTPException(404, "model not found")


@router.get("/models/{model_id}/feature-importance")
def model_feature_importance(request: Request, model_id: str) -> dict:
    try:
        data = services(request).models.feature_importance(model_id)
        return response(data, status="ready" if data else "empty")
    except KeyError:
        raise HTTPException(404, "model not found")


@router.get("/models/{model_id}/predictions")
def model_predictions(request: Request, model_id: str, symbol: str | None = None, limit: int = Query(2_000, le=10_000)) -> dict:
    try:
        data = services(request).models.predictions(model_id, symbol=symbol, limit=limit)
        return response(data, status="ready" if data else "empty")
    except KeyError:
        raise HTTPException(404, "model not found")


@router.get("/models/{model_id}/stocks/{symbol}/prediction-history")
def model_stock_predictions(request: Request, model_id: str, symbol: str, limit: int = Query(2_000, le=10_000)) -> dict:
    return model_predictions(request, model_id, symbol, limit)


@router.get("/risk/overview")
def risk_overview(request: Request, backtest_id: str | None = Query(None, alias="backtestId")) -> dict:
    return response(services(request).risk.overview(backtest_id))


@router.get("/risk/events")
def risk_events(request: Request, backtest_id: str | None = Query(None, alias="backtestId"), page: int = 1, page_size: int = Query(100, alias="pageSize", le=1_000)) -> dict:
    data = services(request).risk.events(backtest_id, page, page_size)
    return response(data, status="ready" if data["items"] else "empty")


@router.get("/risk/stocks")
def risk_stocks(request: Request, backtest_id: str | None = Query(None, alias="backtestId")) -> dict:
    data = services(request).risk.stocks(backtest_id)
    return response(data, status="ready" if data else "empty")


@router.get("/risk/rules")
def risk_rules(request: Request) -> dict:
    return response(services(request).risk.rules())


@router.get("/connections")
def list_connections(request: Request) -> dict:
    return response(services(request).connections.list())


@router.post("/connections/{provider_id}")
def connect_provider(request: Request, provider_id: str, body: ConnectionRequest) -> dict:
    try:
        return response(services(request).connections.connect(provider_id, body.credentials))
    except KeyError:
        raise HTTPException(404, "connector not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.delete("/connections/{provider_id}")
def disconnect_provider(request: Request, provider_id: str) -> dict:
    try:
        return response(services(request).connections.disconnect(provider_id))
    except KeyError:
        raise HTTPException(404, "connector not found")


@router.get("/strategies")
def list_strategies(request: Request, include_versions: bool = Query(False, alias="includeVersions")) -> dict:
    data = services(request).strategies.list(include_versions=include_versions)
    return response(data, status="ready" if data else "empty")


@router.get("/strategies/defaults")
def strategy_defaults(request: Request) -> dict:
    return response(services(request).strategies.defaults())


@router.get("/strategies/runs")
def list_strategy_runs(
    request: Request,
    strategy_id: str | None = Query(None, alias="strategyId"),
) -> dict:
    """Every launched run, with its live job state resolved alongside it."""
    svc = services(request)
    runs = svc.strategies.runs(strategy_id)
    enriched = [{**run, "job": svc.jobs.get(run.get("jobId"))} for run in runs]
    return response(enriched, status="ready" if enriched else "empty")


@router.get("/strategies/runs/compare")
def compare_strategy_runs(
    request: Request,
    runs: str = Query("", max_length=400),
) -> dict:
    """Align up to four runs on the same fields, read from their own artifacts."""
    ids = [item.strip() for item in runs.split(",") if item.strip()]
    if not ids:
        return response({"runs": [], "metrics": [], "gates": []}, status="empty")
    try:
        data = services(request).strategies.compare_runs(ids)
    except KeyError as exc:
        raise HTTPException(404, f"run not found: {exc.args[0]}")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return response(data)


@router.get("/strategies/runs/{run_id}")
def get_strategy_run(request: Request, run_id: str) -> dict:
    svc = services(request)
    data = svc.strategies.run_result(run_id)
    if data is None:
        raise HTTPException(404, "strategy run not found")
    data["job"] = svc.jobs.get(data.get("jobId"))
    result_status = data["result"]["status"]
    return response(
        data,
        status="ready" if result_status in {"complete", "rejected"} else "partial",
    )


@router.get("/strategies/{strategy_id}")
def get_strategy(
    request: Request,
    strategy_id: str,
    version: str | None = None,
) -> dict:
    svc = services(request)
    data = svc.strategies.get(strategy_id, version)
    if data is None:
        raise HTTPException(404, "strategy not found")
    data["runs"] = [
        {**run, "job": svc.jobs.get(run.get("jobId"))}
        for run in data.get("runs", [])
    ]
    return response(data)


@router.delete("/strategies/{strategy_id}")
def delete_strategy(
    request: Request,
    strategy_id: str,
    version: str | None = None,
    delete_outputs: bool = Query(False, alias="deleteOutputs"),
) -> dict:
    """Archive a strategy or one of its versions; outputs only on request."""
    try:
        data = services(request).strategies.delete(
            strategy_id, version=version, delete_outputs=delete_outputs
        )
    except KeyError:
        raise HTTPException(404, "strategy not found")
    except (OSError, ValueError) as exc:
        raise HTTPException(422, str(exc))
    return response(data, status="partial" if data["errors"] else "ready")


@router.post("/strategies/validate")
def validate_strategy(request: Request, body: StrategyDraft) -> dict:
    return response(services(request).strategies.validate(body))


@router.post("/strategies")
def save_strategy(request: Request, body: StrategyDraft) -> dict:
    return response(services(request).strategies.save(body))


@router.post("/strategies/launch")
def launch_strategy(request: Request, body: StrategyDraft) -> dict:
    service_container = services(request)
    run_output = f"{body.output_dir.rstrip('/')}/runs/run_{uuid4().hex[:12]}"
    body = body.model_copy(update={"output_dir": run_output})
    validation = service_container.strategies.validate(body)
    launch = validation["launch"]
    if not validation["valid"] or not launch["armed"]:
        raise HTTPException(
            422,
            {
                "message": "strategy validation and Human Gate are required",
                "errors": validation["errors"],
                "warnings": validation["warnings"],
            },
        )
    service_container.jobs.validate(
        launch["jobType"],
        launch["commandId"],
        launch["parameters"],
    )
    manifest = service_container.strategies.save(body)
    job = service_container.jobs.submit(
        launch["jobType"],
        launch["commandId"],
        launch["parameters"],
        # Labels travel with the job so a run found in the task centre can be
        # traced back to the strategy version that produced it.
        labels={
            "strategyId": manifest["id"],
            "strategyVersion": manifest["version"],
            "strategyName": manifest["name"],
        },
    )
    run = service_container.strategies.register_run(
        strategy_id=manifest["id"],
        version=manifest["version"],
        job_id=job["id"],
        output_dir=run_output,
        name=manifest["name"],
    )
    return response({"job": job, "strategy": manifest, "run": run})


@router.get("/risk/backtests/{backtest_id}")
def risk_backtest(request: Request, backtest_id: str) -> dict:
    if services(request).backtests.get(backtest_id) is None:
        raise HTTPException(404, "backtest not found")
    return response({
        "overview": services(request).risk.overview(backtest_id),
        "events": services(request).risk.events(backtest_id, 1, 200),
        "stocks": services(request).risk.stocks(backtest_id),
    })


@router.get("/do-t/sources")
def do_t_sources(request: Request) -> dict:
    data = services(request).do_t.list_sources()
    return response(data, status="ready" if data else "empty")


@router.get("/do-t/analysis")
def do_t_analysis(request: Request, source_id: str | None = Query(None, alias="sourceId"), symbol: str | None = None, limit: int = Query(500, le=1_000)) -> dict:
    try:
        data = services(request).do_t.analyze(source_id, symbol, limit)
        return response(data, status="ready" if data["pairs"] else "empty")
    except KeyError:
        raise HTTPException(404, "Do-T source not found")



@router.post("/jobs/data")
def create_data_job(request: Request, body: JobRequest) -> dict:
    return _create_job(request, "data", body)


@router.post("/jobs/backtest")
def create_backtest_job(request: Request, body: JobRequest) -> dict:
    return _create_job(request, "backtest", body)


@router.post("/jobs/train")
def create_train_job(request: Request, body: JobRequest) -> dict:
    return _create_job(request, "train", body)


@router.post("/jobs/factor-discovery")
def create_factor_discovery_job(request: Request, body: JobRequest) -> dict:
    return _create_job(request, "factor-discovery", body)


@router.post("/jobs/factor-evaluation")
def create_factor_evaluation_job(request: Request, body: JobRequest) -> dict:
    return _create_job(request, "factor-evaluation", body)


@router.post("/jobs/infer")
def create_infer_job(request: Request, body: JobRequest) -> dict:
    return _create_job(request, "infer", body)


@router.post("/jobs/governance")
def create_governance_job(request: Request, body: JobRequest) -> dict:
    return _create_job(request, "governance", body)


@router.post("/jobs/fusion-search")
def create_fusion_search_job(request: Request, body: JobRequest) -> dict:
    return _create_job(request, "fusion-search", body)


@router.post("/jobs/t-plus-one-research")
def create_t_plus_one_research_job(request: Request, body: JobRequest) -> dict:
    return _create_job(request, "t-plus-one-research", body)


@router.post("/jobs/{job_type}/validate")
def validate_job(request: Request, job_type: str, body: JobRequest) -> dict:
    if job_type not in {"data", "backtest", "train", "infer", "factor-discovery", "factor-evaluation", "fusion-search", "t-plus-one-research", "governance", "strategy-pipeline"}:
        raise HTTPException(404, "job type not found")
    try:
        return response(services(request).jobs.validate(job_type, body.command_id, body.parameters))
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get("/jobs")
def list_jobs(request: Request) -> dict:
    data = services(request).jobs.list()
    return response(data, status="ready" if data else "empty")



@router.post("/jobs/{job_id}/retry")
def retry_job(request: Request, job_id: str) -> dict:
    """Re-run a finished job with the parameters it actually ran with.

    This replays a job that already passed its gate — it cannot introduce new
    parameters — which is why there is still no route for submitting a
    strategy pipeline directly. A retry of a strategy run is registered against
    the same strategy, so its results stay attached to the research record
    rather than appearing as an orphan output directory.
    """
    svc = services(request)
    try:
        job = svc.jobs.retry(job_id)
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    labels = job.get("labels") or {}
    run = None
    if labels.get("strategyId") and job.get("parameters", {}).get("output_dir"):
        run = svc.strategies.register_run(
            strategy_id=labels["strategyId"],
            version=labels.get("strategyVersion", ""),
            job_id=job["id"],
            output_dir=job["parameters"]["output_dir"],
            name=labels.get("strategyName", labels["strategyId"]),
        )
    return response({**job, "run": run})


@router.post("/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> dict:
    try:
        return response(services(request).jobs.cancel(job_id))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.post("/jobs/{job_id}/pause")
def pause_job(request: Request, job_id: str) -> dict:
    try:
        return response(services(request).jobs.pause(job_id))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.post("/jobs/{job_id}/resume")
def resume_job(request: Request, job_id: str) -> dict:
    try:
        return response(services(request).jobs.resume(job_id))
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.delete("/jobs/{job_id}")
def purge_job(
    request: Request,
    job_id: str,
    delete_outputs: bool = Query(False, alias="deleteOutputs"),
) -> dict:
    try:
        return response(
            services(request).jobs.purge(job_id, delete_outputs=delete_outputs)
        )
    except KeyError:
        raise HTTPException(404, "job not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    data = services(request).jobs.get(job_id)
    if data is None:
        raise HTTPException(404, "job not found")
    return response(data)


@router.get("/jobs/{job_id}/logs")
def job_logs(request: Request, job_id: str, limit: int = Query(500, le=10_000)) -> dict:
    if services(request).jobs.get(job_id) is None:
        raise HTTPException(404, "job not found")
    data = services(request).jobs.logs(job_id, limit)
    return response(data, status="ready" if data else "empty")


@router.get("/jobs/{job_id}/stream")
def job_stream(request: Request, job_id: str):
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


# ---------------------------------------------------------------------------
# Paper orders — the only economic submission path in this API
# ---------------------------------------------------------------------------
# Every route here refuses live intent before it does anything else, and the
# broker it reaches has no connector, no credential and no network call. The
# submission is *queued*, not executed inline: making the HTTP response the
# execution would collapse "crash after responding" and "crash before executing"
# into one indistinguishable window, and those are the two that produce duplicates.
@router.get("/paper/policy")
def paper_policy(request: Request) -> dict:
    """The safety boundary, stated rather than implied."""
    service = services(request).paper_orders
    state = service.mode.to_dict()
    # `OperatingModeState.to_dict` is snake_case (it is `asdict`); this API is
    # camelCase. Translating at the boundary rather than changing the domain type,
    # which other surfaces already read.
    return response(
        {
            "mode": state["mode"],
            "declaredAt": state["declared_at"],
            "liveTradingAvailable": state["live_trading_available"],
            "banner": state["banner"],
            "executable": state["executable"],
            "simulatesOrders": state["simulatesOrders"],
            "paperBannerLines": state["paperBannerLines"],
            "writable": service.writable,
            "writerLockError": service.writer_lock_error,
        }
    )


@router.post("/paper/orders")
def submit_paper_order(request: Request, body: PaperOrderSubmission) -> dict:
    try:
        data = services(request).paper_orders.submit(
            body.model_dump(by_alias=True)
        )
    except LiveTradingRejected as exc:
        # 451: the request is refused on policy grounds, not because it was
        # malformed. A 422 would invite the client to "fix" it and retry.
        raise HTTPException(451, str(exc))
    except IdempotencyConflict as exc:
        raise HTTPException(409, str(exc))
    except WriterLockUnavailable as exc:
        # 409: the request is well-formed and permitted; this instance is simply
        # not the writer. Retrying against the writer will succeed.
        raise HTTPException(409, str(exc))
    except SubmissionRejected as exc:
        raise HTTPException(422, f"{exc.reason}: {exc}")
    return response(data, status="ready")


@router.post("/paper/orders/drain")
def drain_paper_orders(request: Request, limit: int = Query(50, ge=1, le=500)) -> dict:
    """Run the worker once. Exposed so an operator can drive it without a terminal."""
    return response(services(request).paper_orders.drain(limit=limit))


@router.post("/paper/orders/recover")
def recover_paper_orders(request: Request) -> dict:
    """Settle every queued submission against the ledger after a restart."""
    return response(services(request).paper_orders.recover())


@router.get("/paper/orders")
def list_paper_orders(request: Request) -> dict:
    data = services(request).paper_orders.orders()
    return response(data, status="ready" if data else "empty")


@router.get("/paper/orders/{idempotency_key}")
def get_paper_order(request: Request, idempotency_key: str, run_id: str = Query(alias="runId")) -> dict:
    data = services(request).paper_orders.status(idempotency_key, run_id)
    if data is None:
        raise HTTPException(404, "no submission is recorded for that key and run")
    return response(data)


@router.post("/paper/orders/{idempotency_key}/cancel")
def cancel_paper_order(
    request: Request, idempotency_key: str, body: PaperOrderCancellation
) -> dict:
    try:
        data = services(request).paper_orders.cancel(idempotency_key, body.run_id)
    except SubmissionRejected as exc:
        raise HTTPException(422, f"{exc.reason}: {exc}")
    return response(data)


@router.get("/paper/account")
def paper_account(request: Request) -> dict:
    """Cash, positions, PnL and NAV — replayed from the ledger, never from memory."""
    return response(services(request).paper_orders.account())
