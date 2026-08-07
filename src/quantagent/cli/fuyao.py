"""Governed Fuyao/HiThink data acquisition commands."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import typer

from quantagent.cli._utils import app
from quantagent.data.fuyao_docs_audit import audit_live_documentation
from quantagent.data.fuyao_dump import DUMP_ENDPOINTS, download_fuyao_market_dump
from quantagent.data.fuyao_full_sync import FuyaoFullSynchronizer, build_coverage_audit
from quantagent.data.manifest import build_manifest_for_frame
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.providers.fuyao_provider import FuyaoProvider


def _parse_symbols(symbols: str) -> tuple[str, ...]:
    values = tuple(
        dict.fromkeys(item.strip().upper() for item in symbols.split(",") if item.strip())
    )
    if not values:
        raise typer.BadParameter("symbols resolved to an empty list")
    for symbol in values:
        if "." not in symbol:
            raise typer.BadParameter(
                f"Fuyao requires canonical thscode with exchange suffix, got {symbol!r}"
            )
    return values


def _parse_optional_symbols(symbols: str) -> tuple[str, ...]:
    if not symbols.strip():
        return ()
    return _parse_symbols(symbols)


def _parse_params(raw: str) -> dict[str, object]:
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"params-json is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("params-json must decode to a JSON object")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@app.command("fetch-fuyao-daily")
def fetch_fuyao_daily(
    symbols: str = typer.Option(
        ..., help="comma-separated canonical thscodes, e.g. 600519.SH,000001.SZ"
    ),
    start_date: str = typer.Option(...),
    end_date: str = typer.Option(...),
    output: Path = typer.Option(..., dir_okay=False),
    adjust: str = typer.Option("none", help="none|forward|backward"),
    allow_network: bool = typer.Option(False),
):
    """Fetch a bounded set of official daily bars into a PIT-attributed parquet."""
    if not allow_network:
        raise typer.BadParameter("network access is fail-closed; pass --allow-network")
    if adjust not in {"none", "forward", "backward"}:
        raise typer.BadParameter("adjust must be none, forward or backward")

    names = _parse_symbols(symbols)
    provider = FuyaoProvider(allow_network=True)
    request = ProviderRequest(start_date=start_date, end_date=end_date, symbols=names)
    result = provider.historical_prices(request, adjust=adjust)

    frame = result.frame.copy()
    if frame.empty:
        raise typer.BadParameter("Fuyao returned no daily bars for the requested window")
    frame = frame.drop_duplicates(["symbol", "trade_date"]).sort_values(
        ["trade_date", "symbol"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest = build_manifest_for_frame(
        dataset_name=f"fuyao_daily_{adjust}",
        vendor="hithink_fuyao",
        frame=frame,
        output_paths=(output,),
        start_date=start_date,
        end_date=end_date,
        symbols=names,
        required_columns=(
            "symbol",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "available_at",
            "source",
            "source_endpoint",
            "quality_status",
        ),
        pit_violation_count=0,
        warnings=("adjusted_view_not_canonical_raw_panel",) if adjust != "none" else (),
        extra={
            "adjustment": adjust,
            "source_endpoint": "/api/a-share/prices/historical",
        },
    )
    manifest.write(manifest_path)
    typer.echo(
        f"rows={len(frame)} symbols={frame['symbol'].nunique()} "
        f"output={output} manifest={manifest_path} source=hithink_fuyao"
    )
    return output


@app.command("fetch-fuyao-capability")
def fetch_fuyao_capability(
    path: str = typer.Option(
        ...,
        help="documented read-only path under /api/, e.g. /api/a-share/special-data/hot-stock-list",
    ),
    params_json: str = typer.Option("{}", help="JSON object of query parameters"),
    output: Path = typer.Option(..., dir_okay=False, help="JSON output path"),
    allow_network: bool = typer.Option(False),
):
    """Fetch any documented Fuyao ``/api/*`` read capability as source-attributed JSON."""
    if not allow_network:
        raise typer.BadParameter("network access is fail-closed; pass --allow-network")
    if not path.startswith("/api/") or "://" in path or "?" in path or "#" in path:
        raise typer.BadParameter(
            "path must be a clean documented /api/* path; put query fields in --params-json"
        )
    if output.suffix.lower() != ".json":
        raise typer.BadParameter("fetch-fuyao-capability output must use .json")

    params = _parse_params(params_json)
    provider = FuyaoProvider(allow_network=True)
    data = provider.get_capability(path, params=params)
    artifact = {
        "source": "hithink_fuyao",
        "source_endpoint": path,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "quality_status": "official_api",
        "point_in_time_valid": False,
        "pit_note": (
            "generic capability output is not automatically PIT-safe; historical model use requires "
            "an endpoint-specific availability timestamp and canonical normalizer"
        ),
        "request_params": params,
        "data": data,
    }
    _write_json(output, artifact)
    typer.echo(f"source=hithink_fuyao endpoint={path} output={output}")
    return output


@app.command("fetch-fuyao-market-dump")
def fetch_fuyao_dump(
    dataset: str = typer.Option(..., help="daily-k|daily-k-10d|adjustment-factors"),
    output: Path = typer.Option(..., dir_okay=False),
    manifest: Path | None = typer.Option(None, dir_okay=False),
    timeout: float = typer.Option(180.0, min=10.0, max=3600.0),
    allow_network: bool = typer.Option(False),
):
    """Download the official full-market parquet path for large research jobs."""
    if dataset not in DUMP_ENDPOINTS:
        raise typer.BadParameter(f"dataset must be one of {sorted(DUMP_ENDPOINTS)}")
    result = download_fuyao_market_dump(
        dataset,
        output,
        manifest_path=manifest,
        allow_network=allow_network,
        timeout=timeout,
    )
    typer.echo(
        f"dataset={result.dataset} rows={result.rows} output={result.output} "
        f"manifest={result.manifest} sha256={result.sha256[:12]}"
    )
    return result.output


@app.command("audit-fuyao-coverage")
def audit_fuyao_coverage(
    output: Path | None = typer.Option(None, dir_okay=False, help="optional JSON audit output"),
    live_docs: bool = typer.Option(
        False, help="also download current llms-full.txt and compare route/tool sets"
    ),
    timeout: float = typer.Option(30.0, min=5.0, max=300.0),
    allow_network: bool = typer.Option(False),
):
    """Prove that every currently documented REST/MCP/dump capability is classified."""
    audit: dict[str, object] = {"registry": build_coverage_audit()}
    if live_docs:
        if not allow_network:
            raise typer.BadParameter("--live-docs requires --allow-network")
        live = audit_live_documentation(timeout=timeout)
        audit["live_docs"] = live
        if not live["ok"]:
            if output is not None:
                _write_json(output, audit)
            raise typer.BadParameter(
                f"Fuyao live documentation drift detected: {live['diffs']}"
            )
    text = json.dumps(audit, ensure_ascii=False, indent=2)
    if output is not None:
        if output.suffix.lower() != ".json":
            raise typer.BadParameter("coverage audit output must use .json")
        _write_json(output, audit)
        typer.echo(f"Fuyao coverage audit written to {output}")
    else:
        typer.echo(text)
    counts = audit["registry"]["counts"]  # type: ignore[index]
    typer.echo(
        "coverage="
        f"rest:{counts['live_rest']} mcp:{counts['live_mcp']} "
        f"dumps:{counts['market_dumps']} coming_soon:{counts['coming_soon']}"
    )
    return output


@app.command("sync-fuyao-all")
def sync_fuyao_all(
    output_dir: Path = typer.Option(..., file_okay=False, help="raw archive + audit root"),
    deep: bool = typer.Option(
        True,
        help="enumerate per-symbol/per-date financial, index, fund and special-data history",
    ),
    include_dumps: bool = typer.Option(
        True, help="download all three official full-market Parquet dumps"
    ),
    resume: bool = typer.Option(True, help="reuse already archived request artifacts"),
    stop_on_error: bool = typer.Option(
        False, help="fail immediately instead of recording a gap and continuing"
    ),
    verify_live_docs: bool = typer.Option(
        True, help="fail closed if current llms-full.txt differs from the checked-in registry"
    ),
    extra_reits: str = typer.Option(
        "",
        help="comma-separated REIT thscodes; official meta ticker-list currently has no fund-reit enum",
    ),
    docs_timeout: float = typer.Option(30.0, min=5.0, max=300.0),
    dump_timeout: float = typer.Option(180.0, min=10.0, max=3600.0),
    allow_network: bool = typer.Option(False),
):
    """Archive every documented live Fuyao data class within official retention limits."""
    if not allow_network:
        raise typer.BadParameter("network access is fail-closed; pass --allow-network")
    reits = _parse_optional_symbols(extra_reits)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verify_live_docs:
        live = audit_live_documentation(timeout=docs_timeout)
        _write_json(output_dir / "fuyao_live_docs_audit.json", live)
        if not live["ok"]:
            raise typer.BadParameter(
                f"Fuyao live documentation drift detected before sync: {live['diffs']}"
            )

    sync = FuyaoFullSynchronizer(
        output_dir,
        provider=FuyaoProvider(allow_network=True),
        resume=resume,
        stop_on_error=stop_on_error,
    )
    report = sync.run(
        deep=deep,
        extra_reits=reits,
        include_dumps=include_dumps,
        dump_timeout=dump_timeout,
    )
    typer.echo(f"Fuyao full sync report={report}")
    return report
