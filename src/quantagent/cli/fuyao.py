"""Governed Fuyao/HiThink data acquisition commands."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import typer

from quantagent.cli._utils import app
from quantagent.data.fuyao_dump import DUMP_ENDPOINTS, download_fuyao_market_dump
from quantagent.data.manifest import build_manifest_for_frame
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.providers.fuyao_provider import FuyaoProvider


def _parse_symbols(symbols: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(item.strip().upper() for item in symbols.split(",") if item.strip()))
    if not values:
        raise typer.BadParameter("symbols resolved to an empty list")
    for symbol in values:
        if "." not in symbol:
            raise typer.BadParameter(
                f"Fuyao requires canonical thscode with exchange suffix, got {symbol!r}"
            )
    return values


def _parse_params(raw: str) -> dict[str, object]:
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"params-json is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("params-json must decode to a JSON object")
    return payload


@app.command("fetch-fuyao-daily")
def fetch_fuyao_daily(
    symbols: str = typer.Option(..., help="comma-separated canonical thscodes, e.g. 600519.SH,000001.SZ"),
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
    frame = frame.drop_duplicates(["symbol", "trade_date"]).sort_values(["trade_date", "symbol"])
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
        extra={"adjustment": adjust, "source_endpoint": "/api/a-share/prices/historical"},
    )
    manifest.write(manifest_path)
    typer.echo(
        f"rows={len(frame)} symbols={frame['symbol'].nunique()} "
        f"output={output} manifest={manifest_path} source=hithink_fuyao"
    )
    return output


@app.command("fetch-fuyao-capability")
def fetch_fuyao_capability(
    path: str = typer.Option(..., help="documented read-only path under /api/, e.g. /api/a-share/special-data/hot-stock-list"),
    params_json: str = typer.Option("{}", help="JSON object of query parameters"),
    output: Path = typer.Option(..., dir_okay=False, help="JSON output path"),
    allow_network: bool = typer.Option(False),
):
    """Fetch any documented Fuyao ``/api/*`` read capability as source-attributed JSON.

    This is the forward-compatible path for funds, index/sector and special-data
    endpoints that do not need a dedicated QuantAgent normalizer yet. The
    adapter still owns authentication, retry and business-error handling; an
    arbitrary external URL cannot be supplied here.
    """
    if not allow_network:
        raise typer.BadParameter("network access is fail-closed; pass --allow-network")
    if not path.startswith("/api/") or "://" in path or "?" in path or "#" in path:
        raise typer.BadParameter("path must be a clean documented /api/* path; put query fields in --params-json")
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
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
