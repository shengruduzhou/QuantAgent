"""Governed Fuyao/HiThink data acquisition commands."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
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
    if adjust == "none":
        result = provider.daily_ohlcv(request)
    else:
        # Public provider method uses forward adjustment by contract. Backward is
        # requested explicitly through the documented endpoint below.
        if adjust == "forward":
            result = provider.adjusted_prices(request)
        else:
            frames = []
            for symbol in names:
                data = provider.get_capability(
                    "/api/a-share/prices/historical",
                    params={
                        "thscode": symbol,
                        "interval": "1d",
                        "start": int(pd.Timestamp(start_date, tz="Asia/Shanghai").timestamp() * 1000),
                        "end": int(pd.Timestamp(end_date, tz="Asia/Shanghai").timestamp() * 1000),
                        "adjust": "backward",
                    },
                )
                rows = data.get("item", [])
                if not rows:
                    continue
                frame = pd.DataFrame(rows).rename(
                    columns={
                        "open_price": "open",
                        "high_price": "high",
                        "low_price": "low",
                        "close_price": "close",
                        "turnover": "amount",
                    }
                )
                frame["symbol"] = symbol
                frame["trade_date"] = pd.to_datetime(frame["date_ms"], unit="ms", errors="coerce").dt.normalize()
                frame["available_at"] = frame["trade_date"] + pd.Timedelta(days=1)
                frame["source"] = "hithink_fuyao"
                frame["source_endpoint"] = "/api/a-share/prices/historical"
                frame["quality_status"] = "official_api"
                frame["adjustment"] = "backward"
                frames.append(frame)
            out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            from quantagent.data.providers.base import ProviderResult

            result = ProviderResult(out, "hithink_fuyao", True, 0.98)

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
