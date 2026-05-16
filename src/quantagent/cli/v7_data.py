"""V7 data CLI: Qlib bootstrap, AkShare bootstrap, labels and training-dataset builders."""

from __future__ import annotations

from pathlib import Path

import typer

from quantagent.cli._utils import app, json_dump, parse_csv_tuple, read_frame, write_frame


@app.command("download-qlib-v7")
def download_qlib_v7(
    target_dir: str = typer.Option("~/.qlib/qlib_data/cn_data", "--target-dir"),
    region: str = typer.Option("cn", "--region"),
) -> None:
    """Print the official Qlib CN data command; run it inside a Qlib checkout."""
    command = f"python scripts/get_data.py qlib_data --target_dir {target_dir} --region {region}"
    typer.echo(json_dump({"status": "manual_step_required", "command": command, "note": "run this inside the official Qlib scripts directory"}))


@app.command("check-qlib-v7")
def check_qlib_v7(
    provider_uri: str = typer.Option(..., "--provider-uri"),
    start_date: str = typer.Option("2026-05-01", "--start-date"),
    end_date: str = typer.Option("2026-05-15", "--end-date"),
    symbols: str = typer.Option("", "--symbols", help="Optional comma-separated symbols for a schema probe."),
    universe: str = typer.Option("", "--universe", help="Optional qlib universe name."),
    region: str = typer.Option("cn", "--region"),
) -> None:
    """Check local Qlib CN provider readiness and PIT market schema."""
    from quantagent.data.providers.base import ProviderRequest
    from quantagent.data.providers.qlib_provider import QlibProvider

    request = None
    symbol_tuple = parse_csv_tuple(symbols)
    if symbol_tuple or universe:
        request = ProviderRequest(
            start_date=start_date,
            end_date=end_date,
            symbols=symbol_tuple,
            universe=universe or None,
        )
    result = QlibProvider(provider_uri=provider_uri, region=region).health_check(request)
    typer.echo(json_dump(result))


@app.command("build-market-panel-v7")
def build_market_panel_v7(
    provider_uri: str = typer.Option(..., "--provider-uri"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    output_root: Path = typer.Option(Path("data/v7"), "--output-root"),
    symbols: str = typer.Option("", "--symbols"),
    universe: str = typer.Option("", "--universe"),
    region: str = typer.Option("cn", "--region"),
    require_optional_flags: bool = typer.Option(False, "--require-optional-flags"),
) -> None:
    """Export a PIT market panel and close-available-next-day features from local Qlib CN data."""
    from quantagent.data.bootstrap.qlib_bootstrap import QlibBootstrapConfig, build_qlib_market_panel

    result = build_qlib_market_panel(
        QlibBootstrapConfig(
            provider_uri=provider_uri,
            start_date=start_date,
            end_date=end_date,
            symbols=parse_csv_tuple(symbols),
            universe=universe or None,
            region=region,
            output_root=str(output_root),
            require_optional_flags=require_optional_flags,
        )
    )
    typer.echo(json_dump(result))


@app.command("build-akshare-v7")
def build_akshare_v7(
    symbols: str = typer.Option(..., "--symbols"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    fundamentals_root: Path = typer.Option(Path("data/v7/fundamentals"), "--fundamentals-root"),
    allow_network: bool = typer.Option(False, "--allow-network"),
    lake_root: Path = typer.Option(Path("data/v7"), "--lake-root"),
) -> None:
    """Download AkShare statements into the PIT financial cache and emit manifests."""
    from quantagent.data.bootstrap.akshare_bootstrap import AkShareBootstrapConfig, build_akshare_financial_cache

    result = build_akshare_financial_cache(
        AkShareBootstrapConfig(
            start_date=start_date,
            end_date=end_date,
            symbols=parse_csv_tuple(symbols),
            fundamentals_root=str(fundamentals_root),
            allow_network=allow_network,
            lake_root=str(lake_root),
        )
    )
    typer.echo(json_dump(result))


@app.command("build-valuation-v7")
def build_valuation_v7(
    as_of_dates: str = typer.Option("", "--as-of-dates", help="Comma-separated valuation snapshot dates (YYYY-MM-DD)."),
    symbols: str = typer.Option("", "--symbols"),
    lake_root: Path = typer.Option(Path("data/v7"), "--lake-root"),
    allow_network: bool = typer.Option(False, "--allow-network"),
    csv_snapshot: Path | None = typer.Option(None, "--csv-snapshot", help="Optional pre-collected valuation snapshot."),
    output_name: str = typer.Option("valuation.parquet", "--output-name"),
) -> None:
    """Build the silver valuation cache from AkShare snapshots or a local CSV."""
    from quantagent.data.bootstrap.valuation_bootstrap import ValuationBootstrapConfig, build_valuation_cache

    result = build_valuation_cache(
        ValuationBootstrapConfig(
            as_of_dates=parse_csv_tuple(as_of_dates),
            symbols=parse_csv_tuple(symbols),
            lake_root=str(lake_root),
            allow_network=allow_network,
            csv_snapshot=str(csv_snapshot) if csv_snapshot else None,
            output_name=output_name,
        )
    )
    typer.echo(json_dump(result))


@app.command("build-fundamentals-v7")
def build_fundamentals_v7(
    symbols: str = typer.Option(..., "--symbols", help="Comma-separated A-share symbols (e.g. 600519.SH,000858.SZ)"),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    provider: str = typer.Option("tushare", "--provider", help="tushare or akshare"),
    fundamentals_root: Path = typer.Option(Path("data/v7/fundamentals"), "--fundamentals-root"),
    allow_network: bool = typer.Option(False, "--allow-network"),
    token_env: str = typer.Option("TUSHARE_TOKEN", "--token-env"),
) -> None:
    """Pull PIT-aware financial statements from TuShare/AkShare and write them to the V7 cache."""
    from quantagent.data.providers.akshare_financial_provider import AkShareFinancialProvider
    from quantagent.data.providers.base import ProviderRequest
    from quantagent.data.providers.financial_cache import FinancialCacheConfig, FinancialStatementCache
    from quantagent.data.providers.tushare_financial_provider import TuShareFinancialProvider

    request = ProviderRequest(
        start_date=start_date,
        end_date=end_date,
        symbols=parse_csv_tuple(symbols),
    )
    if provider == "tushare":
        adapter = TuShareFinancialProvider(allow_network=allow_network, token_env=token_env)
    elif provider == "akshare":
        adapter = AkShareFinancialProvider(allow_network=allow_network)
    else:
        raise typer.BadParameter("provider must be tushare or akshare")
    statements = adapter.all_statements(request)
    cache = FinancialStatementCache(FinancialCacheConfig(root=str(fundamentals_root)))
    summary: dict[str, dict[str, object]] = {}
    for name, result in statements.items():
        path = cache.upsert(name, result.frame)
        summary[name] = {
            "rows": int(0 if result.frame is None else len(result.frame)),
            "source": result.source,
            "path": str(path),
            "warnings": list(result.warnings),
        }
    typer.echo(json_dump({"provider": provider, "statements": summary}))


@app.command("build-labels-v7")
def build_labels_v7(
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    output_path: Path = typer.Option(Path("data/v7/labels.parquet"), "--output"),
    horizons: str = typer.Option("1,5,20,60,120,126", "--horizons"),
) -> None:
    """Build future-return labels for training; labels must never be used for inference."""
    from quantagent.data.v7_label_builder import build_forward_return_labels

    frame = read_frame(market_panel_path)
    result = build_forward_return_labels(frame, tuple(int(item) for item in parse_csv_tuple(horizons)))
    actual = write_frame(result.frame, output_path)
    typer.echo(json_dump({"status": "passed", "output": str(actual), "rows": len(result.frame), "label_schema": result.label_schema}))


@app.command("build-training-dataset-v7")
def build_training_dataset_v7(
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    labels_path: Path = typer.Option(..., "--labels"),
    output_path: Path = typer.Option(Path("data/v7/gold/training_dataset/training_dataset.parquet"), "--output"),
    fundamentals_root: Path | None = typer.Option(None, "--fundamentals-root"),
    valuation_path: Path | None = typer.Option(None, "--valuation"),
    disclosures_path: Path | None = typer.Option(None, "--disclosures"),
    start_date: str | None = typer.Option(None, "--start-date"),
    end_date: str | None = typer.Option(None, "--end-date"),
    symbols: str = typer.Option("", "--symbols"),
    horizons: str = typer.Option("1,5,20,60,120,126", "--horizons"),
    min_rows: int = typer.Option(100, "--min-rows"),
    min_symbols: int = typer.Option(2, "--min-symbols"),
    min_dates: int = typer.Option(5, "--min-dates"),
    enforce_quality_gates: bool = typer.Option(True, "--enforce-quality-gates/--no-enforce-quality-gates"),
    manifest_path: Path | None = typer.Option(None, "--manifest"),
    source_name: str = typer.Option("realdata", "--source-name"),
) -> None:
    """Build the V7 gold-tier training dataset via PIT as-of joins and forward labels."""
    from quantagent.data.dataset_builder import V7TrainingDatasetConfig, build_v7_training_dataset_artifact

    config = V7TrainingDatasetConfig(
        market_panel_path=str(market_panel_path),
        labels_path=str(labels_path),
        output_path=str(output_path),
        manifest_path=str(manifest_path) if manifest_path else None,
        fundamentals_root=str(fundamentals_root) if fundamentals_root else None,
        valuation_path=str(valuation_path) if valuation_path else None,
        disclosures_path=str(disclosures_path) if disclosures_path else None,
        start_date=start_date,
        end_date=end_date,
        symbols=parse_csv_tuple(symbols),
        horizons=tuple(int(item) for item in parse_csv_tuple(horizons)),
        min_rows=min_rows,
        min_symbols=min_symbols,
        min_dates=min_dates,
        enforce_quality_gates=enforce_quality_gates,
        source_name=source_name,
    )
    result = build_v7_training_dataset_artifact(config)
    typer.echo(json_dump(result.summary))
