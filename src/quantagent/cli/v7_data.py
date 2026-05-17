"""V7 data CLI: Qlib bootstrap, AkShare bootstrap, labels and training-dataset builders."""

from __future__ import annotations

from pathlib import Path

import typer

from quantagent.cli._utils import (
    app,
    default_v7_lake_root,
    json_dump,
    merge_symbols,
    parse_csv_tuple,
    read_frame,
    write_frame,
)


@app.command("download-qlib-v7")
def download_qlib_v7(
    target_dir: str = typer.Option(None, "--target-dir"),
    region: str = typer.Option("cn", "--region"),
    interval: str = typer.Option("1d", "--interval"),
) -> None:
    """Deprecated alias for setup-qlib-v7 dry-run output."""
    from quantagent.config.paths import quant_paths

    resolved = Path(target_dir).expanduser() if target_dir else quant_paths().raw / "qlib" / f"{region}_data"
    command = f"python scripts/get_data.py qlib_data --target_dir {resolved} --region {region} --interval {interval}"
    typer.echo(
        json_dump(
            {
                "status": "deprecated_alias",
                "replacement": "setup-qlib-v7",
                "target_dir": str(resolved),
                "official_command": command,
                "note": "download-qlib-v7 is retained only as a backwards-compatible alias.",
            }
        )
    )


@app.command("check-qlib-v7")
def check_qlib_v7(
    provider_uri: str = typer.Option(..., "--provider-uri"),
    start_date: str = typer.Option("2018-01-01", "--start-date"),
    end_date: str = typer.Option("2020-09-25", "--end-date"),
    symbols: str = typer.Option(
        "",
        "--symbols",
        help="Comma-separated qlib instruments, e.g. SH600519,SZ000001 (qlib CN uses uppercase prefix).",
    ),
    symbols_file: Path | None = typer.Option(None, "--symbols-file", help="Optional one-symbol-per-line universe file."),
    universe: str = typer.Option("", "--universe", help="Optional qlib universe name."),
    region: str = typer.Option("cn", "--region"),
) -> None:
    """Check local Qlib CN provider readiness and PIT market schema.

    Defaults to a range covered by the official Qlib CN free release
    (2000-01-04 .. 2020-09-25). For more recent data, prepare a custom
    dump via ``scripts/dump_bin.py`` and pass an explicit range.
    """
    from quantagent.data.providers.base import ProviderRequest
    from quantagent.data.providers.qlib_provider import QlibProvider

    request = None
    symbol_tuple = merge_symbols(symbols, symbols_file)
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
    output_root: Path = typer.Option(None, "--output-root"),
    symbols: str = typer.Option("", "--symbols"),
    symbols_file: Path | None = typer.Option(None, "--symbols-file", help="Optional one-symbol-per-line universe file."),
    universe: str = typer.Option("", "--universe"),
    region: str = typer.Option("cn", "--region"),
    require_optional_flags: bool = typer.Option(False, "--require-optional-flags"),
) -> None:
    """Export a PIT market panel and close-available-next-day features from local Qlib CN data."""
    from quantagent.data.bootstrap.qlib_bootstrap import QlibBootstrapConfig, build_qlib_market_panel

    resolved_root = Path(output_root) if output_root is not None else default_v7_lake_root()
    result = build_qlib_market_panel(
        QlibBootstrapConfig(
            provider_uri=provider_uri,
            start_date=start_date,
            end_date=end_date,
            symbols=merge_symbols(symbols, symbols_file),
            universe=universe or None,
            region=region,
            output_root=str(resolved_root),
            require_optional_flags=require_optional_flags,
        )
    )
    typer.echo(json_dump(result))


@app.command("build-akshare-v7")
def build_akshare_v7(
    symbols: str = typer.Option("", "--symbols"),
    symbols_file: Path | None = typer.Option(None, "--symbols-file", help="Optional one-symbol-per-line universe file."),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    fundamentals_root: Path = typer.Option(None, "--fundamentals-root"),
    allow_network: bool = typer.Option(False, "--allow-network"),
    lake_root: Path = typer.Option(None, "--lake-root"),
) -> None:
    """Download AkShare statements into the PIT financial cache and emit manifests."""
    from quantagent.data.bootstrap.akshare_bootstrap import AkShareBootstrapConfig, build_akshare_financial_cache

    resolved_lake = Path(lake_root) if lake_root is not None else default_v7_lake_root()
    result = build_akshare_financial_cache(
        AkShareBootstrapConfig(
            start_date=start_date,
            end_date=end_date,
            symbols=merge_symbols(symbols, symbols_file),
            fundamentals_root=str(fundamentals_root) if fundamentals_root else None,
            allow_network=allow_network,
            lake_root=str(resolved_lake),
        )
    )
    typer.echo(json_dump(result))
    if result.get("status") == "empty":
        raise typer.Exit(code=1)


@app.command("build-akshare-market-panel-v7")
def build_akshare_market_panel_v7(
    symbols: str = typer.Option("", "--symbols", help="Comma-separated A-share symbols, e.g. 600519.SH,000001.SZ."),
    symbols_file: Path | None = typer.Option(None, "--symbols-file", help="Optional one-symbol-per-line universe file."),
    start_date: str | None = typer.Option(None, "--start-date"),
    end_date: str | None = typer.Option(None, "--end-date"),
    output_root: Path = typer.Option(None, "--output-root"),
    output_path: Path = typer.Option(None, "--output"),
    allow_network: bool = typer.Option(False, "--allow-network"),
    adjust: str = typer.Option("qfq", "--adjust"),
    provider_uri_for_range: Path | None = typer.Option(
        None,
        "--provider-uri-for-range",
        help="Optional Qlib provider_uri used to infer the AkShare start date after the local Qlib calendar.",
    ),
    as_of_date: str | None = typer.Option(None, "--as-of-date", help="Default end-date anchor; weekends roll back to Friday."),
) -> None:
    """Build a recent PIT market panel from AkShare daily OHLCV.

    This is the real-data path for dates beyond the official free Qlib CN
    dump. Daily bars are marked available on the next business day so labels
    and training can enforce PIT semantics downstream.
    """
    from quantagent.config.paths import quant_paths
    from quantagent.data.bootstrap.akshare_market_bootstrap import AkShareMarketPanelConfig, build_akshare_market_panel

    resolved_root = Path(output_root) if output_root is not None else default_v7_lake_root()
    range_provider = provider_uri_for_range or (quant_paths().raw / "qlib" / "cn_data")
    payload = build_akshare_market_panel(
        AkShareMarketPanelConfig(
            symbols=merge_symbols(symbols, symbols_file),
            start_date=start_date,
            end_date=end_date,
            output_root=str(resolved_root),
            output_path=str(output_path) if output_path else None,
            allow_network=allow_network,
            adjust=adjust,
            provider_uri_for_range=str(range_provider) if range_provider else None,
            as_of_date=as_of_date,
        )
    )
    typer.echo(json_dump(payload))
    if payload["status"] != "passed":
        raise typer.Exit(code=1)


@app.command("smoke-akshare-v7")
def smoke_akshare_v7(
    symbols: str = typer.Option("", "--symbols"),
    symbols_file: Path | None = typer.Option(None, "--symbols-file", help="Optional one-symbol-per-line universe file."),
    start_date: str = typer.Option("2025-01-01", "--start-date"),
    end_date: str = typer.Option("2026-05-15", "--end-date"),
    as_of_date: str = typer.Option("2026-05-15", "--as-of-date"),
    allow_network: bool = typer.Option(False, "--allow-network"),
) -> None:
    """Probe AkShare market, valuation and PIT financial availability.

    The command never fabricates rows. Network access must be explicitly
    enabled; failures are returned as per-source warnings with a non-zero
    status when every probe fails.
    """
    from quantagent.data.providers.akshare_financial_provider import AkShareFinancialProvider
    from quantagent.data.providers.akshare_live_provider import AkShareLiveProvider
    from quantagent.data.providers.akshare_valuation_provider import AkShareValuationProvider
    from quantagent.data.providers.base import ProviderRequest

    request = ProviderRequest(start_date=start_date, end_date=end_date, symbols=merge_symbols(symbols, symbols_file))
    probes: dict[str, object] = {}
    warnings: list[str] = []

    try:
        market = AkShareLiveProvider(allow_network=allow_network).daily_ohlcv(request)
        probes["market"] = {
            "status": "passed" if not market.frame.empty else "empty",
            "rows": int(len(market.frame)),
            "warnings": list(market.warnings),
            "schema_report": market.metadata.get("schema_report", {}),
        }
        warnings.extend(market.warnings)
    except Exception as exc:
        warning = f"market_failed:{type(exc).__name__}:{exc}"
        probes["market"] = {"status": "failed", "warning": warning}
        warnings.append(warning)

    provider = AkShareFinancialProvider(allow_network=allow_network)
    for name, fetch in {
        "income": provider.income,
        "balance_sheet": provider.balance_sheet,
        "cashflow": provider.cashflow,
        "financial_indicator": provider.financial_indicator,
    }.items():
        try:
            result = fetch(request)
            probes[name] = {
                "status": "passed" if not result.frame.empty else "empty",
                "rows": int(len(result.frame)),
                "warnings": list(result.warnings),
                "schema_report": result.metadata.get("schema_report", {}),
            }
            warnings.extend(result.warnings)
        except Exception as exc:
            warning = f"{name}_failed:{type(exc).__name__}:{exc}"
            probes[name] = {"status": "failed", "warning": warning}
            warnings.append(warning)

    try:
        valuation = AkShareValuationProvider(allow_network=allow_network).snapshot(as_of_date, request)
        probes["valuation"] = {
            "status": "passed" if not valuation.frame.empty else "empty",
            "rows": int(len(valuation.frame)),
            "warnings": list(valuation.warnings),
            "schema_report": valuation.metadata.get("schema_report", {}),
        }
        warnings.extend(valuation.warnings)
    except Exception as exc:
        warning = f"valuation_failed:{type(exc).__name__}:{exc}"
        probes["valuation"] = {"status": "failed", "warning": warning}
        warnings.append(warning)

    passed = any(isinstance(item, dict) and item.get("status") == "passed" for item in probes.values())
    payload = {
        "status": "passed" if passed else "failed",
        "allow_network": allow_network,
        "symbols": list(request.symbols),
        "probes": probes,
        "warnings": warnings,
    }
    typer.echo(json_dump(payload))
    if not passed:
        raise typer.Exit(code=1)


@app.command("build-valuation-v7")
def build_valuation_v7(
    as_of_dates: str = typer.Option("", "--as-of-dates", help="Comma-separated valuation snapshot dates (YYYY-MM-DD)."),
    symbols: str = typer.Option("", "--symbols"),
    symbols_file: Path | None = typer.Option(None, "--symbols-file", help="Optional one-symbol-per-line universe file."),
    lake_root: Path = typer.Option(None, "--lake-root"),
    allow_network: bool = typer.Option(False, "--allow-network"),
    csv_snapshot: Path | None = typer.Option(None, "--csv-snapshot", help="Optional pre-collected valuation snapshot."),
    output_name: str = typer.Option("valuation.parquet", "--output-name"),
) -> None:
    """Build the silver valuation cache from AkShare snapshots or a local CSV."""
    from quantagent.data.bootstrap.valuation_bootstrap import ValuationBootstrapConfig, build_valuation_cache

    resolved_lake = Path(lake_root) if lake_root is not None else default_v7_lake_root()
    result = build_valuation_cache(
        ValuationBootstrapConfig(
            as_of_dates=parse_csv_tuple(as_of_dates),
            symbols=merge_symbols(symbols, symbols_file),
            lake_root=str(resolved_lake),
            allow_network=allow_network,
            csv_snapshot=str(csv_snapshot) if csv_snapshot else None,
            output_name=output_name,
        )
    )
    typer.echo(json_dump(result))


@app.command("build-fundamentals-v7")
def build_fundamentals_v7(
    symbols: str = typer.Option("", "--symbols", help="Comma-separated A-share symbols (e.g. 600519.SH,000858.SZ)"),
    symbols_file: Path | None = typer.Option(None, "--symbols-file", help="Optional one-symbol-per-line universe file."),
    start_date: str = typer.Option(..., "--start-date"),
    end_date: str = typer.Option(..., "--end-date"),
    provider: str = typer.Option("tushare", "--provider", help="tushare or akshare"),
    fundamentals_root: Path = typer.Option(None, "--fundamentals-root"),
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
        symbols=merge_symbols(symbols, symbols_file),
    )
    if provider == "tushare":
        adapter = TuShareFinancialProvider(allow_network=allow_network, token_env=token_env)
    elif provider == "akshare":
        adapter = AkShareFinancialProvider(allow_network=allow_network)
    else:
        raise typer.BadParameter("provider must be tushare or akshare")
    statements = adapter.all_statements(request)
    if fundamentals_root is None:
        from quantagent.config.paths import quant_paths

        fundamentals_root = quant_paths().data_root / "v7" / "raw" / provider / "fundamentals"
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
    output_path: Path = typer.Option(None, "--output"),
    horizons: str = typer.Option("1,5,20,60,120,126", "--horizons"),
    symbols: str = typer.Option("", "--symbols"),
    symbols_file: Path | None = typer.Option(None, "--symbols-file", help="Optional one-symbol-per-line universe file."),
) -> None:
    """Build future-return labels for training; labels must never be used for inference."""
    from quantagent.data.v7_label_builder import build_forward_return_labels

    resolved_output = Path(output_path) if output_path is not None else default_v7_lake_root() / "labels.parquet"
    frame = read_frame(market_panel_path)
    symbol_tuple = merge_symbols(symbols, symbols_file)
    if symbol_tuple:
        frame = frame[frame["symbol"].astype(str).isin(set(symbol_tuple))].reset_index(drop=True)
    result = build_forward_return_labels(frame, tuple(int(item) for item in parse_csv_tuple(horizons)))
    actual = write_frame(result.frame, resolved_output)
    typer.echo(json_dump({"status": "passed", "output": str(actual), "rows": len(result.frame), "label_schema": result.label_schema}))


@app.command("build-training-dataset-v7")
def build_training_dataset_v7(
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    labels_path: Path = typer.Option(..., "--labels"),
    output_path: Path = typer.Option(None, "--output"),
    fundamentals_root: Path | None = typer.Option(None, "--fundamentals-root"),
    valuation_path: Path | None = typer.Option(None, "--valuation"),
    disclosures_path: Path | None = typer.Option(None, "--disclosures"),
    start_date: str | None = typer.Option(None, "--start-date"),
    end_date: str | None = typer.Option(None, "--end-date"),
    symbols: str = typer.Option("", "--symbols"),
    symbols_file: Path | None = typer.Option(None, "--symbols-file", help="Optional one-symbol-per-line universe file."),
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

    resolved_output = (
        Path(output_path)
        if output_path is not None
        else default_v7_lake_root() / "gold" / "training_dataset" / "training_dataset.parquet"
    )
    config = V7TrainingDatasetConfig(
        market_panel_path=str(market_panel_path),
        labels_path=str(labels_path),
        output_path=str(resolved_output),
        manifest_path=str(manifest_path) if manifest_path else None,
        fundamentals_root=str(fundamentals_root) if fundamentals_root else None,
        valuation_path=str(valuation_path) if valuation_path else None,
        disclosures_path=str(disclosures_path) if disclosures_path else None,
        start_date=start_date,
        end_date=end_date,
        symbols=merge_symbols(symbols, symbols_file),
        horizons=tuple(int(item) for item in parse_csv_tuple(horizons)),
        min_rows=min_rows,
        min_symbols=min_symbols,
        min_dates=min_dates,
        enforce_quality_gates=enforce_quality_gates,
        source_name=source_name,
    )
    result = build_v7_training_dataset_artifact(config)
    typer.echo(json_dump(result.summary))


@app.command("materialize-factors-v7")
def materialize_factors_v7(
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    output_path: Path = typer.Option(None, "--output"),
    backend: str = typer.Option("pandas", "--backend", help="pandas or polars"),
    long_format: bool = typer.Option(False, "--long-format"),
) -> None:
    """Materialize default Alpha101-style factors into the V7 lake."""
    from quantagent.data.manifest import utc_now_iso
    from quantagent.factors.expr import build_factor_frame, build_factor_manifest

    resolved_output = (
        Path(output_path)
        if output_path is not None
        else default_v7_lake_root() / "silver" / "factors" / "factors.parquet"
    )
    factors = build_factor_frame(read_frame(market_panel_path), backend=backend, long_format=long_format)
    written = write_frame(factors, resolved_output)
    manifest = {
        "dataset_name": "factors",
        "created_at": utc_now_iso(),
        "backend": backend,
        "row_count": int(len(factors)),
        "column_count": int(len(factors.columns)),
        "output_path": str(written),
        "factors": build_factor_manifest(backend=backend),
        "warnings": [
            entry["fallback"]
            for entry in build_factor_manifest(backend=backend)
            if entry.get("fallback")
        ],
    }
    manifest_path = Path(written).with_suffix(".manifest.json")
    manifest_path.write_text(json_dump(manifest), encoding="utf-8")
    typer.echo(
        json_dump(
            {
                "status": "passed",
                "backend": backend,
                "rows": int(len(factors)),
                "columns": list(factors.columns),
                "output": str(written),
                "manifest": str(manifest_path),
            }
        )
    )
