"""Sector map and ST flag data-layer commands."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from quantagent.cli._utils import app, default_v7_lake_root, json_dump, merge_symbols, read_frame, write_frame
from quantagent.data.fundamental import (
    FundamentalRankerBuilder,
    FundamentalRankerConfig,
)
from quantagent.data.providers.akshare_valuation_provider import AkShareSectorProvider
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.sector import (
    SectorMapBuilder,
    SectorMapConfig,
    SectorPoolBuilder,
    SectorPoolConfig,
    StFlagBuilder,
    StFlagConfig,
    board_proxy_rows,
    normalize_sector_source,
    validate_sector_map,
    validate_st_table,
)
from quantagent.diagnostics.sector_audit import build_sector_audit, write_sector_audit


def _symbols_from_market_panel(path: Path | None) -> tuple[str, ...]:
    if path is None or not path.exists():
        return ()
    frame = pd.read_parquet(path, columns=["symbol"])
    return tuple(dict.fromkeys(frame["symbol"].dropna().astype(str).tolist()))


@app.command("import-sector-source-v7")
def import_sector_source_v7(
    input_path: Path = typer.Option(..., "--input", help="Manual/vendor sector CSV/parquet."),
    output: Path = typer.Option(default_v7_lake_root() / "bronze" / "sector_sources" / "manual_sector.parquet", "--output"),
    source: str = typer.Option("manual_vendor_sector", "--source"),
    source_version: str = typer.Option("unknown", "--source-version"),
    as_of_date: str | None = typer.Option(None, "--as-of-date"),
    fetched_at: str | None = typer.Option(None, "--fetched-at"),
) -> None:
    """Normalize a legal local sector source into bronze canonical schema."""

    frame = read_frame(input_path)
    normalized = normalize_sector_source(
        frame,
        source=source,
        source_version=source_version,
        as_of_date=as_of_date,
        fetched_at=fetched_at,
    )
    written = write_frame(normalized, output)
    typer.echo(json_dump({"status": "passed", "rows": int(len(normalized)), "output": written}))


@app.command("build-sector-map-v7")
def build_sector_map_v7(
    symbols: str = typer.Option("", "--symbols", help="Comma-separated symbols."),
    symbols_file: Path | None = typer.Option(None, "--symbols-file", help="Optional one-symbol-per-line universe."),
    market_panel: Path | None = typer.Option(None, "--market-panel", help="Optional market panel to derive symbols."),
    local_mapping: Path | None = typer.Option(None, "--local-mapping", help="Local sector CSV/parquet."),
    include_board_proxy: bool = typer.Option(False, "--include-board-proxy", help="Add market-segment fallback rows labelled board_proxy."),
    output_root: Path = typer.Option(default_v7_lake_root(), "--output-root"),
    as_of_date: str | None = typer.Option(None, "--as-of-date"),
    allow_network: bool = typer.Option(False, "--allow-network"),
) -> None:
    """Build PIT-labelled sector_map.parquet plus coverage reports.

    Without ``--local-mapping`` or ``--allow-network`` this command still
    writes a full missing-coverage map for the requested universe. That is
    intentional: downstream diagnostics can fail closed instead of
    silently backfilling future industry labels.
    """

    resolved_symbols = tuple(dict.fromkeys((*merge_symbols(symbols, symbols_file), *_symbols_from_market_panel(market_panel))))
    source_frame: pd.DataFrame | None = None
    source = "manual_vendor_sector"
    source_version = "unknown"
    coverage_status = "pit_historical"
    if local_mapping is not None:
        source_frame = read_frame(local_mapping)
        source = "manual_vendor_sector"
    elif allow_network:
        provider = AkShareSectorProvider(allow_network=True)
        result = provider.industry_classification(
            ProviderRequest("", as_of_date or "", symbols=resolved_symbols),
            as_of_date=as_of_date,
        )
        source_frame = result.frame.rename(columns={"industry": "sector_level_1"})
        source_frame["sector_level_2"] = source_frame["sector_level_1"]
        source_frame["coverage_status"] = "current_snapshot"
        source = result.source
        source_version = "akshare_current_snapshot"
        coverage_status = "current_snapshot"
    if include_board_proxy:
        proxy = board_proxy_rows(resolved_symbols, as_of_date=as_of_date)
        source_frame = pd.concat([source_frame, proxy], ignore_index=True) if source_frame is not None else proxy

    config = SectorMapConfig(
        symbols=resolved_symbols,
        as_of_date=as_of_date,
        source=source,
        source_version=source_version,
        coverage_status=coverage_status,
        output_root=output_root,
    )
    builder = SectorMapBuilder(config)
    built = builder.build(source_frame)
    written = builder.write(built)
    typer.echo(json_dump({"status": written.validation["status"], "coverage": written.coverage, "paths": written.output_paths}))


@app.command("validate-sector-map-v7")
def validate_sector_map_v7(
    sector_map: Path = typer.Option(default_v7_lake_root() / "silver" / "sector_map" / "sector_map.parquet", "--sector-map"),
    symbols: str = typer.Option("", "--symbols"),
    symbols_file: Path | None = typer.Option(None, "--symbols-file"),
) -> None:
    frame = read_frame(sector_map)
    report = validate_sector_map(frame, symbols=merge_symbols(symbols, symbols_file))
    typer.echo(json_dump(report))


@app.command("build-st-flags-v7")
def build_st_flags_v7(
    local_mapping: Path | None = typer.Option(None, "--local-mapping", help="Local ST CSV/parquet."),
    symbols: str = typer.Option("", "--symbols"),
    symbols_file: Path | None = typer.Option(None, "--symbols-file"),
    market_panel: Path | None = typer.Option(None, "--market-panel", help="Optional market panel to derive symbols."),
    output_root: Path = typer.Option(default_v7_lake_root(), "--output-root"),
    as_of_date: str | None = typer.Option(None, "--as-of-date"),
    st_block_weight: float = typer.Option(0.90, "--st-block-weight"),
    unknown_st_block_weight: float = typer.Option(0.00, "--unknown-st-block-weight"),
) -> None:
    resolved_symbols = tuple(dict.fromkeys((*merge_symbols(symbols, symbols_file), *_symbols_from_market_panel(market_panel))))
    config = StFlagConfig(
        symbols=resolved_symbols,
        as_of_date=as_of_date,
        output_root=output_root,
        st_block_weight=st_block_weight,
        unknown_st_block_weight=unknown_st_block_weight,
    )
    builder = StFlagBuilder(config)
    result = builder.build(read_frame(local_mapping) if local_mapping else None)
    written = builder.write(result)
    typer.echo(json_dump({"status": written.validation["status"], "coverage": written.coverage, "paths": written.output_paths}))


@app.command("validate-st-flags-v7")
def validate_st_flags_v7(
    st_flags: Path = typer.Option(default_v7_lake_root() / "silver" / "st_flags" / "st_flags.parquet", "--st-flags"),
    symbols: str = typer.Option("", "--symbols"),
    symbols_file: Path | None = typer.Option(None, "--symbols-file"),
) -> None:
    frame = read_frame(st_flags)
    report = validate_st_table(frame, symbols=merge_symbols(symbols, symbols_file))
    typer.echo(json_dump(report))


@app.command("run-sector-audit-v7")
def run_sector_audit_v7(
    target_weights: Path = typer.Option(..., "--target-weights", help="Wide or long target_weights parquet/csv."),
    sector_map: Path | None = typer.Option(None, "--sector-map"),
    sector_manifest: Path | None = typer.Option(None, "--sector-manifest"),
    st_flags: Path | None = typer.Option(None, "--st-flags"),
    st_manifest: Path | None = typer.Option(None, "--st-manifest"),
    market_panel: Path | None = typer.Option(None, "--market-panel"),
    output_dir: Path = typer.Option(Path("runtime/reports/sector_audit"), "--output-dir"),
) -> None:
    """Write post-hoc sector/board/ST exposure audit artifacts.

    This command is diagnostics-only. It reads manifests to report gate
    status and must not feed sector/ST data into target weight generation.
    """

    weights = read_frame(target_weights)
    sector = read_frame(sector_map) if sector_map is not None and sector_map.exists() else None
    st = read_frame(st_flags) if st_flags is not None and st_flags.exists() else None
    market = read_frame(market_panel) if market_panel is not None and market_panel.exists() else None
    audit = build_sector_audit(
        weights,
        sector_map=sector,
        st_flags=st,
        market_panel=market,
        sector_manifest=sector_manifest,
        st_manifest=st_manifest,
    )
    paths = write_sector_audit(audit, output_dir)
    typer.echo(json_dump({"status": "passed", "gate_status": audit["gate_status"], "paths": paths}))


@app.command("build-fundamental-ranker-v7")
def build_fundamental_ranker_v7(
    metrics_path: Path = typer.Option(
        ...,
        "--metrics",
        help=(
            "Metrics frame parquet/csv. Required columns: symbol, available_at. "
            "Optional metric columns: pe_ttm, pb, ps_ttm, roe, gross_margin, "
            "operating_cf_to_net_income, revenue_yoy, net_income_yoy."
        ),
    ),
    sector_map_path: Path | None = typer.Option(
        None,
        "--sector-map",
        help="Stage 2.2 silver/sector_map.parquet. When provided, within-sector ranks use sector_level_1.",
    ),
    as_of_dates: str = typer.Option(
        ...,
        "--as-of-dates",
        help="Comma-separated YYYY-MM-DD as_of dates to score.",
    ),
    output_root: Path = typer.Option(default_v7_lake_root(), "--output-root"),
    min_universe_per_bucket: int = typer.Option(5, "--min-universe-per-bucket"),
    valuation_weight: float = typer.Option(0.40, "--valuation-weight"),
    quality_weight: float = typer.Option(0.35, "--quality-weight"),
    growth_weight: float = typer.Option(0.25, "--growth-weight"),
    source_version: str = typer.Option("metrics_join", "--source-version"),
) -> None:
    """Score symbols across valuation × quality × growth at PIT-safe dates.

    The output is a silver-layer data product. Downstream consumers
    that want to use it as a weight overlay must route through
    ``quantagent.data.fundamental.fundamental_ranker_for_overlay``,
    which enforces the manifest gate (composite coverage ≥ 50% and
    real-sector share ≥ 30%).
    """

    metrics = read_frame(metrics_path)
    sector_map = read_frame(sector_map_path) if sector_map_path is not None and sector_map_path.exists() else None
    dates = [d.strip() for d in as_of_dates.split(",") if d.strip()]
    if not dates:
        raise typer.BadParameter("--as-of-dates must contain at least one YYYY-MM-DD value")

    config = FundamentalRankerConfig(
        min_universe_per_bucket=min_universe_per_bucket,
        dimension_weights={
            "valuation": float(valuation_weight),
            "quality": float(quality_weight),
            "growth": float(growth_weight),
        },
        source_version=source_version,
        output_root=output_root,
    )
    builder = FundamentalRankerBuilder(config)
    result = builder.build(metrics, as_of_dates=dates, sector_map=sector_map)
    written = builder.write(result)
    typer.echo(
        json_dump(
            {
                "status": written.validation["status"],
                "coverage": written.coverage,
                "paths": written.output_paths,
            }
        )
    )


@app.command("build-sector-pool-v7")
def build_sector_pool_v7(
    ic_report: Path = typer.Option(
        ...,
        "--ic-report",
        help=(
            "Path to a stratified-IC sector table. Accepts either "
            "the JSON produced by stratified_ic_report.py "
            "(tables.sector_level_1) or a CSV/parquet with columns "
            "[bucket, horizon, ic_mean, ic_std, ic_ir, n_dates, n_symbols]."
        ),
    ),
    output_root: Path = typer.Option(default_v7_lake_root(), "--output-root"),
    reference_horizon: int = typer.Option(20, "--reference-horizon"),
    min_dates: int = typer.Option(60, "--min-dates"),
    min_symbols: int = typer.Option(20, "--min-symbols"),
    core_quantile: float = typer.Option(0.30, "--core-quantile"),
    core_ir_threshold: float = typer.Option(0.30, "--core-ir-threshold"),
    watch_ir_threshold: float = typer.Option(0.10, "--watch-ir-threshold"),
    short_term_vol_threshold: float = typer.Option(0.10, "--short-term-vol-threshold"),
    source_version: str = typer.Option("ic_report", "--source-version"),
) -> None:
    """Build silver/sector_pool.parquet from a stratified-IC sector table.

    The pool is **diagnostic** by default — it materialises a tiered
    sector list but does not change optimiser weights. To consume it as
    an overlay use ``quantagent.data.sector.sector_pool_for_weight_overlay``
    which enforces the manifest gate.
    """

    if ic_report.suffix == ".json":
        payload = json.loads(ic_report.read_text(encoding="utf-8"))
        sector_rows = payload.get("tables", {}).get("sector_level_1", [])
        if not sector_rows:
            raise typer.BadParameter(f"no sector_level_1 table in {ic_report}")
        ic_table = pd.DataFrame(sector_rows)
    else:
        ic_table = read_frame(ic_report)

    config = SectorPoolConfig(
        reference_horizon=reference_horizon,
        min_dates=min_dates,
        min_symbols=min_symbols,
        core_quantile=core_quantile,
        core_ir_threshold=core_ir_threshold,
        watch_ir_threshold=watch_ir_threshold,
        short_term_vol_threshold=short_term_vol_threshold,
        source_version=source_version,
        output_root=output_root,
    )
    builder = SectorPoolBuilder(config)
    written = builder.write(builder.build(ic_table))
    typer.echo(
        json_dump(
            {
                "status": written.validation["status"],
                "coverage": written.coverage,
                "paths": written.output_paths,
            }
        )
    )


if __name__ == "__main__":  # pragma: no cover - direct module CLI
    app()
