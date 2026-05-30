"""Stage 4.3 — bond-market flow CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from quantagent.cli._utils import app, default_v7_lake_root, json_dump, read_frame
from quantagent.data.bond import BondFlowBuilder, BondFlowConfig


@app.command("import-bond-flows-v7")
def import_bond_flows_v7(
    input_path: Path = typer.Option(
        ...,
        "--input",
        help=(
            "Bond flows CSV/parquet. Required col: trade_date. "
            "Optional yields: yield_1y/5y/10y/3m/aa/aaa, dr007, bond_fund_flow."
        ),
    ),
    output_root: Path = typer.Option(default_v7_lake_root(), "--output-root"),
    source: str = typer.Option("manual_local_import", "--source"),
    source_version: str = typer.Option("unknown", "--source-version"),
    min_days: int = typer.Option(30, "--min-days"),
    min_field_coverage: float = typer.Option(0.50, "--min-field-coverage"),
    min_date_continuity: float = typer.Option(0.95, "--min-date-continuity"),
) -> None:
    """Normalise a local bond-market CSV into silver/bond_flows.parquet."""

    raw = read_frame(input_path)
    config = BondFlowConfig(
        source=source,
        source_version=source_version,
        output_root=output_root,
        min_days=min_days,
        min_field_coverage=min_field_coverage,
        min_date_continuity=min_date_continuity,
    )
    builder = BondFlowBuilder(config)
    result = builder.write(builder.build(raw))
    typer.echo(
        json_dump(
            {
                "status": result.validation["status"],
                "coverage": result.coverage,
                "paths": result.output_paths,
            }
        )
    )


if __name__ == "__main__":  # pragma: no cover
    app()
