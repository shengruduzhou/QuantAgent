"""Stage 4 — policy event CLI commands."""

from __future__ import annotations

from pathlib import Path

import typer

from quantagent.cli._utils import app, default_v7_lake_root, json_dump, read_frame
from quantagent.data.policy import (
    PolicyEventBuilder,
    PolicyEventConfig,
)


@app.command("import-policy-events-v7")
def import_policy_events_v7(
    input_path: Path = typer.Option(
        ...,
        "--input",
        help=(
            "Raw policy CSV/parquet. Required cols: source, announced_at, title. "
            "Optional: url, body_summary, effective_at, fetched_at, "
            "themes_override, sectors_hint_override."
        ),
    ),
    output_root: Path = typer.Option(default_v7_lake_root(), "--output-root"),
    source_version: str = typer.Option("manual_local_import", "--source-version"),
    min_events: int = typer.Option(5, "--min-events"),
    min_theme_coverage: float = typer.Option(0.50, "--min-theme-coverage"),
    min_strength_median: float = typer.Option(0.30, "--min-strength-median"),
) -> None:
    """Normalise a local policy CSV/parquet into silver/policy_events.parquet.

    The command never crawls. Use it to ingest a vendor-supplied or
    manually-curated CSV of policy announcements; theme + sector tagging
    happens automatically via the rule-based tagger.
    """

    raw = read_frame(input_path)
    config = PolicyEventConfig(
        source_version=source_version,
        output_root=output_root,
        min_events=min_events,
        min_theme_coverage=min_theme_coverage,
        min_strength_median=min_strength_median,
    )
    builder = PolicyEventBuilder(config)
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
