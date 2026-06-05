"""Live-readiness and target-weights health commands."""

from __future__ import annotations

from pathlib import Path

import typer

from quantagent.cli._utils import app, json_dump, read_frame


@app.command("target-weights-liveness-v7")
def target_weights_liveness_v7(
    target_weights: Path = typer.Option(..., "--target-weights"),
    predictions: Path | None = typer.Option(None, "--predictions"),
    diagnostics: Path | None = typer.Option(None, "--diagnostics"),
    output_dir: Path = typer.Option(Path("runtime/reports/target_weights_liveness"), "--output-dir"),
) -> None:
    """Write target_weights liveness artifacts."""

    from quantagent.diagnostics.target_liveness import (
        build_target_weights_liveness,
        load_diagnostics,
        write_target_weights_liveness,
    )

    report = build_target_weights_liveness(
        read_frame(target_weights),
        predictions=read_frame(predictions) if predictions is not None and predictions.exists() else None,
        diagnostics=load_diagnostics(diagnostics),
    )
    paths = write_target_weights_liveness(report, output_dir)
    typer.echo(json_dump({"status": report["summary"]["status"], "summary": report["summary"], "paths": paths}))


@app.command("health-check-v7")
def health_check_v7(
    lake_root: Path = typer.Option(Path("runtime/data/v7"), "--lake-root", help="Data lake root."),
    output_root: Path = typer.Option(Path("runtime/reports/daily_health"), "--output-root"),
    no_write: bool = typer.Option(False, "--no-write", help="Print report without writing files."),
) -> None:
    """Check the five data-layer manifest gates and emit a health report.

    Exit code: 0=OK, 1=WARN, 2=FAIL.  systemd OnFailure can key off this.
    """

    from quantagent.diagnostics.daily_health import DailyHealthChecker, DailyHealthConfig

    config = DailyHealthConfig(lake_root=lake_root, output_root=output_root)
    checker = DailyHealthChecker(config)
    report = checker.run(write=not no_write)
    typer.echo(report.to_markdown())
    if not no_write:
        typer.echo(f"\nWrote reports to: {config.reports_dir}")
    raise typer.Exit(code=report.exit_code)


if __name__ == "__main__":  # pragma: no cover
    app()
