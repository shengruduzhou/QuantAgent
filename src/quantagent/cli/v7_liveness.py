"""Step 2.5 liveness and training-status commands."""

from __future__ import annotations

from pathlib import Path

import typer

from quantagent.cli._utils import app, json_dump, read_frame


@app.command("scan-v10-training-status-v7")
def scan_v10_training_status_v7(
    output_dir: Path = typer.Option(Path("runtime/reports/v10_training_status"), "--output-dir"),
    base_output: Path = typer.Option(Path("runtime/models/v7_alpha_full_universe_nosynth_v10"), "--base-output"),
    log_dir: Path = typer.Option(Path("runtime/logs"), "--log-dir"),
    expected_folds: int = typer.Option(12, "--expected-folds"),
) -> None:
    """Write v10 fold/process reconciliation report."""

    from quantagent.diagnostics.training_status import V10StatusConfig, scan_v10_training_status, write_training_status

    status = scan_v10_training_status(
        V10StatusConfig(base_output=base_output, log_dir=log_dir, expected_folds=expected_folds)
    )
    paths = write_training_status(status, output_dir)
    typer.echo(json_dump({"status": "passed", "aggregate_ready": status["aggregate_ready"], "paths": paths}))


@app.command("resume-v10-seed-v7")
def resume_v10_seed_v7(
    seed: int = typer.Option(..., "--seed"),
    dry_run: bool = typer.Option(True, "--dry-run/--run"),
    output_dir: Path = typer.Option(Path("runtime/reports/v10_training_status"), "--output-dir"),
) -> None:
    """Print the safe resume command for an interrupted v10 seed.

    The trainer now skips fully completed fold directories, so rerunning a
    seed resumes from the first missing fold instead of overwriting
    complete folds. This command stays dry-run by default.
    """

    from quantagent.diagnostics.training_status import build_resume_command, scan_v10_training_status, write_training_status

    status = scan_v10_training_status()
    write_training_status(status, output_dir)
    seed_status = status["seeds"].get(str(seed), {})
    command = " ".join(build_resume_command(seed))
    payload = {
        "status": "dry_run" if dry_run else "not_executed",
        "seed": int(seed),
        "completed_folds": seed_status.get("completed_folds", []),
        "missing_folds": seed_status.get("missing_folds", []),
        "resume_from": seed_status.get("resume_from"),
        "completed_fold_overwrite_policy": "skip_completed_folds",
        "command": command,
    }
    if not dry_run:
        raise typer.BadParameter("resume execution is intentionally not launched from this CLI; run the printed command explicitly")
    typer.echo(json_dump(payload))


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
