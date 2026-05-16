"""V7 readiness CLI: live-readiness gate report and pipeline validation."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from quantagent.cli._utils import app, json_dump


@app.command("v7-live-readiness-report")
def v7_live_readiness_report(
    metrics_path: Path = typer.Option(..., "--metrics"),
    paper_report: Path = typer.Option(..., "--paper-report"),
    output_path: Path = typer.Option(Path("reports/v7/live_readiness_report.json"), "--output"),
) -> None:
    """Evaluate live-readiness gates without enabling live trading."""
    from quantagent.data.v7_quality_gates import evaluate_model_acceptance_gates

    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    report = evaluate_model_acceptance_gates(metrics, paper_report_path=paper_report).to_dict()
    report["safety_defaults"] = {"live_trading_enabled": False, "dry_run": True, "virtual_broker_only": True}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_dump(report), encoding="utf-8")
    typer.echo(json_dump(report))


@app.command("validate-v7")
def validate_v7(
    config: Path = Path("configs/v7.default.yaml"),
) -> None:
    """Validate a V7 config and report safety/data-quality status."""
    from quantagent.services.v7_pipeline_service import validate_v7 as service_validate_v7

    typer.echo(json_dump(service_validate_v7(config)))


@app.command("run-daily-v7")
def run_daily_v7(
    config: Path = Path("configs/v7.default.yaml"),
    date: str = typer.Option("2026-05-15", "--date"),
    output_dir: Path = Path("reports/v7"),
) -> None:
    """Run the daily V7 research orchestrator end-to-end against a config (mock-friendly)."""
    from quantagent.services.v7_pipeline_service import run_daily_v7_research

    result = run_daily_v7_research(config, as_of_date=date)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "v7_daily_research_report.json"
    output_path.write_text(json_dump(result), encoding="utf-8")
    typer.echo(
        f"status=ok themes={len(result['theme_ranking'])} "
        f"targets={len(result['portfolio_plan']['target_weights'])} output={output_path}"
    )
