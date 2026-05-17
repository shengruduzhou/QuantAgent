"""Daily paper-loop CLI commands."""

from __future__ import annotations

from pathlib import Path
import time

import typer

from quantagent.cli._utils import app, json_dump

paper_app = typer.Typer(help="Daily V7 paper trading loop commands.")

@paper_app.command("run-once")
@app.command("paper-run-once")
def paper_run_once(
    date: str = typer.Option("today", "--date"),
    model_dir: Path | None = typer.Option(None, "--model-dir"),
    feature_dataset: Path | None = typer.Option(None, "--feature-dataset"),
    market_panel: Path | None = typer.Option(None, "--market-panel"),
    sector_map: Path | None = typer.Option(None, "--sector-map"),
    output_root: Path | None = typer.Option(None, "--output-root"),
    primary_horizon: int = typer.Option(5, "--primary-horizon"),
    top_k: int = typer.Option(30, "--top-k"),
    selection_mode: str = typer.Option("ai_threshold", "--selection-mode", help="ai_threshold | top_k"),
    alpha_threshold: float = typer.Option(0.0, "--alpha-threshold"),
    confidence_floor: float = typer.Option(0.55, "--confidence-floor"),
    selection_top_k_min: int = typer.Option(5, "--selection-top-k-min"),
    selection_top_k_max: int = typer.Option(100, "--selection-top-k-max"),
    min_order_value_yuan: float = typer.Option(100.0, "--min-order-value-yuan"),
) -> None:
    """Run one safe daily paper iteration and write target weights + HTML report."""
    from quantagent.paper.daily_loop import DailyPaperLoopConfig, run_once

    cfg = DailyPaperLoopConfig(
        as_of_date=date,
        model_dir=str(model_dir) if model_dir else DailyPaperLoopConfig(as_of_date=date).model_dir,
        feature_dataset_path=str(feature_dataset) if feature_dataset else DailyPaperLoopConfig(as_of_date=date).feature_dataset_path,
        market_panel_path=str(market_panel) if market_panel else DailyPaperLoopConfig(as_of_date=date).market_panel_path,
        sector_map_path=str(sector_map) if sector_map else None,
        output_root=str(output_root) if output_root else DailyPaperLoopConfig(as_of_date=date).output_root,
        primary_horizon=primary_horizon,
        top_k=top_k,
        selection_mode=selection_mode,
        alpha_threshold=alpha_threshold,
        confidence_floor=confidence_floor,
        selection_top_k_min=selection_top_k_min,
        selection_top_k_max=selection_top_k_max,
        min_order_value_yuan=min_order_value_yuan,
    )
    typer.echo(json_dump(run_once(cfg).to_dict()))


@paper_app.command("run-loop")
@app.command("paper-run-loop")
def paper_run_loop(
    interval_seconds: int = typer.Option(86_400, "--interval-seconds"),
    date: str = typer.Option("today", "--date"),
) -> None:
    """Minimal restartable paper loop.

    Use an external scheduler/systemd for exact market-time execution on
    production servers; this command keeps all state in repo runtime.
    """
    from quantagent.paper.daily_loop import DailyPaperLoopConfig, run_once

    while True:
        typer.echo(json_dump(run_once(DailyPaperLoopConfig(as_of_date=date)).to_dict()))
        time.sleep(max(60, int(interval_seconds)))


@paper_app.command("reflect-and-retrain")
@app.command("paper-reflect-and-retrain")
def paper_reflect_and_retrain(
    dataset: Path = typer.Option(..., "--dataset"),
    window: str = typer.Option("7d", "--window"),
    n_trials: int = typer.Option(10, "--n-trials"),
    generations: int = typer.Option(3, "--generations"),
    timesteps: int = typer.Option(50_000, "--timesteps"),
    require_gpu: bool = typer.Option(True, "--require-gpu/--no-require-gpu"),
) -> None:
    """Trigger a compact autopilot retrain after a paper-performance check."""
    from quantagent.cli.v7_train import _run_autopilot_impl

    result = _run_autopilot_impl(
        dataset_path=dataset,
        market_panel_path=None,
        predictions_path=None,
        n_trials=n_trials,
        generations=generations,
        timesteps=timesteps,
        study_name=f"reflect_{window}",
        require_gpu=require_gpu,
    )
    typer.echo(json_dump({"window": window, **result}))


app.add_typer(paper_app, name="paper")
