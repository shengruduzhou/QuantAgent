"""Daily and continuous paper-loop CLI commands."""

from __future__ import annotations

from pathlib import Path
import time

import typer

from quantagent.cli._utils import app, json_dump, read_frame

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
    initial_cash: float = typer.Option(1_000_000.0, "--initial-cash", min=0.01),
    portfolio_id: str = typer.Option("v7-paper", "--portfolio-id"),
    primary_horizon: int = typer.Option(5, "--primary-horizon"),
    top_k: int = typer.Option(30, "--top-k"),
    selection_mode: str = typer.Option("ai_threshold", "--selection-mode", help="ai_threshold | top_k"),
    alpha_threshold: float = typer.Option(0.0, "--alpha-threshold"),
    confidence_floor: float = typer.Option(0.55, "--confidence-floor"),
    selection_top_k_min: int = typer.Option(5, "--selection-top-k-min"),
    selection_top_k_max: int = typer.Option(100, "--selection-top-k-max"),
    min_order_value_yuan: float = typer.Option(100.0, "--min-order-value-yuan"),
) -> None:
    """Run one safe daily paper iteration and freeze target weights.

    ``portfolio_id`` and ``initial_cash`` are immutable account-genesis fields.
    The first worker persists them under ``QUANTAGENT_HOME/paper``; every later
    target/execution worker must pass the same values or the account fails closed.
    """
    from quantagent.paper.daily_loop import DailyPaperLoopConfig, run_once

    defaults = DailyPaperLoopConfig(as_of_date=date)
    cfg = DailyPaperLoopConfig(
        as_of_date=date,
        model_dir=str(model_dir) if model_dir else defaults.model_dir,
        feature_dataset_path=str(feature_dataset) if feature_dataset else defaults.feature_dataset_path,
        market_panel_path=str(market_panel) if market_panel else defaults.market_panel_path,
        sector_map_path=str(sector_map) if sector_map else None,
        output_root=str(output_root) if output_root else defaults.output_root,
        account_identity_path=defaults.account_identity_path,
        portfolio_id=portfolio_id,
        initial_cash=initial_cash,
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


@paper_app.command("execute-session")
@app.command("paper-execute-session")
def paper_execute_session(
    date: str = typer.Option("today", "--date"),
    market_panel: Path | None = typer.Option(None, "--market-panel"),
    initial_cash: float = typer.Option(1_000_000.0, "--initial-cash", min=0.01),
    portfolio_id: str = typer.Option("v7-paper", "--portfolio-id"),
    execution_clock: str = typer.Option("14:59:00+08:00", "--execution-clock"),
    max_participation_rate: float = typer.Option(0.05, "--max-participation-rate", min=0.0, max=1.0),
    min_order_value_yuan: float = typer.Option(100.0, "--min-order-value-yuan", min=0.0),
) -> None:
    """Consume frozen targets on one observed session using the canonical paper account.

    This command intentionally does **not** accept a hand-written session list as
    authoritative evidence. It consumes only sessions actually present in the
    supplied market panel and writes every account/journal/idempotency artifact
    under ``QUANTAGENT_HOME/paper`` so the execution worker, API and UI share the
    same source of truth. An observed panel remains non-certifying calendar
    evidence until the authoritative-calendar gate is implemented.

    The requested ``portfolio_id``/``initial_cash`` must exactly match the
    immutable account identity created by the target worker. Cross-process
    serialization is owned by ``execute_pending_for_session`` itself, not by this
    CLI adapter, so direct service callers cannot bypass the account boundary.
    """
    from quantagent.paper.continuous_execution import (
        ContinuousPaperExecutionConfig,
        execute_pending_for_session,
    )
    from quantagent.paper.daily_loop import DailyPaperLoopConfig
    from quantagent.paper.runtime_paths import paper_runtime_paths

    defaults = DailyPaperLoopConfig(as_of_date=date)
    market_path = Path(market_panel) if market_panel else Path(defaults.market_panel_path)
    frame = read_frame(market_path)
    paths = paper_runtime_paths().ensure()
    config = ContinuousPaperExecutionConfig(
        pending_signal_dir=str(paths.pending_signals),
        execution_journal_path=str(paths.execution_journal),
        canonical_ledger_path=str(paths.canonical_ledger),
        operational_ledger_path=str(paths.operational_ledger),
        idempotency_path=str(paths.idempotency),
        account_identity_path=str(paths.account_identity),
        portfolio_id=portfolio_id,
        initial_cash=initial_cash,
        min_order_value_yuan=min_order_value_yuan,
        max_participation_rate=max_participation_rate,
        execution_clock=execution_clock,
    )
    results = execute_pending_for_session(
        date,
        frame,
        config=config,
        authoritative_sessions=None,
    )
    typer.echo(
        json_dump(
            {
                "date": date,
                "marketPanel": str(market_path),
                "runtime": paths.as_dict(),
                "paperAccount": {
                    "portfolioId": portfolio_id,
                    "initialCash": initial_cash,
                    "identityPath": str(paths.account_identity),
                },
                "calendarAssurance": "observed_market_panel_only",
                "shadowAcceptanceCalendarEligible": False,
                "results": [result.to_dict() for result in results],
            }
        )
    )


@paper_app.command("run-loop")
@app.command("paper-run-loop")
def paper_run_loop(
    interval_seconds: int = typer.Option(86_400, "--interval-seconds"),
    date: str = typer.Option("today", "--date"),
    initial_cash: float = typer.Option(1_000_000.0, "--initial-cash", min=0.01),
    portfolio_id: str = typer.Option("v7-paper", "--portfolio-id"),
) -> None:
    """Minimal restartable target-generation loop.

    Use an external scheduler/systemd for exact market-time execution on
    production servers. This command freezes targets; ``execute-session`` is
    the separate next-session consumer so target construction can never be
    mistaken for a completed paper fill. Account genesis values are immutable
    and must match the same values used by ``execute-session``.
    """
    from quantagent.paper.daily_loop import DailyPaperLoopConfig, run_once

    while True:
        typer.echo(
            json_dump(
                run_once(
                    DailyPaperLoopConfig(
                        as_of_date=date,
                        portfolio_id=portfolio_id,
                        initial_cash=initial_cash,
                    )
                ).to_dict()
            )
        )
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