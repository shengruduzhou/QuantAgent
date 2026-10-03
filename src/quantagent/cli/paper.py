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
    sector_map: Path | None = typer.Option(
        None, "--sector-map",
        help="symbol/industry table for the venue's industry limit. The current "
             "snapshot is point-in-time for forward paper. Without it every BUY is "
             "refused industry_unmeasured unless --max-industry-weight 1.0.",
    ),
    max_industry_weight: float = typer.Option(
        0.30, "--max-industry-weight", min=0.0, max=1.0,
        help="Venue industry concentration limit; 1.0 is the explicit opt-out.",
    ),
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
    from quantagent.paper.risk import RiskLimits
    from quantagent.paper.runtime_paths import paper_runtime_paths

    defaults = DailyPaperLoopConfig(as_of_date=date)
    market_path = Path(market_panel) if market_panel else Path(defaults.market_panel_path)
    frame = read_frame(market_path)
    paths = paper_runtime_paths().ensure()
    config = ContinuousPaperExecutionConfig(
        sector_map_path=str(sector_map) if sector_map else None,
        # Production defaults (pre-trade participation at a full bar: the
        # venue's participation cap meters fills) with the industry limit the
        # operator chose; 1.0 is an explicit, visible opt-out.
        risk_limits=RiskLimits(max_participation=1.0, max_industry_weight=max_industry_weight),
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
                "riskLimits": config.risk_limits.to_dict(),
                "sectorMap": str(sector_map) if sector_map else None,
                "calendarAssurance": "observed_market_panel_only",
                "shadowAcceptanceCalendarEligible": False,
                "results": [result.to_dict() for result in results],
            }
        )
    )


@paper_app.command("bind-legacy-terminal")
@app.command("paper-bind-legacy-terminal")
def paper_bind_legacy_terminal(
    pending_payload_sha256: str = typer.Option(..., "--pending-payload-sha256"),
    date: str = typer.Option("today", "--date"),
    reason: str = typer.Option(..., "--reason"),
    initial_cash: float = typer.Option(1_000_000.0, "--initial-cash", min=0.01),
    portfolio_id: str = typer.Option("v7-paper", "--portfolio-id"),
) -> None:
    """Append an audited lower-assurance binding for one legacy terminal."""
    from quantagent.paper.continuous_execution import (
        ContinuousPaperExecutionConfig,
        bind_legacy_terminal_account,
    )
    from quantagent.paper.runtime_paths import paper_runtime_paths

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
    )
    typer.echo(
        json_dump(
            bind_legacy_terminal_account(
                config=config,
                pending_payload_sha256=pending_payload_sha256,
                as_of_date=date,
                reason=reason,
            )
        )
    )


@paper_app.command("run-loop")
@app.command("paper-run-loop")
def paper_run_loop(
    interval_seconds: int = typer.Option(86_400, "--interval-seconds"),
    date: str = typer.Option("today", "--date"),
    initial_cash: float = typer.Option(1_000_000.0, "--initial-cash", min=0.01),
    portfolio_id: str = typer.Option("v7-paper", "--portfolio-id"),
    sector_map: Path | None = typer.Option(
        None, "--sector-map",
        help="symbol/industry table; without it the optimiser's sector cap is "
             "published as unenforced (sector_map_absent).",
    ),
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
                        sector_map_path=str(sector_map) if sector_map else None,
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

@app.command("paper-clear-kill-switch")
def paper_clear_kill_switch(
    ledger: Path = typer.Option(..., "--ledger", exists=True, dir_okay=False,
                                help="The account's operational/risk-state ledger (JSONL)."),
    scope: str = typer.Option("PORTFOLIO", "--scope"),
    key: str | None = typer.Option(None, "--key"),
    author: str = typer.Option(..., "--author"),
    reason: str = typer.Option(..., "--reason"),
    confirm: bool = typer.Option(False, "--confirm", help="Required: a human clears a latch."),
) -> None:
    """Clear a latched kill switch after human review (drawdown breaches latch).

    The clear is appended to the same durable ledger the risk engine replays, so
    every venue sees it on its next start. Refused while another process holds
    the ledger directory's writer lock - stop the API / paper loop first, or the
    two writers would fork the ledger's hash chain.
    """
    import fcntl

    from quantagent.paper import ledger as paper_ledger
    from quantagent.paper.risk import RiskEngine

    if not confirm:
        raise typer.BadParameter("pass --confirm: clearing a kill switch is a human decision")
    if not author.strip() or len(reason.strip()) < 8:
        raise typer.BadParameter("--author is required and --reason must be at least 8 characters")
    lock_path = ledger.parent / "writer.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise typer.BadParameter(
                f"{lock_path} is held by a running writer; stop it before clearing"
            ) from exc
        engine = RiskEngine(state_ledger=paper_ledger.EventLedger(ledger))
        active_before = engine.kill_switch.active()
        cleared = engine.kill_switch.clear(
            scope.upper(), key, human_confirmation=True,
            author=author.strip(), reason=reason.strip(),
        )
    typer.echo(json_dump({
        "cleared": cleared,
        "scope": scope.upper(),
        "key": key,
        "activeBefore": active_before,
        "activeAfter": engine.kill_switch.active(),
    }))
    if not cleared:
        raise typer.Exit(code=1)
