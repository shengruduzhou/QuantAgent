"""V7 backtest CLI: walk-forward backtest, paper trade, sleeve walk-forward allocator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from quantagent.cli._utils import app, default_reports_root, json_dump, read_frame, write_frame


@app.command("walk-forward-backtest-v7")
def walk_forward_backtest_v7(
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    output_path: Path = typer.Option(None, "--output"),
    target_weights_path: Path | None = typer.Option(
        None, "--target-weights", help="Optional explicit target_weights frame; mutually exclusive with --predictions."
    ),
    predictions_path: Path | None = typer.Option(
        None, "--predictions", help="If provided, run the V7 target-weights optimiser before backtest."
    ),
    sector_map_path: Path | None = typer.Option(None, "--sector-map"),
    top_k: int = typer.Option(30, "--top-k"),
    top_k_ratio: float | None = typer.Option(0.10, "--top-k-ratio"),
    min_selection_pressure: float = typer.Option(3.0, "--min-selection-pressure"),
    fail_if_top_k_covers_universe: bool = typer.Option(
        True,
        "--fail-if-top-k-covers-universe/--allow-top-k-covers-universe",
    ),
    max_weight_per_name: float = typer.Option(0.10, "--max-weight"),
    max_sector_weight: float = typer.Option(0.30, "--max-sector"),
    max_turnover: float = typer.Option(0.50, "--max-turnover"),
    long_short: bool = typer.Option(False, "--long-short/--long-only"),
    horizon_column: str | None = typer.Option(None, "--horizon-column"),
    optimizer_backend: str = typer.Option("auto", "--optimizer-backend", help="auto | deterministic | cvxpy"),
    objective: str = typer.Option("max_expected_alpha", "--objective"),
    cash_floor: float = typer.Option(0.0, "--cash-floor"),
) -> None:
    """Replay weights through the A-share OrderManager dry-run execution simulator.

    Accepts either an explicit ``--target-weights`` frame or
    ``--predictions``. When predictions are provided, the V7 target
    weights optimiser is invoked first to build the weight panel.
    """
    from quantagent.backtest.ashare_execution_simulator import simulate_ashare_target_weights

    if (target_weights_path is None) == (predictions_path is None):
        raise typer.BadParameter("provide exactly one of --target-weights or --predictions")

    if predictions_path is not None:
        from quantagent.portfolio.v7_target_weights import (
            V7TargetWeightsConfig,
            build_v7_target_weights,
        )

        weights_result = build_v7_target_weights(
            read_frame(predictions_path),
            read_frame(market_panel_path),
            sector_map=read_frame(sector_map_path) if sector_map_path else None,
            config=V7TargetWeightsConfig(
                long_short=long_short,
                top_k=top_k,
                top_k_ratio=top_k_ratio,
                min_selection_pressure=min_selection_pressure,
                fail_if_top_k_covers_universe=fail_if_top_k_covers_universe,
                max_weight_per_name=max_weight_per_name,
                max_sector_weight=max_sector_weight,
                max_turnover=max_turnover,
                horizon_column=horizon_column,
                optimizer_backend=optimizer_backend,
                objective=objective,
                cash_floor=cash_floor,
            ),
        )
        weights = weights_result.target_weights
        optimizer_diagnostics: dict | None = dict(weights_result.diagnostics)
    else:
        weights = read_frame(target_weights_path)
        optimizer_diagnostics = None

    if "trade_date" in weights.columns:
        weights = weights.set_index("trade_date")
    result = simulate_ashare_target_weights(weights, read_frame(market_panel_path))
    resolved_output = (
        Path(output_path)
        if output_path is not None
        else default_reports_root() / "walk_forward_backtest.json"
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "nav": result.nav.to_dict(),
        "orders": result.order_audit.to_dict("records"),
        "failed_orders": result.failed_order_audit.to_dict("records"),
        "skipped_orders": result.skipped_order_audit.to_dict("records"),
        "holdings": result.position_history.to_dict("records"),
        "config": result.config,
    }
    if optimizer_diagnostics is not None:
        payload["optimizer_diagnostics"] = optimizer_diagnostics
    resolved_output.write_text(json_dump(payload), encoding="utf-8")
    typer.echo(f"status=ok output={resolved_output} failed_orders={len(result.failed_order_audit)}")


@app.command("run-paper-backtest-v7")
def run_paper_backtest_v7(
    target_weights_path: Path = typer.Option(..., "--target-weights"),
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    output_dir: Path = typer.Option(None, "--output-dir"),
    initial_cash: float = typer.Option(1_000_000.0, "--initial-cash"),
    benchmark_symbol: str | None = typer.Option(None, "--benchmark-symbol"),
    lot_size: int = typer.Option(100, "--lot-size"),
    volume_participation_cap: float = typer.Option(0.10, "--volume-participation-cap"),
    slippage_bps: float = typer.Option(8.0, "--slippage-bps"),
    min_order_value_yuan: float = typer.Option(100.0, "--min-order-value-yuan"),
) -> None:
    """Run A-share constrained paper backtest and write user-facing reports."""
    from quantagent.backtest.ashare_execution_simulator import (
        AShareExecutionSimulationConfig,
        simulate_ashare_target_weights,
    )
    from quantagent.backtest.paper_report import PaperReportConfig, write_paper_report

    weights = read_frame(target_weights_path)
    if "trade_date" in weights.columns:
        weights = weights.set_index("trade_date")
    market = read_frame(market_panel_path)
    resolved_dir = Path(output_dir) if output_dir is not None else default_reports_root() / "paper_report"
    report_weights = weights.reset_index().rename(columns={"index": "trade_date"})
    written_weights = write_frame(report_weights, resolved_dir / "target_weights.parquet")
    sim = simulate_ashare_target_weights(
        weights,
        market,
        AShareExecutionSimulationConfig(
            initial_cash=initial_cash,
            lot_size=lot_size,
            min_order_value_yuan=min_order_value_yuan,
            volume_participation_cap=volume_participation_cap,
            slippage_bps=slippage_bps,
            audit_log_dir=str(resolved_dir / "audit"),
        ),
    )
    report = write_paper_report(
        sim,
        market_panel=market,
        config=PaperReportConfig(
            initial_cash=initial_cash,
            benchmark_symbol=benchmark_symbol,
            slippage_bps=slippage_bps,
            output_dir=resolved_dir,
            target_weights_path=str(written_weights),
        ),
    )
    typer.echo(json_dump(report))


@app.command("generate-paper-report-v7")
def generate_paper_report_v7(
    target_weights_path: Path = typer.Option(..., "--target-weights"),
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    output_dir: Path = typer.Option(None, "--output-dir"),
    initial_cash: float = typer.Option(1_000_000.0, "--initial-cash"),
    benchmark_symbol: str | None = typer.Option(None, "--benchmark-symbol"),
) -> None:
    """Compatibility wrapper for report generation from target weights."""
    run_paper_backtest_v7(
        target_weights_path=target_weights_path,
        market_panel_path=market_panel_path,
        output_dir=output_dir,
        initial_cash=initial_cash,
        benchmark_symbol=benchmark_symbol,
        lot_size=100,
        volume_participation_cap=0.10,
        slippage_bps=8.0,
        min_order_value_yuan=100.0,
    )


@app.command("paper-trade-v7")
def paper_trade_v7(
    target_weights_path: Path = typer.Option(..., "--target-weights"),
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    output_path: Path = typer.Option(None, "--output"),
) -> None:
    """Run the same dry-run VirtualBroker path used by backtests; no live submit is possible here."""
    resolved_output = (
        Path(output_path)
        if output_path is not None
        else default_reports_root() / "paper_trade_report.json"
    )
    walk_forward_backtest_v7(
        market_panel_path=market_panel_path,
        output_path=resolved_output,
        target_weights_path=target_weights_path,
    )


@app.command("walk-forward-v7")
def walk_forward_v7(
    sleeve_returns_path: Path = typer.Option(..., "--sleeve-returns"),
    output_path: Path = typer.Option(None, "--output"),
    grid_step: float = typer.Option(0.05, "--grid-step"),
    embargo_days: int = typer.Option(5, "--embargo-days"),
    walk_forward_splits: int = typer.Option(4, "--splits"),
    drawdown_penalty: float = typer.Option(0.50, "--drawdown-penalty"),
) -> None:
    """Run the walk-forward sleeve allocator on a daily sleeve-returns panel."""
    from quantagent.portfolio.walk_forward_sleeve_allocator import (
        WalkForwardSleeveConfig,
        allocate_sleeves_walk_forward,
    )

    frame = pd.read_csv(sleeve_returns_path)
    if "trade_date" not in frame.columns:
        raise typer.BadParameter("sleeve-returns CSV must contain a trade_date column")
    panel = frame.set_index("trade_date")
    result = allocate_sleeves_walk_forward(
        panel,
        config=WalkForwardSleeveConfig(
            walk_forward_splits=walk_forward_splits,
            embargo_days=embargo_days,
            grid_step=grid_step,
            drawdown_penalty=drawdown_penalty,
        ),
    )
    resolved_output = (
        Path(output_path)
        if output_path is not None
        else default_reports_root() / "walk_forward_sleeve_allocation.json"
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(json_dump(result), encoding="utf-8")
    typer.echo(
        f"status=ok windows={result.diagnostics.get('walk_forward_windows', 0)} "
        f"cash_weight={result.cash_weight:.3f} output={resolved_output}"
    )
