"""V7 backtest CLI: walk-forward backtest, paper trade, sleeve walk-forward allocator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from quantagent.cli._utils import app, json_dump, read_frame


@app.command("walk-forward-backtest-v7")
def walk_forward_backtest_v7(
    target_weights_path: Path = typer.Option(..., "--target-weights"),
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    output_path: Path = typer.Option(Path("reports/v7/walk_forward_backtest.json"), "--output"),
) -> None:
    """Replay target_weights through the A-share OrderManager execution simulator."""
    from quantagent.backtest.ashare_execution_simulator import simulate_ashare_target_weights

    weights = read_frame(target_weights_path)
    if "trade_date" in weights.columns:
        weights = weights.set_index("trade_date")
    result = simulate_ashare_target_weights(weights, read_frame(market_panel_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "nav": result.nav.to_dict(),
        "orders": result.order_audit.to_dict("records"),
        "failed_orders": result.failed_order_audit.to_dict("records"),
        "config": result.config,
    }
    output_path.write_text(json_dump(payload), encoding="utf-8")
    typer.echo(f"status=ok output={output_path} failed_orders={len(result.failed_order_audit)}")


@app.command("paper-trade-v7")
def paper_trade_v7(
    target_weights_path: Path = typer.Option(..., "--target-weights"),
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    output_path: Path = typer.Option(Path("reports/v7/paper_trade_report.json"), "--output"),
) -> None:
    """Run the same dry-run VirtualBroker path used by backtests; no live submit is possible here."""
    walk_forward_backtest_v7(target_weights_path, market_panel_path, output_path)


@app.command("walk-forward-v7")
def walk_forward_v7(
    sleeve_returns_path: Path = typer.Option(..., "--sleeve-returns"),
    output_path: Path = Path("reports/v7/walk_forward_sleeve_allocation.json"),
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json_dump(result), encoding="utf-8")
    typer.echo(
        f"status=ok windows={result.diagnostics.get('walk_forward_windows', 0)} "
        f"cash_weight={result.cash_weight:.3f} output={output_path}"
    )
