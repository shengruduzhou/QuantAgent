"""V7 backtest CLI: walk-forward backtest, paper trade, sleeve walk-forward allocator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from quantagent.cli._utils import app, json_dump, read_frame


@app.command("walk-forward-backtest-v7")
def walk_forward_backtest_v7(
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    output_path: Path = typer.Option(Path("reports/v7/walk_forward_backtest.json"), "--output"),
    target_weights_path: Path | None = typer.Option(
        None, "--target-weights", help="Optional explicit target_weights frame; mutually exclusive with --predictions."
    ),
    predictions_path: Path | None = typer.Option(
        None, "--predictions", help="If provided, run the V7 target-weights optimiser before backtest."
    ),
    sector_map_path: Path | None = typer.Option(None, "--sector-map"),
    top_k: int = typer.Option(30, "--top-k"),
    max_weight_per_name: float = typer.Option(0.10, "--max-weight"),
    max_sector_weight: float = typer.Option(0.30, "--max-sector"),
    max_turnover: float = typer.Option(0.50, "--max-turnover"),
    long_short: bool = typer.Option(False, "--long-short/--long-only"),
    horizon_column: str | None = typer.Option(None, "--horizon-column"),
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
                max_weight_per_name=max_weight_per_name,
                max_sector_weight=max_sector_weight,
                max_turnover=max_turnover,
                horizon_column=horizon_column,
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "nav": result.nav.to_dict(),
        "orders": result.order_audit.to_dict("records"),
        "failed_orders": result.failed_order_audit.to_dict("records"),
        "config": result.config,
    }
    if optimizer_diagnostics is not None:
        payload["optimizer_diagnostics"] = optimizer_diagnostics
    output_path.write_text(json_dump(payload), encoding="utf-8")
    typer.echo(f"status=ok output={output_path} failed_orders={len(result.failed_order_audit)}")


@app.command("paper-trade-v7")
def paper_trade_v7(
    target_weights_path: Path = typer.Option(..., "--target-weights"),
    market_panel_path: Path = typer.Option(..., "--market-panel"),
    output_path: Path = typer.Option(Path("reports/v7/paper_trade_report.json"), "--output"),
) -> None:
    """Run the same dry-run VirtualBroker path used by backtests; no live submit is possible here."""
    walk_forward_backtest_v7(
        market_panel_path=market_panel_path,
        output_path=output_path,
        target_weights_path=target_weights_path,
    )


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
