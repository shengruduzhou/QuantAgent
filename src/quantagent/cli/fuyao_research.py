"""Executable Fuyao / Financial-API research recipes 13-15.

The strategy layer generates T-close target weights.  Execution is delegated to
QuantAgent's canonical A-share simulator so T+1 fills, limit/suspension rules,
lot size, slippage, costs and ledger semantics remain identical to every other
backtest path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from quantagent.cli._utils import app, default_reports_root, read_frame, write_frame


@app.command("run-fuyao-research-backtest")
def run_fuyao_research_backtest(
    strategy: str = typer.Option(..., help="price-volume-breakout | time-series-momentum | short-term-reversal"),
    market_panel_path: Path = typer.Option(..., "--market-panel", exists=True, dir_okay=False),
    output_dir: Path = typer.Option(default_reports_root() / "fuyao_research", "--output-dir"),
    benchmark_symbol: str = typer.Option(..., "--benchmark-symbol", help="Explicit benchmark, e.g. 000300.SH"),
    benchmark_panel_path: Path | None = typer.Option(None, "--benchmark-panel", exists=True, dir_okay=False),
    slippage_bps: float = typer.Option(8.0, "--slippage-bps", min=0.0, max=500.0),
    initial_cash: float = typer.Option(1_000_000.0, "--initial-cash", min=1_000.0),
    lot_size: int = typer.Option(100, "--lot-size", min=1),
    volume_participation_cap: float = typer.Option(0.10, "--volume-participation-cap", min=0.001, max=1.0),
    max_positions: int = typer.Option(20, "--max-positions", min=1, max=500),
    volume_ratio_min: float = typer.Option(1.5, "--volume-ratio-min", min=0.0, max=20.0),
    momentum_weighting: str = typer.Option("inverse_volatility", "--momentum-weighting", help="equal | inverse_volatility"),
    momentum_rebalance: str = typer.Option("week", "--momentum-rebalance", help="day | week | month"),
    min_amount: float = typer.Option(0.0, "--min-amount", min=0.0),
) -> None:
    """Run Fuyao example 13/14/15 through the canonical T+1 simulator."""
    from quantagent.backtest.ashare_execution_simulator import (
        AShareExecutionSimulationConfig,
        simulate_ashare_target_weights,
    )
    from quantagent.backtest.paper_report import PaperReportConfig, write_paper_report
    from quantagent.research.fuyao_strategy_recipes import (
        BreakoutConfig,
        MomentumConfig,
        ReversalConfig,
        price_volume_breakout_weights,
        short_term_reversal_weights,
        time_series_momentum_weights,
    )

    normalized = strategy.strip().lower().replace("_", "-")
    benchmark_symbol = benchmark_symbol.strip().upper()
    if not benchmark_symbol:
        raise typer.BadParameter("--benchmark-symbol is required")
    market = read_frame(market_panel_path)
    if market.empty:
        raise typer.BadParameter("market panel is empty")
    report_market = market.copy()

    if normalized == "price-volume-breakout":
        investable = market[market["symbol"].astype(str) != benchmark_symbol].copy()
        recipe = price_volume_breakout_weights(
            investable,
            BreakoutConfig(max_positions=max_positions, volume_ratio_min=volume_ratio_min),
        )
    elif normalized == "time-series-momentum":
        if momentum_weighting not in {"equal", "inverse_volatility"}:
            raise typer.BadParameter("--momentum-weighting must be equal or inverse_volatility")
        if momentum_rebalance not in {"day", "week", "month"}:
            raise typer.BadParameter("--momentum-rebalance must be day/week/month")
        recipe = time_series_momentum_weights(
            market,
            MomentumConfig(weighting=momentum_weighting, rebalance=momentum_rebalance),  # type: ignore[arg-type]
        )
        investable = market
    elif normalized == "short-term-reversal":
        benchmark_frame = read_frame(benchmark_panel_path) if benchmark_panel_path else market
        benchmark_rows = benchmark_frame[benchmark_frame["symbol"].astype(str) == benchmark_symbol].copy()
        if benchmark_rows.empty:
            raise typer.BadParameter(
                "short-term-reversal requires benchmark close history; include benchmark in --market-panel or provide --benchmark-panel"
            )
        benchmark_rows["trade_date"] = pd.to_datetime(benchmark_rows["trade_date"], errors="coerce")
        benchmark = benchmark_rows.sort_values("trade_date").set_index("trade_date")["close"].astype(float)
        investable = market[market["symbol"].astype(str) != benchmark_symbol].copy()
        recipe = short_term_reversal_weights(
            investable,
            benchmark,
            ReversalConfig(max_positions=max_positions, min_amount=min_amount),
        )
        if benchmark_panel_path is not None:
            report_market = pd.concat([market, benchmark_rows], ignore_index=True, sort=False)
    else:
        raise typer.BadParameter("strategy must be price-volume-breakout, time-series-momentum or short-term-reversal")

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = write_frame(recipe.target_weights.reset_index(), output_dir / "target_weights.parquet")
    signals_path = write_frame(recipe.signal_frame, output_dir / "signals.parquet")
    diagnostics_path = output_dir / "strategy_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "strategy": recipe.strategy,
                "config": recipe.config,
                "diagnostics": recipe.diagnostics,
                "source": "Fuyao/Financial-API best-practice recipe",
                "signalTiming": "T close",
                "executionTiming": "T+1 open",
                "benchmarkSymbol": benchmark_symbol,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    simulation = simulate_ashare_target_weights(
        recipe.target_weights,
        investable,
        AShareExecutionSimulationConfig(
            initial_cash=initial_cash,
            lot_size=lot_size,
            volume_participation_cap=volume_participation_cap,
            slippage_bps=slippage_bps,
            audit_log_dir=str(output_dir / "audit"),
        ),
    )
    report = write_paper_report(
        simulation,
        market_panel=report_market,
        config=PaperReportConfig(
            initial_cash=initial_cash,
            benchmark_symbol=benchmark_symbol,
            slippage_bps=slippage_bps,
            output_dir=output_dir,
            title=f"Fuyao {recipe.strategy} governed backtest",
            target_weights_path=str(weights_path),
        ),
    )
    summary = {
        "status": "passed",
        "strategy": recipe.strategy,
        "weights": str(weights_path),
        "signals": str(signals_path),
        "diagnostics": str(diagnostics_path),
        "report": report.files,
        "failedOrders": int(len(simulation.failed_order_audit)),
        "skippedOrders": int(len(simulation.skipped_order_audit)),
    }
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
