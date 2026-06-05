"""build-alpha-portfolio-v8 — turnover-controlled construction + strict backtest.

Takes an existing v8 ``predictions.parquet`` (a per-date alpha score frame
from ``train-v8-deep``) and constructs target weights with explicit
turnover control via :func:`build_alpha_portfolio`, then runs the strict v8
backtest and reports headline metrics *and* excess return vs the
equal-weight all-A benchmark — the number that says whether the model has
real harvestable alpha rather than market beta.

This deliberately does NOT retrain anything: it operates on saved
predictions so construction choices (book width, rebalance cadence,
long vs market-neutral) can be swept in seconds, not GPU-hours.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from quantagent.cli._utils import app


def _equal_weight_benchmark(panel: pd.DataFrame, start, end) -> dict:
    """Equal-weight, daily-rebalanced all-A benchmark over [start, end]."""
    p = panel[["symbol", "trade_date", "close"]].copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"], errors="coerce")
    p = p[(p["trade_date"] >= pd.Timestamp(start)) & (p["trade_date"] <= pd.Timestamp(end))]
    piv = p.pivot_table(index="trade_date", columns="symbol", values="close")
    rets = piv.pct_change(fill_method=None).mean(axis=1).dropna()
    n = len(rets)
    if n < 2:
        return {"ann": float("nan"), "sharpe": float("nan"), "max_dd": float("nan")}
    nav = (1 + rets).cumprod()
    ann = float(nav.iloc[-1] ** (252 / n) - 1)
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * (252 ** 0.5))
    max_dd = float((nav / nav.cummax() - 1.0).min())
    return {"ann": ann, "sharpe": sharpe, "max_dd": abs(max_dd), "days": int(n)}


def _resolve_predictions(deep_run_dir: Optional[Path], horizon: str,
                         predictions_path: Optional[Path]) -> tuple[Path, str]:
    """Return (predictions_path, score_column)."""
    if predictions_path is not None:
        return predictions_path, "alpha_score"
    if deep_run_dir is None:
        raise typer.BadParameter("provide --predictions-path or --deep-run-dir")
    # Prefer the tuned ensemble composite if present (richer signal), else
    # the single-horizon predictions.
    composite = deep_run_dir / "ensemble_composite.parquet"
    if horizon == "ensemble" and composite.exists():
        return composite, "composite_score"
    p = deep_run_dir / horizon / "predictions.parquet"
    if not p.exists():
        raise typer.BadParameter(f"no predictions at {p}")
    return p, "alpha_score"


@app.command("build-alpha-portfolio-v8")
def build_alpha_portfolio_v8(
    deep_run_dir: Optional[Path] = typer.Option(None, help="deep run dir (uses <dir>/<horizon>/predictions.parquet)"),
    horizon: str = typer.Option("mid_5d_30d", help="short_5d | mid_5d_30d | long_30d_120d | ensemble"),
    predictions_path: Optional[Path] = typer.Option(None, help="explicit predictions.parquet (overrides deep_run_dir)"),
    silver_panel_path: Path = typer.Option(
        Path("runtime/data/v7/silver/market_panel/market_panel.parquet"),
        exists=True, dir_okay=False,
    ),
    book_fraction: float = typer.Option(0.10, help="fraction of cross-section held long (0.10=decile)"),
    weighting: str = typer.Option("equal", help="equal | rank"),
    max_name_weight: float = typer.Option(0.05),
    rebalance_interval: int = typer.Option(20, help="emit a target every N trading days (sim holds between)"),
    long_short: bool = typer.Option(False, "--long-short/--long-only", help="market-neutral +top/-bottom book"),
    gross_scale: float = typer.Option(1.0, help="scalar exposure multiplier (regime hook)"),
    min_avg_amount: float = typer.Option(0.0, help="liquidity floor: drop names below this trailing avg amount (yuan); 0=down-cap"),
    liquidity_window: int = typer.Option(20, help="trailing window (days) for the liquidity floor"),
    slippage_bps: float = typer.Option(8.0),
    initial_cash: float = typer.Option(1_000_000.0),
    regime: bool = typer.Option(True, "--regime/--no-regime",
                                help="scale daily gross exposure by market regime; default on after bear-OOS validation"),
    hedge_ratio: float = typer.Option(0.0, help="short index-future overlay (1.0 ≈ market-neutral; 0=off)"),
    regime_hedge: bool = typer.Option(
        True, "--regime-hedge/--no-regime-hedge",
        help="dynamic index-hedge overlay from regime scale; can make net exposure negative in bear/crisis research NAV",
    ),
    hedge_cost_bps: float = typer.Option(50.0, help="annual hedge roll/basis/commission drag in bps"),
    output_dir: Path = typer.Option(...),
):
    """Construct turnover-controlled target weights → strict backtest → excess vs EW."""
    from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
    from quantagent.backtest.strict_v8 import run_strict_backtest_v8
    from quantagent.portfolio.alpha_portfolio import AlphaPortfolioConfig, build_alpha_portfolio

    pred_path, score_col = _resolve_predictions(deep_run_dir, horizon, predictions_path)
    typer.echo(f"[info] predictions={pred_path} score_col={score_col}")
    predictions = pd.read_parquet(pred_path)

    panel = pd.read_parquet(silver_panel_path)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")

    # Optional market-regime gross-exposure scaling (per rebalance date).
    gross_scale_by_date = None
    if regime:
        from quantagent.risk.decision_chain import DecisionChainConfig, _compute_market_regime
        reg = _compute_market_regime(panel, config=DecisionChainConfig())
        gross_scale_by_date = reg["position_scale"]
        typer.echo(f"[info] regime scaling on — scale range "
                   f"[{gross_scale_by_date.min():.2f}, {gross_scale_by_date.max():.2f}]")

    # Optional trailing-avg-amount liquidity table for the floor.
    liquidity = None
    if min_avg_amount > 0.0 and "amount" in panel.columns:
        amt = panel[["trade_date", "symbol", "amount"]].sort_values(["symbol", "trade_date"]).copy()
        amt["avg_amount"] = amt.groupby("symbol")["amount"].transform(
            lambda s: s.rolling(liquidity_window, min_periods=5).mean())
        liquidity = amt[["trade_date", "symbol", "avg_amount"]]
        typer.echo(f"[info] liquidity floor {min_avg_amount:,.0f} yuan ({liquidity_window}d avg amount)")

    cfg = AlphaPortfolioConfig(
        book_fraction=book_fraction, weighting=weighting,
        max_name_weight=max_name_weight, rebalance_interval=rebalance_interval,
        long_short=long_short, gross_scale=gross_scale,
        min_avg_amount_yuan=min_avg_amount,
    )
    sparse = build_alpha_portfolio(predictions, config=cfg, score_column=score_col,
                                   gross_scale_by_date=gross_scale_by_date, liquidity=liquidity)
    if sparse.empty:
        typer.echo("[fatal] constructor produced no target weights", err=True)
        raise typer.Exit(code=1)
    # Forward-fill rebalance-date weights to every trading day so the strict
    # simulator marks the (held) book to market daily — daily NAV → correct
    # annualised/Sharpe/max-DD. The signal set still only changes on the
    # rebalance cadence; daily rows just hold the last target between them.
    pred_dates = pd.to_datetime(predictions["trade_date"], errors="coerce").dropna().unique()
    daily_index = sorted(d for d in pred_dates
                         if sparse.index.min() <= d <= sparse.index.max())
    target_weights = sparse.reindex(daily_index).ffill().fillna(0.0)
    target_weights.index.name = "trade_date"

    output_dir.mkdir(parents=True, exist_ok=True)
    target_weights.to_parquet(output_dir / "target_weights.parquet")
    typer.echo(f"[info] {len(sparse)} rebalance dates → {len(target_weights)} daily rows × "
               f"{int((target_weights != 0).any().sum())} symbols ever held")

    bt_start, bt_end = target_weights.index.min(), target_weights.index.max()
    bt_panel = panel[(panel["trade_date"] >= bt_start) & (panel["trade_date"] <= bt_end)]
    bt_panel = bt_panel[bt_panel["symbol"].isin(target_weights.columns)].reset_index(drop=True)
    for col in ("is_suspended", "is_st", "is_limit_up", "is_limit_down"):
        if col not in bt_panel.columns:
            bt_panel[col] = False

    bt = run_strict_backtest_v8(
        target_weights, bt_panel,
        config=AShareExecutionSimulationConfig(slippage_bps=slippage_bps, initial_cash=initial_cash),
    )
    bt.write(output_dir / "backtest")
    bench = _equal_weight_benchmark(panel, bt_start, bt_end)
    excess = bt.metrics.annualized_return - bench.get("ann", float("nan"))
    headline = {
        "horizon": horizon, "score_column": score_col,
        "oos_start": str(bt_start), "oos_end": str(bt_end),
        "config": cfg.__dict__, "regime": regime,
        "strategy_ann": bt.metrics.annualized_return,
        "strategy_sharpe": bt.metrics.sharpe,
        "strategy_max_dd": bt.metrics.max_drawdown,
        "strategy_turnover": bt.metrics.turnover,
        "benchmark_equal_weight_ann": bench.get("ann"),
        "benchmark_equal_weight_sharpe": bench.get("sharpe"),
        "benchmark_equal_weight_max_dd": bench.get("max_dd"),
        "excess_return_ann": excess,
    }

    # Optional index-hedge overlay → market-neutral NAV (engine is long-only,
    # so the short leg is modelled here as a short index future).
    if regime_hedge and gross_scale_by_date is not None:
        from quantagent.portfolio.index_hedge import (
            apply_dynamic_index_hedge, equal_weight_market_return, nav_metrics,
        )
        idx_ret = equal_weight_market_return(panel)
        ratio_by_date = (1.0 - gross_scale_by_date).clip(lower=0.0, upper=1.2)
        hedged_nav = apply_dynamic_index_hedge(
            bt.nav, idx_ret, ratio_by_date, annual_cost_bps=hedge_cost_bps,
        )
        hm = nav_metrics(hedged_nav)
        hedged_nav.to_frame("nav").to_csv(output_dir / "regime_hedged_nav.csv", index_label="trade_date")
        ratio_by_date.rename("hedge_ratio").to_csv(output_dir / "regime_hedge_ratio_by_date.csv",
                                                   index_label="trade_date")
        headline["regime_hedge"] = True
        headline["regime_hedge_rule"] = "hedge_ratio = clip(1 - regime_position_scale, 0, 1.2)"
        headline["regime_hedged_ann"] = hm["ann"]
        headline["regime_hedged_sharpe"] = hm["sharpe"]
        headline["regime_hedged_max_dd"] = hm["max_dd"]
        headline["regime_hedge_min_net_exposure"] = float(
            (gross_scale_by_date.reindex(target_weights.index).ffill().fillna(1.0) -
             ratio_by_date.reindex(target_weights.index).ffill().fillna(0.0)).min()
        )
        typer.echo(
            f"[regime-hedge] dynamic → ann={hm['ann']:+.4f} "
            f"sharpe={hm['sharpe']:.2f} maxDD={hm['max_dd']:.4f} "
            f"min_net={headline['regime_hedge_min_net_exposure']:+.2f}")

    if hedge_ratio and hedge_ratio != 0.0:
        from quantagent.portfolio.index_hedge import (
            apply_index_hedge, equal_weight_market_return, nav_metrics,
        )
        idx_ret = equal_weight_market_return(panel)
        hedged_nav = apply_index_hedge(
            bt.nav, idx_ret, hedge_ratio=hedge_ratio, annual_cost_bps=hedge_cost_bps,
        )
        hm = nav_metrics(hedged_nav)
        hedged_nav.to_frame("nav").to_csv(output_dir / "hedged_nav.csv", index_label="trade_date")
        headline["hedge_ratio"] = hedge_ratio
        headline["hedged_ann"] = hm["ann"]
        headline["hedged_sharpe"] = hm["sharpe"]
        headline["hedged_max_dd"] = hm["max_dd"]
        typer.echo(
            f"[hedge] ratio={hedge_ratio:.2f} → hedged_ann={hm['ann']:+.4f} "
            f"sharpe={hm['sharpe']:.2f} maxDD={hm['max_dd']:.4f}")

    (output_dir / "headline_report.json").write_text(
        json.dumps(headline, indent=2, default=str), encoding="utf-8")
    typer.echo(
        f"[headline] strat_ann={bt.metrics.annualized_return:+.4f} "
        f"sharpe={bt.metrics.sharpe:.2f} maxDD={bt.metrics.max_drawdown:.4f} "
        f"turn={bt.metrics.turnover:.3f} | bench_ann={bench.get('ann', float('nan')):+.4f} "
        f"EXCESS={excess:+.4f}")
    return output_dir


__all__ = ["build_alpha_portfolio_v8"]
