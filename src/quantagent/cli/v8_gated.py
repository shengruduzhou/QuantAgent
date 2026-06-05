"""End-to-end gated backtest CLI for v8 deep runs.

Three commands:

* ``optimize-ensemble-weights-v8`` — given a deep run directory with
  ``{short_5d,mid_5d_30d,long_30d_120d}/predictions.parquet`` and the
  gold dataset, search the 2-simplex for the OOS-best blend and write
  ``ensemble_weights.json`` + ``ensemble_composite_tuned.parquet``.

* ``apply-decision-chain-v8`` — given a composite_score frame and a
  silver market panel (+ optional sector_map and sector_pool), run
  the 15-gate decision chain and write ``target_weights.parquet``
  ``decision_traces.parquet`` ``risk_events.json`` ``summary.json``.

* ``run-gated-backtest-v8`` — composes the two above + the strict v8
  backtest into one shot. Default-loads the most recent deep run.

The three are kept independent so an operator can re-tune weights
without re-running the backtest, etc.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from quantagent.cli._utils import app


def _build_fundamental_quality(panel: pd.DataFrame, metrics_path: Path | None) -> pd.DataFrame:
    """PIT-join a cross-sectional ``fundamental_quality`` factor onto the panel.

    Uses the latest fundamentals row available *at* each trade_date
    (``available_at <= trade_date``) via a per-symbol as-of merge, then turns
    ROE / net-margin / revenue-growth into a small positive quality score and
    penalises high leverage. The decision chain's pool re-ranker picks it up
    automatically when the ``fundamental_quality`` column is present.
    """
    if metrics_path is None or not Path(metrics_path).exists():
        return panel
    mp = pd.read_parquet(metrics_path)
    if mp.empty or "available_at" not in mp.columns:
        return panel
    keep = [c for c in ("symbol", "available_at", "roe", "net_margin",
                        "revenue_yoy", "net_income_yoy", "debt_to_asset_ratio")
            if c in mp.columns]
    mp = mp[keep].copy()
    mp["available_at"] = pd.to_datetime(mp["available_at"], errors="coerce")
    mp = mp.dropna(subset=["available_at"]).sort_values("available_at")
    out = panel.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.sort_values("trade_date")
    merged = pd.merge_asof(
        out, mp, left_on="trade_date", right_on="available_at",
        by="symbol", direction="backward",
    )
    # Cross-sectional quality: reward ROE / growth / margin, penalise leverage.
    def _z(col):
        if col not in merged.columns:
            return pd.Series(0.0, index=merged.index)
        s = pd.to_numeric(merged[col], errors="coerce")
        g = s.groupby(merged["trade_date"])
        return ((s - g.transform("mean")) / g.transform("std").replace(0.0, pd.NA)).fillna(0.0)
    quality = (
        _z("roe") + _z("net_margin") + _z("net_income_yoy") + _z("revenue_yoy")
        - 0.5 * _z("debt_to_asset_ratio")
    )
    # squash to a bounded positive 0-2 contribution comparable to trend_quality
    merged["fundamental_quality"] = (quality.clip(-3, 3) / 3.0 + 1.0)
    drop = [c for c in ("available_at", "roe", "net_margin", "revenue_yoy",
                        "net_income_yoy", "debt_to_asset_ratio") if c in merged.columns]
    return merged.drop(columns=drop)


def _equal_weight_benchmark(panel: pd.DataFrame, start, end) -> dict:
    """Equal-weight all-A daily-rebalanced benchmark for [start, end]."""
    import numpy as np
    p = panel[["symbol", "trade_date", "close"]].copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"], errors="coerce")
    p = p[(p["trade_date"] >= pd.Timestamp(start)) & (p["trade_date"] <= pd.Timestamp(end))]
    piv = p.pivot_table(index="trade_date", columns="symbol", values="close")
    rets = piv.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    n = len(rets)
    if n < 2:
        return {"ann": float("nan"), "sharpe": float("nan")}
    cum = float((1 + rets).prod() - 1)
    ann = float((1 + cum) ** (252 / n) - 1)
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * (252 ** 0.5))
    return {"ann": ann, "sharpe": sharpe, "cum": cum, "days": int(n)}


def _latest_deep_run(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith(("v8_deep_", "v8_deep_wide"))),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _regime_by_date_from_panel(panel: pd.DataFrame) -> pd.Series:
    """Compute PIT market regime labels from the silver market panel."""
    from quantagent.risk.decision_chain import DecisionChainConfig, _compute_market_regime

    panel = panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    regimes = _compute_market_regime(panel, config=DecisionChainConfig())
    return regimes["regime"]


@app.command("optimize-ensemble-weights-v8")
def optimize_ensemble_weights_v8(
    deep_run_dir: Optional[Path] = typer.Option(
        None,
        help="deep run directory (default: latest under runtime/reports/v8/deep)",
    ),
    gold_path: Path = typer.Option(
        Path("runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet"),
        exists=True, dir_okay=False,
        help="gold training dataset (used for realised forward_return labels)",
    ),
    metric: str = typer.Option(
        "topk_excess_return",
        help="objective: rank_ic | topk_return | topk_excess_return | topk_utility",
    ),
    target_label: str = typer.Option(
        "forward_return_20d",
        help="realised label to score against",
    ),
    objective_top_k: int = typer.Option(30, help="top-K sleeve size used by topk_* objectives"),
    drawdown_penalty: float = typer.Option(1.0, help="topk_utility drawdown penalty"),
    turnover_penalty: float = typer.Option(0.05, help="topk_utility selected-set turnover penalty"),
    grid_step: float = typer.Option(0.05),
    n_folds: int = typer.Option(3),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="output dir (default: <deep_run_dir>/ensemble_tuned)",
    ),
):
    """Search 3-horizon simplex for the OOS-best blend weights."""
    from quantagent.ensemble.blend_optimizer import (
        BlendObjective, optimize_blend_weights, save_blend_result,
        write_blended_composite,
    )

    if deep_run_dir is None:
        deep_run_dir = _latest_deep_run(Path("runtime/reports/v8/deep"))
        if deep_run_dir is None:
            typer.echo("[fatal] could not find any deep run under runtime/reports/v8/deep", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"[info] using latest deep run: {deep_run_dir}")
    if output_dir is None:
        output_dir = deep_run_dir / "ensemble_tuned"

    obj = BlendObjective(
        metric=metric,
        target_label=target_label,
        top_k=objective_top_k,
        drawdown_penalty=drawdown_penalty,
        turnover_penalty=turnover_penalty,
    )
    typer.echo(f"[info] searching simplex (step={grid_step}, n_folds={n_folds}, metric={metric})")
    result = optimize_blend_weights(
        deep_run_dir,
        gold_path=gold_path, objective=obj,
        step=grid_step, n_folds=n_folds,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_blend_result(result, output_path=output_dir / "ensemble_weights.json")
    write_blended_composite(
        deep_run_dir,
        weights=result.best_weights,
        output_path=output_dir / "ensemble_composite_tuned.parquet",
    )
    typer.echo(
        f"[ok] best weights = short={result.best_weights[0]:.2f} "
        f"mid={result.best_weights[1]:.2f} long={result.best_weights[2]:.2f}; "
        f"score = {result.best_score:.4f}; wrote {output_dir}"
    )
    return output_dir


@app.command("optimize-regime-aware-ensemble-v8")
def optimize_regime_aware_ensemble_v8(
    deep_run_dir: Optional[Path] = typer.Option(
        None,
        help="deep run directory (default: latest under runtime/reports/v8/deep)",
    ),
    gold_path: Path = typer.Option(
        Path("runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet"),
        exists=True, dir_okay=False,
        help="gold training dataset (used for realised forward_return labels)",
    ),
    silver_panel_path: Path = typer.Option(
        Path("runtime/data/v7/silver/market_panel/market_panel.parquet"),
        exists=True, dir_okay=False,
        help="silver market panel used to compute PIT market regime labels",
    ),
    metric: str = typer.Option(
        "topk_excess_return",
        help="objective: rank_ic | topk_return | topk_excess_return | topk_utility",
    ),
    target_label: str = typer.Option(
        "forward_return_20d",
        help="realised label to score against",
    ),
    objective_top_k: int = typer.Option(30, help="top-K sleeve size used by topk_* objectives"),
    drawdown_penalty: float = typer.Option(1.0, help="topk_utility drawdown penalty"),
    turnover_penalty: float = typer.Option(0.05, help="topk_utility selected-set turnover penalty"),
    grid_step: float = typer.Option(0.05),
    n_folds: int = typer.Option(3),
    min_regime_days: int = typer.Option(40),
    output_dir: Optional[Path] = typer.Option(
        None,
        help="output dir (default: <deep_run_dir>/ensemble_regime_aware)",
    ),
):
    """Tune horizon blend weights separately by market regime."""
    from quantagent.ensemble.blend_optimizer import (
        BlendObjective,
        optimize_regime_aware_blend_weights,
        save_regime_blend_result,
        write_regime_aware_composite,
    )

    if deep_run_dir is None:
        deep_run_dir = _latest_deep_run(Path("runtime/reports/v8/deep"))
        if deep_run_dir is None:
            typer.echo("[fatal] could not find any deep run under runtime/reports/v8/deep", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"[info] using latest deep run: {deep_run_dir}")
    if output_dir is None:
        output_dir = deep_run_dir / "ensemble_regime_aware"

    panel = pd.read_parquet(silver_panel_path)
    regime_by_date = _regime_by_date_from_panel(panel)
    obj = BlendObjective(
        metric=metric,
        target_label=target_label,
        top_k=objective_top_k,
        drawdown_penalty=drawdown_penalty,
        turnover_penalty=turnover_penalty,
    )
    typer.echo(
        f"[info] regime-aware simplex search "
        f"(step={grid_step}, n_folds={n_folds}, min_days={min_regime_days}, metric={metric})"
    )
    result = optimize_regime_aware_blend_weights(
        deep_run_dir,
        gold_path=gold_path,
        regime_by_date=regime_by_date,
        objective=obj,
        step=grid_step,
        n_folds=n_folds,
        min_regime_days=min_regime_days,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_regime_blend_result(result, output_path=output_dir / "regime_ensemble_weights.json")
    write_regime_aware_composite(
        deep_run_dir,
        regime_by_date=regime_by_date,
        result=result,
        output_path=output_dir / "ensemble_composite_regime_aware.parquet",
    )
    gw = result.global_result.best_weights
    typer.echo(
        f"[ok] global weights = short={gw[0]:.2f} mid={gw[1]:.2f} long={gw[2]:.2f}; "
        f"score={result.global_result.best_score:.4f}"
    )
    for regime, regime_result in sorted(result.regime_results.items()):
        w = regime_result.best_weights
        typer.echo(
            f"[ok] {regime}: short={w[0]:.2f} mid={w[1]:.2f} long={w[2]:.2f}; "
            f"score={regime_result.best_score:.4f}; days={result.regime_days.get(regime, 0)}"
        )
    for regime, reason in sorted(result.skipped_regimes.items()):
        typer.echo(f"[info] {regime}: fallback to global ({reason})")
    typer.echo(f"[DONE] wrote {output_dir}")
    return output_dir


@app.command("apply-decision-chain-v8")
def apply_decision_chain_v8(
    composite_path: Path = typer.Option(..., exists=True, dir_okay=False),
    market_panel_path: Path = typer.Option(
        Path("runtime/data/v7/silver/market_panel/market_panel.parquet"),
        exists=True, dir_okay=False,
    ),
    sector_map_path: Optional[Path] = typer.Option(
        Path("runtime/data/v7/silver/sector_map/sector_map.parquet"),
        help="silver sector_map (skip the sector gate if absent / all-missing)",
    ),
    sector_pool_path: Optional[Path] = typer.Option(None),
    top_k: int = typer.Option(30),
    max_name_weight: float = typer.Option(0.05),
    max_sector_weight: float = typer.Option(0.30),
    max_consecutive_limit_up: int = typer.Option(2),
    min_avg_amount_yuan: float = typer.Option(5e7),
    liquidity_window: int = typer.Option(20, help="trailing days for the PIT liquidity gate"),
    candidate_pool_size: int = typer.Option(0, help="top-N model pool → trend-rank to top_k (0=off)"),
    regime_position_scaling: bool = typer.Option(False, "--regime/--no-regime",
                                                 help="scale gross by market regime (牛市满仓/熊市空仓)"),
    sector_pool_top_n: int = typer.Option(0, help="0 disables the sector pool filter"),
    model_score_min: float = typer.Option(float("-inf")),
    require_known_sector: bool = typer.Option(False, "--require-known-sector/--allow-unknown-sector"),
    output_dir: Path = typer.Option(...),
):
    """Run the 15-gate decision chain and emit gated target_weights."""
    from quantagent.risk.decision_chain import (
        DecisionChainConfig, run_decision_chain,
    )

    composite = pd.read_parquet(composite_path) if composite_path.suffix != ".csv" else pd.read_csv(composite_path)
    panel = pd.read_parquet(market_panel_path)
    sector_map = None
    if sector_map_path is not None and sector_map_path.exists():
        try:
            sm = pd.read_parquet(sector_map_path)
            # If everything is "missing" we ignore it
            if "sector_level_1" in sm.columns and sm["sector_level_1"].notna().any():
                sector_map = sm
            else:
                typer.echo("[info] sector_map is all-missing; skipping sector gates")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"[warn] sector_map unreadable: {exc}", err=True)
    sector_pool = None
    if sector_pool_path is not None and sector_pool_path.exists():
        try:
            sector_pool = pd.read_parquet(sector_pool_path)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"[warn] sector_pool unreadable: {exc}", err=True)

    cfg = DecisionChainConfig(
        top_k=top_k, max_name_weight=max_name_weight, max_sector_weight=max_sector_weight,
        max_consecutive_limit_up=max_consecutive_limit_up,
        min_avg_amount_yuan=min_avg_amount_yuan,
        liquidity_window=liquidity_window,
        candidate_pool_size=candidate_pool_size,
        regime_position_scaling=regime_position_scaling,
        sector_pool_top_n=sector_pool_top_n,
        model_score_min=model_score_min,
        require_known_sector=require_known_sector,
    )
    typer.echo("[info] running decision chain ...")
    result = run_decision_chain(
        composite=composite, market_panel=panel,
        sector_map=sector_map, sector_pool=sector_pool, config=cfg,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.target_weights.to_parquet(output_dir / "target_weights.parquet")
    result.decision_traces.to_parquet(output_dir / "decision_traces.parquet")
    (output_dir / "risk_events.json").write_text(
        json.dumps(result.risk_events, indent=2, default=str), encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(result.summary, indent=2, default=str), encoding="utf-8",
    )
    typer.echo(
        f"[ok] {result.summary['n_accepted']} / {result.summary['n_candidates']} accepted "
        f"across {result.summary['n_dates']} dates; wrote {output_dir}"
    )
    return output_dir


@app.command("run-gated-backtest-v8")
def run_gated_backtest_v8(
    deep_run_dir: Optional[Path] = typer.Option(
        None,
        help="deep run directory (default: latest under runtime/reports/v8/deep)",
    ),
    gold_path: Path = typer.Option(
        Path("runtime/data/v7/gold/training_dataset/training_dataset_alpha181_full_nosynth.parquet"),
    ),
    silver_panel_path: Path = typer.Option(
        Path("runtime/data/v7/silver/market_panel/market_panel.parquet"),
    ),
    sector_map_path: Optional[Path] = typer.Option(
        Path("runtime/data/v7/silver/sector_map/sector_map.parquet"),
    ),
    sector_pool_path: Optional[Path] = typer.Option(None),
    fundamentals_path: Optional[Path] = typer.Option(
        Path("runtime/data/v7/silver/fundamentals/metrics_panel.parquet"),
        help="metrics_panel for the fundamental-quality factor in pool ranking",
    ),
    top_k: int = typer.Option(10, help="final holdings cap (≤ this many names)"),
    candidate_pool_size: int = typer.Option(40, help="model top-N pool before trend/fundamental filter; 0 disables two-stage"),
    limit_up_position_cap: float = typer.Option(0.05, help="regular 涨停 capped position weight"),
    block_one_word_limit_up: bool = typer.Option(True, "--block-one-word/--allow-one-word"),
    allow_limit_up_small_position: bool = typer.Option(True, "--limit-up-small/--limit-up-block"),
    max_consecutive_limit_up: int = typer.Option(2),
    min_avg_amount_yuan: float = typer.Option(5e7),
    liquidity_window: int = typer.Option(60, help="trailing days for the PIT liquidity gate"),
    regime_position_scaling: bool = typer.Option(False, "--regime/--no-regime",
                                                 help="scale gross by market regime (牛市满仓/熊市空仓)"),
    sector_pool_top_n: int = typer.Option(0),
    metric: str = typer.Option("topk_excess_return"),
    target_label: str = typer.Option("forward_return_20d"),
    drawdown_penalty: float = typer.Option(1.0, help="topk_utility drawdown penalty"),
    turnover_penalty: float = typer.Option(0.05, help="topk_utility selected-set turnover penalty"),
    n_folds: int = typer.Option(3),
    grid_step: float = typer.Option(0.05),
    slippage_bps: float = typer.Option(8.0),
    initial_cash: float = typer.Option(1_000_000.0),
    output_dir: Optional[Path] = typer.Option(None),
    skip_ensemble_tune: bool = typer.Option(False, "--skip-ensemble-tune"),
    regime_aware_ensemble: bool = typer.Option(
        True,
        "--regime-aware-ensemble/--global-ensemble",
        help="blend horizons by market regime instead of one global weight vector",
    ),
):
    """Tune ensemble → apply 15-gate chain → strict v8 backtest, one shot."""
    from quantagent.backtest.ashare_execution_simulator import (
        AShareExecutionSimulationConfig,
    )
    from quantagent.backtest.strict_v8 import run_strict_backtest_v8
    from quantagent.ensemble.blend_optimizer import (
        BlendObjective, optimize_blend_weights, save_blend_result,
        write_blended_composite,
        optimize_regime_aware_blend_weights, save_regime_blend_result,
        write_regime_aware_composite,
    )
    from quantagent.risk.decision_chain import (
        DecisionChainConfig, run_decision_chain,
    )

    if deep_run_dir is None:
        deep_run_dir = _latest_deep_run(Path("runtime/reports/v8/deep"))
        if deep_run_dir is None:
            typer.echo("[fatal] no deep run found", err=True)
            raise typer.Exit(code=1)
    output_dir = output_dir or (deep_run_dir / "gated")
    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"[info] gated backtest on {deep_run_dir} → {output_dir}")

    # 1. Ensemble tune (or use baseline 30/45/25)
    panel_for_regime = None
    if skip_ensemble_tune:
        weights = (0.30, 0.45, 0.25)
        composite_path = output_dir / "ensemble_composite_baseline.parquet"
        write_blended_composite(deep_run_dir, weights=weights, output_path=composite_path)
        typer.echo(f"[info] using baseline weights {weights}")
    elif regime_aware_ensemble:
        panel_for_regime = pd.read_parquet(silver_panel_path)
        regime_by_date = _regime_by_date_from_panel(panel_for_regime)
        result = optimize_regime_aware_blend_weights(
            deep_run_dir,
            gold_path=gold_path,
            regime_by_date=regime_by_date,
            objective=BlendObjective(
                metric=metric,
                target_label=target_label,
                top_k=top_k,
                drawdown_penalty=drawdown_penalty,
                turnover_penalty=turnover_penalty,
            ),
            step=grid_step,
            n_folds=n_folds,
        )
        save_regime_blend_result(result, output_path=output_dir / "regime_ensemble_weights.json")
        composite_path = output_dir / "ensemble_composite_regime_aware.parquet"
        write_regime_aware_composite(
            deep_run_dir,
            regime_by_date=regime_by_date,
            result=result,
            output_path=composite_path,
        )
        gw = result.global_result.best_weights
        typer.echo(
            f"[ok] global fallback weights = short={gw[0]:.2f} mid={gw[1]:.2f} "
            f"long={gw[2]:.2f}; score = {result.global_result.best_score:.4f}"
        )
        for regime, regime_result in sorted(result.regime_results.items()):
            w = regime_result.best_weights
            typer.echo(
                f"[ok] {regime} weights = short={w[0]:.2f} mid={w[1]:.2f} "
                f"long={w[2]:.2f}; score = {regime_result.best_score:.4f}"
            )
        for regime, reason in sorted(result.skipped_regimes.items()):
            typer.echo(f"[info] {regime}: fallback to global ({reason})")
    else:
        result = optimize_blend_weights(
            deep_run_dir, gold_path=gold_path,
            objective=BlendObjective(
                metric=metric,
                target_label=target_label,
                top_k=top_k,
                drawdown_penalty=drawdown_penalty,
                turnover_penalty=turnover_penalty,
            ),
            step=grid_step, n_folds=n_folds,
        )
        save_blend_result(result, output_path=output_dir / "ensemble_weights.json")
        weights = result.best_weights
        composite_path = output_dir / "ensemble_composite_tuned.parquet"
        write_blended_composite(deep_run_dir, weights=weights, output_path=composite_path)
        typer.echo(
            f"[ok] tuned weights = short={weights[0]:.2f} mid={weights[1]:.2f} "
            f"long={weights[2]:.2f}; score = {result.best_score:.4f}"
        )

    # 2. Decision chain
    composite = pd.read_parquet(composite_path)
    panel = panel_for_regime.copy() if panel_for_regime is not None else pd.read_parquet(silver_panel_path)
    sector_map = None
    if sector_map_path is not None and sector_map_path.exists():
        try:
            sm = pd.read_parquet(sector_map_path)
            if "sector_level_1" in sm.columns and sm["sector_level_1"].notna().any():
                sector_map = sm
        except Exception:  # noqa: BLE001
            pass
    sector_pool = None
    if sector_pool_path is not None and sector_pool_path.exists():
        try:
            sector_pool = pd.read_parquet(sector_pool_path)
        except Exception:  # noqa: BLE001
            pass
    # Fundamental-quality factor (PIT) for the two-stage pool re-ranker.
    panel = _build_fundamental_quality(panel, fundamentals_path)
    cfg = DecisionChainConfig(
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        limit_up_position_cap=limit_up_position_cap,
        block_one_word_limit_up=block_one_word_limit_up,
        allow_limit_up_small_position=allow_limit_up_small_position,
        max_consecutive_limit_up=max_consecutive_limit_up,
        min_avg_amount_yuan=min_avg_amount_yuan,
        liquidity_window=liquidity_window,
        regime_position_scaling=regime_position_scaling,
        sector_pool_top_n=sector_pool_top_n,
    )
    typer.echo(f"[info] running 15-gate chain (pool={candidate_pool_size}→top_k={top_k}, "
               f"regime={'on' if regime_position_scaling else 'off'}) ...")
    dc = run_decision_chain(
        composite=composite, market_panel=panel,
        sector_map=sector_map, sector_pool=sector_pool, config=cfg,
    )
    dc.target_weights.to_parquet(output_dir / "target_weights.parquet")
    dc.decision_traces.to_parquet(output_dir / "decision_traces.parquet")
    (output_dir / "risk_events.json").write_text(
        json.dumps(dc.risk_events, indent=2, default=str), encoding="utf-8",
    )
    (output_dir / "decision_summary.json").write_text(
        json.dumps(dc.summary, indent=2, default=str), encoding="utf-8",
    )
    typer.echo(
        f"[ok] {dc.summary['n_accepted']}/{dc.summary['n_candidates']} accepted "
        f"across {dc.summary['n_dates']} dates"
    )

    # 3. Strict v8 backtest
    if dc.target_weights.empty:
        typer.echo("[warn] decision chain produced 0 accepted weights; skipping backtest", err=True)
        return output_dir
    typer.echo("[info] strict backtest ...")
    bt_start = pd.to_datetime(dc.target_weights.index.min())
    bt_end = pd.to_datetime(dc.target_weights.index.max())
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    bt_panel = panel[(panel["trade_date"] >= bt_start) & (panel["trade_date"] <= bt_end)]
    bt_panel = bt_panel[bt_panel["symbol"].isin(dc.target_weights.columns)].reset_index(drop=True)
    for col in ("is_suspended", "is_st", "is_limit_up", "is_limit_down"):
        if col not in bt_panel.columns:
            bt_panel[col] = False
    cfg_bt = AShareExecutionSimulationConfig(
        initial_cash=initial_cash, slippage_bps=slippage_bps,
    )
    bt = run_strict_backtest_v8(dc.target_weights, bt_panel, config=cfg_bt)
    bt.write(output_dir / "backtest")
    typer.echo(f"[ok] backtest metrics: {bt.metrics.to_dict()}")

    # ── Headline report the operator asked for: excess return vs the
    #    equal-weight benchmark + per-date rank IC + top-10 tradeable return.
    bench = _equal_weight_benchmark(panel, bt_start, bt_end)
    rank_ic = float("nan")
    try:
        gold = pd.read_parquet(gold_path, columns=["symbol", "trade_date", target_label])
        gold["trade_date"] = pd.to_datetime(gold["trade_date"], errors="coerce")
        cm = composite[["symbol", "trade_date", "composite_score"]].copy()
        cm["trade_date"] = pd.to_datetime(cm["trade_date"], errors="coerce")
        mj = cm.merge(gold, on=["symbol", "trade_date"], how="inner").dropna()
        ics = []
        for _, g in mj.groupby("trade_date"):
            if len(g) >= 20:
                c = g["composite_score"].rank().corr(g[target_label].rank())
                if pd.notna(c):
                    ics.append(c)
        rank_ic = float(sum(ics) / len(ics)) if ics else float("nan")
    except Exception:  # noqa: BLE001
        pass
    excess = bt.metrics.annualized_return - bench.get("ann", float("nan"))
    headline = {
        "oos_start": str(bt_start), "oos_end": str(bt_end),
        "strategy_ann": bt.metrics.annualized_return,
        "strategy_sharpe": bt.metrics.sharpe,
        "strategy_max_dd": bt.metrics.max_drawdown,
        "benchmark_equal_weight_ann": bench.get("ann"),
        "benchmark_equal_weight_sharpe": bench.get("sharpe"),
        "excess_return_ann": excess,
        "rank_ic": rank_ic,
        "top_k": top_k, "candidate_pool_size": candidate_pool_size,
        "n_accepted": dc.summary.get("n_accepted"),
        "gate_counts": dc.summary.get("gate_counts"),
    }
    (output_dir / "headline_report.json").write_text(
        json.dumps(headline, indent=2, default=str), encoding="utf-8",
    )
    typer.echo(
        f"[headline] strat_ann={bt.metrics.annualized_return:.4f} "
        f"bench_ann={bench.get('ann', float('nan')):.4f} "
        f"EXCESS={excess:+.4f}  rank_IC={rank_ic:.4f}  "
        f"sharpe={bt.metrics.sharpe:.3f}"
    )

    # 4. Daily decision report on the final day
    from quantagent.diagnostics.daily_decision_report import (
        DailyDecisionInputs, build_daily_decision_report,
    )
    last_day_weights = dc.target_weights.iloc[-1] if not dc.target_weights.empty else None
    report_inputs = DailyDecisionInputs(
        as_of_date=bt_end,
        target_weights=last_day_weights,
        risk_events=bt.risk_events,
        global_conviction=float(min(1.0, max(0.0, bt.metrics.sharpe / 2.0))),
        gross_exposure=float(last_day_weights.sum()) if last_day_weights is not None else 0.0,
        market_regime="oos_backtest",
    )
    report = build_daily_decision_report(report_inputs)
    report.write(output_dir / "daily_decision_report.md")
    typer.echo(f"[ok] wrote {output_dir / 'daily_decision_report.md'}")
    typer.echo(f"\n[DONE] artifacts: {output_dir}")
    typer.echo(f"  total_return={bt.metrics.total_return:.4f}")
    typer.echo(f"  ann_return  ={bt.metrics.annualized_return:.4f}")
    typer.echo(f"  sharpe      ={bt.metrics.sharpe:.3f}")
    typer.echo(f"  max_dd      ={bt.metrics.max_drawdown:.4f}")
    typer.echo(f"  calmar      ={bt.metrics.calmar:.3f}")
    return output_dir


@app.command("search-strict-regime-policy-v8")
def search_strict_regime_policy_v8(
    deep_run_dir: Optional[Path] = typer.Option(
        None,
        help="deep run directory (default: latest under runtime/reports/v8/deep)",
    ),
    silver_panel_path: Path = typer.Option(
        Path("runtime/data/v7/silver/market_panel/market_panel.parquet"),
        exists=True,
        dir_okay=False,
    ),
    sector_map_path: Optional[Path] = typer.Option(
        Path("runtime/data/v7/silver/sector_map/sector_map.parquet"),
    ),
    sector_pool_path: Optional[Path] = typer.Option(None),
    fundamentals_path: Optional[Path] = typer.Option(
        Path("runtime/data/v7/silver/fundamentals/metrics_panel.parquet"),
        help="metrics_panel for the fundamental-quality factor in pool ranking",
    ),
    top_k: int = typer.Option(10, help="final holdings cap"),
    candidate_pool_size: int = typer.Option(40, help="model top-N pool before quality re-rank"),
    grid_step: float = typer.Option(0.25, help="simplex grid step; 0.25 is a fast strict-search default"),
    coordinate_passes: int = typer.Option(1, help="per-regime coordinate-search passes"),
    min_regime_days: int = typer.Option(20, help="skip sparse regimes below this many OOS dates"),
    slippage_bps: float = typer.Option(8.0),
    initial_cash: float = typer.Option(1_000_000.0),
    return_weight: float = typer.Option(1.0),
    excess_weight: float = typer.Option(1.0),
    drawdown_penalty: float = typer.Option(0.50),
    turnover_penalty: float = typer.Option(0.02),
    cost_penalty: float = typer.Option(0.50),
    panel_lookback_days: int = typer.Option(
        180,
        help="calendar-day lookback kept before prediction OOS start for rolling gates",
    ),
    start_date: Optional[str] = typer.Option(None, help="optional OOS prediction start date, YYYY-MM-DD"),
    end_date: Optional[str] = typer.Option(None, help="optional OOS prediction end date, YYYY-MM-DD"),
    output_dir: Optional[Path] = typer.Option(None),
):
    """Search regime-specific horizon/exposure policy using strict backtest metrics."""
    from quantagent.ensemble.blend_optimizer import load_predictions
    from quantagent.ensemble.strict_policy_search import (
        StrictPolicySearchConfig,
        evaluate_strict_policy,
        search_strict_regime_policy,
        write_strict_policy_search_result,
    )

    if deep_run_dir is None:
        deep_run_dir = _latest_deep_run(Path("runtime/reports/v8/deep"))
        if deep_run_dir is None:
            typer.echo("[fatal] no deep run found", err=True)
            raise typer.Exit(code=1)
    output_dir = output_dir or (deep_run_dir / "strict_regime_policy_search")
    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"[info] strict regime policy search on {deep_run_dir} → {output_dir}")

    per_horizon = load_predictions(deep_run_dir)
    if not per_horizon:
        typer.echo("[fatal] no horizon predictions found", err=True)
        raise typer.Exit(code=1)
    if start_date is not None or end_date is not None:
        lo = pd.Timestamp(start_date) if start_date is not None else None
        hi = pd.Timestamp(end_date) if end_date is not None else None
        filtered = {}
        for horizon, frame in per_horizon.items():
            f = frame.copy()
            f["trade_date"] = pd.to_datetime(f["trade_date"], errors="coerce")
            if lo is not None:
                f = f[f["trade_date"] >= lo]
            if hi is not None:
                f = f[f["trade_date"] <= hi]
            filtered[horizon] = f.reset_index(drop=True)
        per_horizon = filtered
        typer.echo(f"[info] filtered predictions to {start_date or '-inf'}→{end_date or '+inf'}")
    pred_dates = pd.DatetimeIndex(sorted({
        pd.Timestamp(d)
        for frame in per_horizon.values()
        for d in pd.to_datetime(frame["trade_date"], errors="coerce").dropna().unique()
    }))
    pred_symbols = {
        str(symbol)
        for frame in per_horizon.values()
        for symbol in frame["symbol"].dropna().astype(str).unique()
    }
    panel = pd.read_parquet(silver_panel_path)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    if len(pred_dates):
        lookback_start = pred_dates[0] - pd.Timedelta(days=int(panel_lookback_days))
        panel = panel[
            (panel["trade_date"] >= lookback_start)
            & (panel["trade_date"] <= pred_dates[-1])
            & (panel["symbol"].astype(str).isin(pred_symbols))
        ].reset_index(drop=True)
        typer.echo(
            f"[info] cropped panel to {len(panel):,} rows, "
            f"{panel['symbol'].nunique()} symbols, "
            f"{panel['trade_date'].min()}→{panel['trade_date'].max()}"
        )
    panel = _build_fundamental_quality(panel, fundamentals_path)
    regime_by_date = _regime_by_date_from_panel(panel)

    sector_map = None
    if sector_map_path is not None and sector_map_path.exists():
        sm = pd.read_parquet(sector_map_path)
        if "sector_level_1" in sm.columns and sm["sector_level_1"].notna().any():
            sector_map = sm
    sector_pool = None
    if sector_pool_path is not None and sector_pool_path.exists():
        sector_pool = pd.read_parquet(sector_pool_path)

    cfg = StrictPolicySearchConfig(
        grid_step=grid_step,
        coordinate_passes=coordinate_passes,
        min_regime_days=min_regime_days,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        slippage_bps=slippage_bps,
        initial_cash=initial_cash,
        return_weight=return_weight,
        excess_weight=excess_weight,
        drawdown_penalty=drawdown_penalty,
        turnover_penalty=turnover_penalty,
        cost_penalty=cost_penalty,
    )
    result = search_strict_regime_policy(
        per_horizon=per_horizon,
        regime_by_date=regime_by_date,
        market_panel=panel,
        sector_map=sector_map,
        sector_pool=sector_pool,
        config=cfg,
        on_trial=lambda trial: typer.echo(
            f"[trial {trial.trial_id:04d}] {trial.stage}/{trial.regime} "
            f"score={trial.score:.4f} "
            f"ann={trial.metrics.get('annualized_return', float('nan')):.4f} "
            f"excess={trial.metrics.get('excess_return_ann', float('nan')):+.4f} "
            f"dd={trial.metrics.get('max_drawdown', float('nan')):.4f}",
        ),
    )
    write_strict_policy_search_result(result, output_dir=output_dir)

    best = evaluate_strict_policy(
        per_horizon=per_horizon,
        policy=result.best_policy,
        regime_by_date=regime_by_date,
        market_panel=panel,
        sector_map=sector_map,
        sector_pool=sector_pool,
        config=cfg,
        write_backtest=True,
    )
    best.composite.to_parquet(output_dir / "best_composite.parquet")
    best.target_weights.to_parquet(output_dir / "best_target_weights.parquet")
    if best.backtest is not None:
        best.backtest.write(output_dir / "best_backtest")
    (output_dir / "best_decision_summary.json").write_text(
        json.dumps(best.decision_summary, indent=2, default=str), encoding="utf-8",
    )

    gp = result.best_policy.global_policy
    typer.echo(
        f"[ok] best score={result.best_score:.4f}; "
        f"global short={gp.weights[0]:.2f} mid={gp.weights[1]:.2f} "
        f"long={gp.weights[2]:.2f} scale={gp.gross_scale:.2f}"
    )
    for regime, policy in sorted(result.best_policy.regime_policies.items()):
        typer.echo(
            f"[ok] {regime}: short={policy.weights[0]:.2f} mid={policy.weights[1]:.2f} "
            f"long={policy.weights[2]:.2f} scale={policy.gross_scale:.2f}"
        )
    typer.echo(
        f"[DONE] ann={best.metrics.get('annualized_return', float('nan')):.4f} "
        f"excess={best.metrics.get('excess_return_ann', float('nan')):+.4f} "
        f"max_dd={best.metrics.get('max_drawdown', float('nan')):.4f} "
        f"turnover={best.metrics.get('turnover', float('nan')):.4f}"
    )
    return output_dir


@app.command("search-regime-factor-experts-v8")
def search_regime_factor_experts_v8(
    factor_frame_path: Path = typer.Option(..., exists=True, dir_okay=False),
    silver_panel_path: Path = typer.Option(
        Path("runtime/data/v7/silver/market_panel/market_panel.parquet"),
        exists=True,
        dir_okay=False,
    ),
    sector_map_path: Optional[Path] = typer.Option(
        Path("runtime/data/v7/silver/sector_map/sector_map.parquet"),
        help="optional sector map for sector concentration and resonance gates",
    ),
    sector_pool_path: Optional[Path] = typer.Option(None),
    regimes: str = typer.Option("bull,neutral,bear", help="comma-separated: bull,neutral,bear"),
    candidate_factors: Optional[str] = typer.Option(
        None,
        help="optional comma-separated factor columns. Default: infer numeric factor candidates.",
    ),
    top_k_values: str = typer.Option("10,15,20,30"),
    prefix_sizes: str = typer.Option("3,5,8,12,16,24,32"),
    max_candidate_factors: int = typer.Option(64),
    interaction_search: bool = typer.Option(True, "--interaction-search/--no-interaction-search"),
    beam_width: int = typer.Option(6),
    max_interaction_size: int = typer.Option(0, help="0 = use max(prefix_sizes)"),
    min_non_null_ratio: float = typer.Option(0.20),
    min_unique_values: int = typer.Option(10),
    candidate_pool_size: int = typer.Option(60),
    min_avg_amount_yuan: float = typer.Option(5e7),
    liquidity_window: int = typer.Option(60),
    slippage_bps: float = typer.Option(8.0),
    initial_cash: float = typer.Option(1_000_000.0),
    return_weight: float = typer.Option(0.50),
    excess_weight: float = typer.Option(2.00),
    drawdown_penalty: float = typer.Option(0.35),
    turnover_penalty: float = typer.Option(0.02),
    cost_penalty: float = typer.Option(0.50),
    start_date: Optional[str] = typer.Option(None),
    end_date: Optional[str] = typer.Option(None),
    output_dir: Path = typer.Option(
        Path("runtime/reports/v8/regime_factor_experts") / datetime.now().strftime("%Y%m%d_%H%M%S"),
    ),
):
    """Strict-search bull/neutral/bear factor subsets and stitch one OOS backtest.

    This is the factor-selection counterpart to ``search-strict-regime-policy-v8``:
    every regime gets its own factor subset and top-K, judged by decision-chain
    + strict execution metrics rather than a pre-gate label proxy.
    """
    from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
    from quantagent.backtest.strict_v8 import run_strict_backtest_v8
    from quantagent.ensemble.strict_factor_search import (
        StrictFactorSearchConfig,
        evaluate_strict_factor_subset,
        search_strict_factors,
        write_strict_factor_search_result,
    )
    from quantagent.ensemble.strict_policy_search import (
        StrictPolicySearchConfig,
        equal_weight_benchmark,
        prepare_decision_chain_panel,
    )
    from quantagent.risk.regime_family import compute_regime_family

    output_dir.mkdir(parents=True, exist_ok=True)
    explicit_factors = [c.strip() for c in candidate_factors.split(",") if c.strip()] if candidate_factors else None
    factor_frame = _read_factor_frame_for_search(factor_frame_path, explicit_factors)
    factor_frame["trade_date"] = pd.to_datetime(factor_frame["trade_date"], errors="coerce")
    factor_frame["symbol"] = factor_frame["symbol"].astype(str)
    panel = pd.read_parquet(silver_panel_path)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel["symbol"] = panel["symbol"].astype(str)
    if start_date is not None:
        lo = pd.Timestamp(start_date)
        factor_frame = factor_frame[factor_frame["trade_date"] >= lo]
        panel = panel[panel["trade_date"] >= lo - pd.Timedelta(days=260)]
    if end_date is not None:
        hi = pd.Timestamp(end_date)
        factor_frame = factor_frame[factor_frame["trade_date"] <= hi]
        panel = panel[panel["trade_date"] <= hi]
    symbols = set(factor_frame["symbol"].dropna().astype(str).unique())
    panel = panel[panel["symbol"].isin(symbols)].reset_index(drop=True)
    sector_map = (
        pd.read_parquet(sector_map_path)
        if sector_map_path is not None and sector_map_path.exists() and sector_map_path.suffix != ".csv"
        else (pd.read_csv(sector_map_path) if sector_map_path is not None and sector_map_path.exists() else None)
    )
    sector_pool = (
        pd.read_parquet(sector_pool_path)
        if sector_pool_path is not None and sector_pool_path.exists() and sector_pool_path.suffix != ".csv"
        else (pd.read_csv(sector_pool_path) if sector_pool_path is not None and sector_pool_path.exists() else None)
    )
    decision_cfg = StrictPolicySearchConfig(
        top_k=30,
        candidate_pool_size=candidate_pool_size,
        min_avg_amount_yuan=min_avg_amount_yuan,
        liquidity_window=liquidity_window,
        slippage_bps=slippage_bps,
        initial_cash=initial_cash,
        return_weight=return_weight,
        excess_weight=excess_weight,
        drawdown_penalty=drawdown_penalty,
        turnover_penalty=turnover_penalty,
        cost_penalty=cost_penalty,
    )
    prepared_panel = prepare_decision_chain_panel(panel, decision_cfg, sector_map=sector_map)
    regime_by_date = compute_regime_family(prepared_panel)
    requested_regimes = [r.strip() for r in regimes.split(",") if r.strip()]
    cfg_base = {
        "top_k_values": _parse_int_tuple(top_k_values),
        "prefix_sizes": _parse_int_tuple(prefix_sizes),
        "max_candidate_factors": int(max_candidate_factors),
        "interaction_search": bool(interaction_search),
        "beam_width": int(beam_width),
        "max_interaction_size": int(max_interaction_size),
        "min_non_null_ratio": float(min_non_null_ratio),
        "min_unique_values": int(min_unique_values),
        "return_weight": float(return_weight),
        "excess_weight": float(excess_weight),
        "drawdown_penalty": float(drawdown_penalty),
        "turnover_penalty": float(turnover_penalty),
        "cost_penalty": float(cost_penalty),
        "decision": decision_cfg,
    }
    target_parts: list[pd.DataFrame] = []
    composite_parts: list[pd.DataFrame] = []
    summaries: dict[str, object] = {
        "factor_frame_path": str(factor_frame_path),
        "silver_panel_path": str(silver_panel_path),
        "regime_days": {str(k): int(v) for k, v in regime_by_date.value_counts().to_dict().items()},
        "regimes": {},
    }
    for regime in requested_regimes:
        cfg = StrictFactorSearchConfig(regime_filter=regime, **cfg_base)
        regime_dir = output_dir / regime
        typer.echo(f"[info] searching {regime} factor expert → {regime_dir}")
        try:
            result = search_strict_factors(
                factor_frame=factor_frame,
                market_panel=prepared_panel,
                sector_map=sector_map,
                sector_pool=sector_pool,
                candidate_factors=explicit_factors,
                config=cfg,
            )
        except ValueError as exc:
            typer.echo(f"[warn] {regime} skipped: {exc}", err=True)
            summaries["regimes"][regime] = {"status": "skipped", "reason": str(exc)}
            continue
        write_strict_factor_search_result(
            result,
            factor_frame=factor_frame,
            market_panel=prepared_panel,
            sector_map=sector_map,
            sector_pool=sector_pool,
            output_dir=regime_dir,
        )
        regime_dates = set(regime_by_date[regime_by_date == regime].index)
        filtered = factor_frame[factor_frame["trade_date"].isin(regime_dates)].reset_index(drop=True)
        ev = evaluate_strict_factor_subset(
            factor_frame=filtered,
            factors=result.best_factors,
            top_k=result.best_top_k,
            market_panel=prepared_panel,
            sector_map=sector_map,
            sector_pool=sector_pool,
            config=cfg,
            factor_signs=result.factor_signs,
            write_backtest=False,
        )
        if not ev.target_weights.empty:
            tw = ev.target_weights.copy()
            tw["__regime"] = regime
            target_parts.append(tw)
        if not ev.composite.empty:
            comp = ev.composite.copy()
            comp["regime"] = regime
            comp["alpha_score"] = comp["composite_score"]
            composite_parts.append(comp)
        summaries["regimes"][regime] = {
            "status": "passed",
            "best_factors": list(result.best_factors),
            "best_top_k": int(result.best_top_k),
            "best_score": float(result.best_score),
            "best_metrics": result.best_metrics,
            "n_trials": len(result.trials),
            "candidate_factors": list(result.candidate_factors),
        }
        typer.echo(
            f"[ok] {regime}: score={result.best_score:.4f} "
            f"top_k={result.best_top_k} factors={','.join(result.best_factors)} "
            f"excess={result.best_metrics.get('excess_return_ann', float('nan')):+.4f}"
        )
    if composite_parts:
        composite_all = pd.concat(composite_parts, ignore_index=True).sort_values(["trade_date", "symbol"])
        composite_all.to_parquet(output_dir / "regime_factor_predictions.parquet", index=False)
    if target_parts:
        target_all = pd.concat(target_parts, axis=0).sort_index()
        if "__regime" in target_all.columns:
            regime_column = target_all.pop("__regime")
            regime_column.to_frame("regime").to_csv(output_dir / "target_weight_regimes.csv")
        target_all = target_all.fillna(0.0)
        target_all.to_parquet(output_dir / "regime_factor_target_weights.parquet")
        bt_start = pd.to_datetime(target_all.index.min())
        bt_end = pd.to_datetime(target_all.index.max())
        bt_panel = prepared_panel[
            (prepared_panel["trade_date"] >= bt_start)
            & (prepared_panel["trade_date"] <= bt_end)
            & (prepared_panel["symbol"].isin(target_all.columns.astype(str)))
        ].reset_index(drop=True)
        bt = run_strict_backtest_v8(
            target_all,
            bt_panel,
            sector_map=sector_map,
            config=AShareExecutionSimulationConfig(initial_cash=initial_cash, slippage_bps=slippage_bps),
        )
        bt.write(output_dir / "stitched_backtest")
        stitched_metrics = bt.metrics.to_dict()
        bench = equal_weight_benchmark(prepared_panel, bt_start, bt_end)
        stitched_metrics["benchmark_equal_weight_ann"] = bench.get("ann", float("nan"))
        stitched_metrics["benchmark_equal_weight_total"] = bench.get("total_return", float("nan"))
        stitched_metrics["excess_return_ann"] = float(
            stitched_metrics["annualized_return"] - stitched_metrics["benchmark_equal_weight_ann"]
        )
        summaries["stitched_backtest"] = stitched_metrics
        typer.echo(
            f"[DONE] stitched ann={stitched_metrics.get('annualized_return', float('nan')):.4f} "
            f"excess={stitched_metrics.get('excess_return_ann', float('nan')):+.4f} "
            f"max_dd={stitched_metrics.get('max_drawdown', float('nan')):.4f}"
        )
    else:
        summaries["stitched_backtest"] = {"status": "skipped", "reason": "no target weights"}
    (output_dir / "regime_factor_experts_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return output_dir


@app.command("assemble-regime-expert-predictions-v8")
def assemble_regime_expert_predictions_v8(
    bull_dir: Optional[Path] = typer.Option(None, help="root with short/mid/long subdirs for bull expert"),
    neutral_dir: Optional[Path] = typer.Option(None, help="root with short/mid/long subdirs for neutral expert"),
    bear_dir: Optional[Path] = typer.Option(None, help="root with short/mid/long subdirs for bear expert"),
    silver_panel_path: Path = typer.Option(
        Path("runtime/data/v7/silver/market_panel/market_panel.parquet"),
        exists=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(..., help="combined deep_run directory"),
):
    """Assemble separately trained regime experts into one deep_run structure."""
    from quantagent.ensemble.blend_optimizer import HORIZONS
    from quantagent.risk.regime_family import compute_regime_family

    regime_roots = {
        "bull": bull_dir,
        "neutral": neutral_dir,
        "bear": bear_dir,
    }
    panel = pd.read_parquet(silver_panel_path)
    regime_by_date = compute_regime_family(panel)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"horizons_used": [], "regime_sources": {}, "rows": {}}
    for horizon in HORIZONS:
        parts = []
        for regime, root in regime_roots.items():
            if root is None:
                continue
            pred_path = root / horizon / "predictions.parquet"
            if not pred_path.exists():
                alt = root / horizon / "predictions.parquet"
                if not alt.exists():
                    continue
                pred_path = alt
            pred = pd.read_parquet(pred_path)
            pred["trade_date"] = pd.to_datetime(pred["trade_date"], errors="coerce")
            dates = set(regime_by_date[regime_by_date.astype(str) == regime].index)
            pred = pred[pred["trade_date"].isin(dates)].copy()
            if pred.empty:
                continue
            pred["source_regime_expert"] = regime
            parts.append(pred)
            summary["regime_sources"][f"{horizon}:{regime}"] = str(pred_path)
        if not parts:
            typer.echo(f"[warn] no predictions assembled for {horizon}", err=True)
            continue
        combined = pd.concat(parts, ignore_index=True)
        combined = combined.sort_values(["trade_date", "symbol", "source_regime_expert"])
        combined = combined.drop_duplicates(["trade_date", "symbol"], keep="last")
        out_h = output_dir / horizon
        out_h.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_h / "predictions.parquet", index=False)
        summary["horizons_used"].append(horizon)
        summary["rows"][horizon] = int(len(combined))
        typer.echo(f"[ok] {horizon}: rows={len(combined):,} → {out_h / 'predictions.parquet'}")
    (output_dir / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    typer.echo(f"[DONE] assembled regime expert deep_run: {output_dir}")
    return output_dir


def _read_frame_any(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def _read_factor_frame_for_search(path: Path, candidate_factors: list[str] | None) -> pd.DataFrame:
    if path.suffix == ".csv" or not candidate_factors:
        return _read_frame_any(path)
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(path).schema.names)
    cols = ["symbol", "trade_date"] + [c for c in candidate_factors if c in names]
    missing = sorted(set(["symbol", "trade_date", *candidate_factors]).difference(names))
    if missing:
        raise typer.BadParameter(f"factor frame missing columns: {missing}")
    return pd.read_parquet(path, columns=cols)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    out = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if not out:
        raise typer.BadParameter("expected at least one integer")
    return out


__all__ = [
    "apply_decision_chain_v8",
    "assemble_regime_expert_predictions_v8",
    "optimize_ensemble_weights_v8",
    "optimize_regime_aware_ensemble_v8",
    "run_gated_backtest_v8",
    "search_regime_factor_experts_v8",
    "search_strict_regime_policy_v8",
]
