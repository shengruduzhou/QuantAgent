"""Strict-backtest-driven regime policy search.

This module deliberately optimises the same surface the operator will later
judge: composite alpha -> decision chain -> strict A-share execution backtest.
It is slower than label/proxy objectives, but it avoids selecting horizon
weights that look good before gates and then fail after T+1, limit-state,
liquidity, turnover and cost constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import StrictBacktestArtifactSet, run_strict_backtest_v8
from quantagent.ensemble.blend_optimizer import HORIZONS, _apply_weights, _simplex_grid
from quantagent.risk.decision_chain import DecisionChainConfig, run_decision_chain
from quantagent.risk.decision_chain import _compute_avg_amount, _compute_trend_quality, _ensure_state_flags


REGIME_ORDER = ("bull", "neutral_up", "neutral_down", "bear", "crisis")


@dataclass(frozen=True)
class ExpertPolicy:
    """One regime expert: horizon blend plus gross exposure scale."""

    weights: tuple[float, float, float]
    gross_scale: float = 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "weights": {
                HORIZONS[0]: self.weights[0],
                HORIZONS[1]: self.weights[1],
                HORIZONS[2]: self.weights[2],
            },
            "gross_scale": self.gross_scale,
        }


@dataclass(frozen=True)
class RegimePolicy:
    """Full meta-policy with global fallback and optional regime experts."""

    global_policy: ExpertPolicy
    regime_policies: dict[str, ExpertPolicy] = field(default_factory=dict)

    def policy_for(self, regime: str | None) -> ExpertPolicy:
        if regime and regime in self.regime_policies:
            return self.regime_policies[regime]
        return self.global_policy

    def as_dict(self) -> dict[str, object]:
        return {
            "global": self.global_policy.as_dict(),
            "regimes": {
                regime: policy.as_dict()
                for regime, policy in sorted(self.regime_policies.items())
            },
        }


@dataclass(frozen=True)
class StrictPolicySearchConfig:
    """Configuration for strict policy search."""

    grid_step: float = 0.25
    coordinate_passes: int = 1
    min_regime_days: int = 20
    top_k: int = 10
    candidate_pool_size: int = 40
    max_name_weight: float = 0.05
    max_sector_weight: float = 0.30
    max_consecutive_limit_up: int = 2
    min_avg_amount_yuan: float = 5e7
    liquidity_window: int = 60
    sector_pool_top_n: int = 0
    limit_up_position_cap: float = 0.05
    block_one_word_limit_up: bool = True
    allow_limit_up_small_position: bool = True
    slippage_bps: float = 8.0
    initial_cash: float = 1_000_000.0
    return_weight: float = 1.0
    excess_weight: float = 1.0
    drawdown_penalty: float = 0.50
    turnover_penalty: float = 0.02
    cost_penalty: float = 0.50


@dataclass(frozen=True)
class StrictPolicyTrial:
    trial_id: int
    stage: str
    regime: str
    policy: RegimePolicy
    score: float
    metrics: dict[str, object]
    decision_summary: dict[str, object]

    def as_row(self) -> dict[str, object]:
        row = {
            "trial_id": self.trial_id,
            "stage": self.stage,
            "regime": self.regime,
            "score": self.score,
        }
        row.update({f"metric_{k}": v for k, v in self.metrics.items() if isinstance(v, (int, float, str))})
        gp = self.policy.global_policy
        row.update({
            "global_short": gp.weights[0],
            "global_mid": gp.weights[1],
            "global_long": gp.weights[2],
            "global_scale": gp.gross_scale,
        })
        for regime, policy in sorted(self.policy.regime_policies.items()):
            row[f"{regime}_short"] = policy.weights[0]
            row[f"{regime}_mid"] = policy.weights[1]
            row[f"{regime}_long"] = policy.weights[2]
            row[f"{regime}_scale"] = policy.gross_scale
        return row


@dataclass(frozen=True)
class StrictPolicySearchResult:
    best_policy: RegimePolicy
    best_score: float
    best_metrics: dict[str, object]
    trials: list[StrictPolicyTrial]
    regime_days: dict[str, int]
    config: StrictPolicySearchConfig

    def as_dict(self) -> dict[str, object]:
        return {
            "best_policy": self.best_policy.as_dict(),
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "regime_days": dict(sorted(self.regime_days.items())),
            "config": self.config.__dict__,
            "n_trials": len(self.trials),
            "top_trials": [t.as_row() for t in sorted(self.trials, key=lambda t: -t.score)[:10]],
        }


@dataclass(frozen=True)
class StrictPolicyEvaluation:
    policy: RegimePolicy
    score: float
    metrics: dict[str, object]
    decision_summary: dict[str, object]
    composite: pd.DataFrame
    target_weights: pd.DataFrame
    backtest: StrictBacktestArtifactSet | None


def normalize_regime_by_date(regime_by_date: pd.Series | pd.DataFrame | None) -> pd.Series:
    """Return a Timestamp-indexed regime label series."""
    if regime_by_date is None:
        return pd.Series(dtype="object")
    if isinstance(regime_by_date, pd.DataFrame):
        if "regime" not in regime_by_date.columns:
            raise ValueError("regime_by_date DataFrame must include a regime column")
        series = regime_by_date["regime"].copy()
        if "trade_date" in regime_by_date.columns:
            series.index = pd.to_datetime(regime_by_date["trade_date"], errors="coerce")
    else:
        series = regime_by_date.copy()
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[series.index.notna()].dropna()
    return series.astype(str).sort_index()


def build_regime_policy_composite(
    per_horizon: Mapping[str, pd.DataFrame],
    policy: RegimePolicy,
    regime_by_date: pd.Series | pd.DataFrame | None,
) -> pd.DataFrame:
    """Blend horizon predictions by date-specific regime policy."""
    regimes = normalize_regime_by_date(regime_by_date)
    all_frames: list[pd.DataFrame] = []
    weights_needed = {policy.global_policy.weights}
    weights_needed.update(p.weights for p in policy.regime_policies.values())
    blended = {
        weights: _apply_weights(dict(per_horizon), weights)
        for weights in sorted(weights_needed)
    }
    all_dates = sorted({
        pd.Timestamp(d)
        for frame in per_horizon.values()
        for d in pd.to_datetime(frame["trade_date"], errors="coerce").dropna().unique()
    })
    for date in all_dates:
        regime = str(regimes.get(date)) if date in regimes.index else None
        expert = policy.policy_for(regime)
        frame = blended[expert.weights]
        day = frame[frame["trade_date"] == date]
        if not day.empty:
            all_frames.append(day[["trade_date", "symbol", "composite_score"]])
    if not all_frames:
        return pd.DataFrame(columns=["trade_date", "symbol", "composite_score"])
    return pd.concat(all_frames, ignore_index=True)


def scale_target_weights_by_regime(
    target_weights: pd.DataFrame,
    policy: RegimePolicy,
    regime_by_date: pd.Series | pd.DataFrame | None,
) -> pd.DataFrame:
    """Apply policy gross scale to a decision-chain target weight frame."""
    if target_weights.empty:
        return target_weights.copy()
    regimes = normalize_regime_by_date(regime_by_date)
    out = target_weights.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    for date in out.index:
        regime = str(regimes.get(date)) if date in regimes.index else None
        scale = float(policy.policy_for(regime).gross_scale)
        out.loc[date] = out.loc[date].astype(float) * scale
    return out


def evaluate_strict_policy(
    *,
    per_horizon: Mapping[str, pd.DataFrame],
    policy: RegimePolicy,
    regime_by_date: pd.Series | pd.DataFrame | None,
    market_panel: pd.DataFrame,
    sector_map: pd.DataFrame | None,
    sector_pool: pd.DataFrame | None,
    config: StrictPolicySearchConfig,
    write_backtest: bool = False,
) -> StrictPolicyEvaluation:
    """Run composite -> decision chain -> strict backtest for one policy."""
    composite = build_regime_policy_composite(per_horizon, policy, regime_by_date)
    if composite.empty:
        return StrictPolicyEvaluation(policy, -np.inf, {}, {}, composite, pd.DataFrame(), None)
    dc_config = DecisionChainConfig(
        top_k=config.top_k,
        candidate_pool_size=config.candidate_pool_size,
        max_name_weight=config.max_name_weight,
        max_sector_weight=config.max_sector_weight,
        max_consecutive_limit_up=config.max_consecutive_limit_up,
        min_avg_amount_yuan=config.min_avg_amount_yuan,
        liquidity_window=config.liquidity_window,
        sector_pool_top_n=config.sector_pool_top_n,
        limit_up_position_cap=config.limit_up_position_cap,
        block_one_word_limit_up=config.block_one_word_limit_up,
        allow_limit_up_small_position=config.allow_limit_up_small_position,
        old_dealer_risk_max=0.70,
        regime_position_scaling=False,
    )
    dc = run_decision_chain(
        composite=composite,
        market_panel=market_panel,
        sector_map=sector_map,
        sector_pool=sector_pool,
        config=dc_config,
    )
    target = scale_target_weights_by_regime(dc.target_weights, policy, regime_by_date)
    if target.empty or float(target.abs().sum(axis=1).sum()) <= 0.0:
        metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "turnover": 0.0,
            "total_cost": 0.0,
            "benchmark_equal_weight_ann": 0.0,
            "excess_return_ann": 0.0,
        }
        return StrictPolicyEvaluation(policy, _score_metrics(metrics, config), metrics, dc.summary, composite, target, None)

    bt_start = pd.to_datetime(target.index.min())
    bt_end = pd.to_datetime(target.index.max())
    panel = market_panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    bt_panel = panel[
        (panel["trade_date"] >= bt_start)
        & (panel["trade_date"] <= bt_end)
        & (panel["symbol"].isin(target.columns))
    ].reset_index(drop=True)
    for col in ("is_suspended", "is_st", "is_limit_up", "is_limit_down"):
        if col not in bt_panel.columns:
            bt_panel[col] = False
    bt = run_strict_backtest_v8(
        target,
        bt_panel,
        sector_map=sector_map,
        factor_weights=_policy_factor_weights(policy),
        config=AShareExecutionSimulationConfig(
            initial_cash=config.initial_cash,
            slippage_bps=config.slippage_bps,
        ),
    )
    metrics = bt.metrics.to_dict()
    bench = equal_weight_benchmark(market_panel, bt_start, bt_end)
    metrics["benchmark_equal_weight_ann"] = bench.get("ann", float("nan"))
    metrics["benchmark_equal_weight_total"] = bench.get("total_return", float("nan"))
    metrics["excess_return_ann"] = float(metrics["annualized_return"] - metrics["benchmark_equal_weight_ann"])
    score = _score_metrics(metrics, config)
    return StrictPolicyEvaluation(
        policy,
        score,
        metrics,
        dc.summary,
        composite,
        target,
        bt if write_backtest else None,
    )


def search_strict_regime_policy(
    *,
    per_horizon: Mapping[str, pd.DataFrame],
    regime_by_date: pd.Series | pd.DataFrame | None,
    market_panel: pd.DataFrame,
    sector_map: pd.DataFrame | None = None,
    sector_pool: pd.DataFrame | None = None,
    config: StrictPolicySearchConfig | None = None,
    on_trial: Callable[[StrictPolicyTrial], None] | None = None,
) -> StrictPolicySearchResult:
    """Search global and per-regime policies with strict backtest scoring."""
    cfg = config or StrictPolicySearchConfig()
    market_panel = prepare_decision_chain_panel(market_panel, cfg, sector_map=sector_map)
    regimes = normalize_regime_by_date(regime_by_date)
    prediction_dates = _prediction_dates_from_horizons(per_horizon)
    if len(prediction_dates):
        regimes = regimes.reindex(prediction_dates).dropna()
    grid = _simplex_grid(step=cfg.grid_step)
    trials: list[StrictPolicyTrial] = []
    trial_id = 0

    def _eval(stage: str, regime: str, policy: RegimePolicy) -> StrictPolicyEvaluation:
        nonlocal trial_id
        trial_id += 1
        ev = evaluate_strict_policy(
            per_horizon=per_horizon,
            policy=policy,
            regime_by_date=regimes,
            market_panel=market_panel,
            sector_map=sector_map,
            sector_pool=sector_pool,
            config=cfg,
        )
        trials.append(StrictPolicyTrial(
            trial_id=trial_id,
            stage=stage,
            regime=regime,
            policy=policy,
            score=ev.score,
            metrics=ev.metrics,
            decision_summary=ev.decision_summary,
        ))
        if on_trial is not None:
            on_trial(trials[-1])
        return ev

    best_eval: StrictPolicyEvaluation | None = None
    for weights in grid:
        policy = RegimePolicy(global_policy=ExpertPolicy(weights=weights, gross_scale=1.0))
        ev = _eval("global", "global", policy)
        if best_eval is None or ev.score > best_eval.score:
            best_eval = ev
    if best_eval is None:
        raise ValueError("strict policy search produced no valid global policy")

    best_policy = best_eval.policy
    regime_days = _regime_day_counts(regimes)
    active_regimes = [
        r for r in REGIME_ORDER
        if regime_days.get(r, 0) >= int(cfg.min_regime_days)
    ]
    for _ in range(max(0, int(cfg.coordinate_passes))):
        improved = False
        for regime in active_regimes:
            regime_best = best_eval
            for weights in grid:
                for scale in _scale_grid_for_regime(regime):
                    candidate = RegimePolicy(
                        global_policy=best_policy.global_policy,
                        regime_policies={
                            **best_policy.regime_policies,
                            regime: ExpertPolicy(weights=weights, gross_scale=scale),
                        },
                    )
                    ev = _eval("regime_coordinate", regime, candidate)
                    if ev.score > regime_best.score:
                        regime_best = ev
            if regime_best is not best_eval and regime_best.score > best_eval.score:
                best_eval = regime_best
                best_policy = regime_best.policy
                improved = True
        if not improved:
            break

    return StrictPolicySearchResult(
        best_policy=best_policy,
        best_score=best_eval.score,
        best_metrics=best_eval.metrics,
        trials=trials,
        regime_days=regime_days,
        config=cfg,
    )


def prepare_decision_chain_panel(
    market_panel: pd.DataFrame,
    config: StrictPolicySearchConfig,
    sector_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Precompute rolling decision-chain columns once for many strict trials."""
    panel = _ensure_state_flags(market_panel.copy())
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    if "avg_amount_20d" not in panel.columns:
        panel = _compute_avg_amount(panel, window=config.liquidity_window)
    if not {"trend_quality", "dist_to_ma", "is_one_word_limit_up"}.issubset(panel.columns):
        panel = _compute_trend_quality(panel, ma_window=20, max_extension=1.20)
    if not {"old_dealer_risk_score", "old_dealer_block", "sector_resonance_score", "dip_buy_flow_score"}.issubset(panel.columns):
        panel = _compute_decision_core_scores(panel, sector_map=sector_map)
    return panel


def _compute_decision_core_scores(panel: pd.DataFrame, sector_map: pd.DataFrame | None) -> pd.DataFrame:
    """Attach lightweight core risk scores to the decision-chain panel."""
    from quantagent.factors.core_policy import build_core_factor_frame

    data = panel.sort_values(["symbol", "trade_date"]).copy()
    close = pd.to_numeric(data["close"], errors="coerce") if "close" in data.columns else pd.Series(np.nan, index=data.index)
    openp = pd.to_numeric(data["open"], errors="coerce") if "open" in data.columns else close
    data["return_1d"] = data.groupby("symbol", sort=False)["close"].pct_change(fill_method=None) if "close" in data.columns else 0.0
    data["momentum_5d"] = data.groupby("symbol", sort=False)["close"].pct_change(5, fill_method=None) if "close" in data.columns else 0.0
    data["momentum_20d"] = data.groupby("symbol", sort=False)["close"].pct_change(20, fill_method=None) if "close" in data.columns else 0.0
    data["intraday_return"] = close / openp.replace(0.0, np.nan) - 1.0
    amount = (
        pd.to_numeric(data["amount"], errors="coerce")
        if "amount" in data.columns else pd.Series(np.nan, index=data.index)
    )
    volume = (
        pd.to_numeric(data["volume"], errors="coerce")
        if "volume" in data.columns else pd.Series(np.nan, index=data.index)
    )
    if "amount_mean_20d" not in data.columns:
        data["amount_mean_20d"] = (
            amount.groupby(data["symbol"], sort=False)
            .rolling(20, min_periods=5)
            .mean()
            .reset_index(level=0, drop=True)
        )
    if "volume_mean_20d" not in data.columns:
        data["volume_mean_20d"] = (
            volume.groupby(data["symbol"], sort=False)
            .rolling(20, min_periods=5)
            .mean()
            .reset_index(level=0, drop=True)
        )
    core, _ = build_core_factor_frame(data, sector_map=sector_map)
    keep = [
        "symbol", "trade_date", "trend_strength_score", "sector_resonance_score",
        "dip_buy_flow_score", "old_dealer_risk_score", "old_dealer_block",
    ]
    merged = panel.merge(core[[c for c in keep if c in core.columns]], on=["symbol", "trade_date"], how="left")
    for col in ("trend_strength_score", "sector_resonance_score", "dip_buy_flow_score", "old_dealer_risk_score"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    if "old_dealer_block" in merged.columns:
        merged["old_dealer_block"] = merged["old_dealer_block"].fillna(0).astype(bool)
    return merged


def write_strict_policy_search_result(
    result: StrictPolicySearchResult,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["summary"] = output_dir / "strict_policy_search.json"
    paths["summary"].write_text(json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8")
    paths["trials"] = output_dir / "strict_policy_trials.csv"
    pd.DataFrame([trial.as_row() for trial in result.trials]).to_csv(paths["trials"], index=False)
    return paths


def equal_weight_benchmark(panel: pd.DataFrame, start, end) -> dict[str, float]:
    """Equal-weight all-A daily-rebalanced benchmark for [start, end]."""
    p = panel[["symbol", "trade_date", "close"]].copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"], errors="coerce")
    p = p[(p["trade_date"] >= pd.Timestamp(start)) & (p["trade_date"] <= pd.Timestamp(end))]
    piv = p.pivot_table(index="trade_date", columns="symbol", values="close")
    rets = piv.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    n = len(rets)
    if n < 2:
        return {"ann": float("nan"), "sharpe": float("nan"), "total_return": float("nan"), "days": float(n)}
    total = float((1.0 + rets).prod() - 1.0)
    ann = float((1.0 + total) ** (252.0 / n) - 1.0)
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(252.0))
    return {"ann": ann, "sharpe": sharpe, "total_return": total, "days": float(n)}


def _score_metrics(metrics: Mapping[str, object], config: StrictPolicySearchConfig) -> float:
    ann = _float_metric(metrics, "annualized_return")
    excess = _float_metric(metrics, "excess_return_ann")
    drawdown = _float_metric(metrics, "max_drawdown")
    turnover = _float_metric(metrics, "turnover")
    cost_ratio = _float_metric(metrics, "total_cost") / max(1.0, float(config.initial_cash))
    return float(
        config.return_weight * ann
        + config.excess_weight * excess
        - config.drawdown_penalty * drawdown
        - config.turnover_penalty * turnover
        - config.cost_penalty * cost_ratio
    )


def _float_metric(metrics: Mapping[str, object], key: str) -> float:
    try:
        value = float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        value = 0.0
    return value if np.isfinite(value) else 0.0


def _regime_day_counts(regimes: pd.Series) -> dict[str, int]:
    if regimes.empty:
        return {}
    return {str(k): int(v) for k, v in regimes.value_counts().to_dict().items()}


def _prediction_dates_from_horizons(per_horizon: Mapping[str, pd.DataFrame]) -> pd.DatetimeIndex:
    dates = pd.Index([], dtype="datetime64[ns]")
    for frame in per_horizon.values():
        if "trade_date" not in frame.columns:
            continue
        dates = dates.union(pd.Index(pd.to_datetime(frame["trade_date"], errors="coerce").dropna().unique()))
    return pd.DatetimeIndex(sorted(dates))


def _scale_grid_for_regime(regime: str) -> tuple[float, ...]:
    if regime in {"bull", "neutral_up"}:
        return (0.70, 0.85, 1.00)
    if regime == "neutral_down":
        return (0.30, 0.50, 0.70, 1.00)
    if regime == "bear":
        return (0.00, 0.15, 0.30, 0.50, 0.70)
    if regime == "crisis":
        return (0.00, 0.10, 0.20)
    return (0.50, 0.70, 1.00)


def _policy_factor_weights(policy: RegimePolicy) -> dict[str, float]:
    out = {
        "global_short_5d": policy.global_policy.weights[0],
        "global_mid_5d_30d": policy.global_policy.weights[1],
        "global_long_30d_120d": policy.global_policy.weights[2],
        "global_gross_scale": policy.global_policy.gross_scale,
    }
    for regime, expert in policy.regime_policies.items():
        out[f"{regime}_short_5d"] = expert.weights[0]
        out[f"{regime}_mid_5d_30d"] = expert.weights[1]
        out[f"{regime}_long_30d_120d"] = expert.weights[2]
        out[f"{regime}_gross_scale"] = expert.gross_scale
    return out


__all__ = [
    "ExpertPolicy",
    "RegimePolicy",
    "StrictPolicyEvaluation",
    "StrictPolicySearchConfig",
    "StrictPolicySearchResult",
    "StrictPolicyTrial",
    "build_regime_policy_composite",
    "evaluate_strict_policy",
    "normalize_regime_by_date",
    "prepare_decision_chain_panel",
    "scale_target_weights_by_regime",
    "search_strict_regime_policy",
    "write_strict_policy_search_result",
]
