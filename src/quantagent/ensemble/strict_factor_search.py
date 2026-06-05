"""Strict-backtest-driven factor ranking, ablation, and top-k search."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import StrictBacktestArtifactSet, run_strict_backtest_v8
from quantagent.ensemble.strict_policy_search import (
    StrictPolicySearchConfig,
    _score_metrics,
    equal_weight_benchmark,
    prepare_decision_chain_panel,
)
from quantagent.risk.decision_chain import DecisionChainConfig, run_decision_chain
from quantagent.risk.regime_family import compute_regime_family


IDENTIFIER_COLUMNS = {
    "trade_date",
    "symbol",
    "datetime",
    "instrument",
    "code",
    "name",
    "sector_level_1",
    "sector_level_2",
}
LABEL_PREFIXES = ("label_", "target_", "forward_", "future_")
DEFAULT_NEGATIVE_FACTOR_HINTS = (
    "old_dealer",
    "risk",
    "volatility",
    "drawdown",
    "concentration",
    "spike",
    "turnover",
    "illiquidity",
    "leverage",
    "fraud",
    "downside",
)


@dataclass(frozen=True)
class StrictFactorSearchConfig:
    """Search settings for factor subsets judged by strict backtest objective."""

    top_k_values: tuple[int, ...] = (10, 15, 20, 30)
    prefix_sizes: tuple[int, ...] = (3, 5, 8, 12, 16, 24, 32)
    max_candidate_factors: int = 64
    interaction_search: bool = True
    beam_width: int = 6
    max_interaction_size: int = 0
    min_non_null_ratio: float = 0.20
    min_unique_values: int = 10
    regime_filter: str = "all"
    return_weight: float = 1.0
    excess_weight: float = 1.0
    drawdown_penalty: float = 0.50
    turnover_penalty: float = 0.02
    cost_penalty: float = 0.50
    decision: StrictPolicySearchConfig = field(default_factory=StrictPolicySearchConfig)


@dataclass(frozen=True)
class StrictFactorTrial:
    """One strict factor-search trial."""

    trial_id: int
    stage: str
    top_k: int
    factors: tuple[str, ...]
    score: float
    metrics: dict[str, object]
    decision_summary: dict[str, object]

    def as_row(self) -> dict[str, object]:
        row = {
            "trial_id": self.trial_id,
            "stage": self.stage,
            "top_k": self.top_k,
            "n_factors": len(self.factors),
            "factors": ",".join(self.factors),
            "score": self.score,
        }
        row.update({f"metric_{k}": v for k, v in self.metrics.items() if isinstance(v, (int, float, str))})
        return row


@dataclass(frozen=True)
class StrictFactorSearchResult:
    """Search output including the strict-selected factor subset."""

    best_factors: tuple[str, ...]
    best_top_k: int
    best_score: float
    best_metrics: dict[str, object]
    trials: list[StrictFactorTrial]
    candidate_factors: tuple[str, ...]
    factor_signs: dict[str, float]
    regime_filter: str
    config: StrictFactorSearchConfig

    def as_dict(self) -> dict[str, object]:
        return {
            "best_factors": list(self.best_factors),
            "best_top_k": int(self.best_top_k),
            "best_score": float(self.best_score),
            "best_metrics": self.best_metrics,
            "candidate_factors": list(self.candidate_factors),
            "factor_signs": self.factor_signs,
            "regime_filter": self.regime_filter,
            "config": {
                "top_k_values": list(self.config.top_k_values),
                "prefix_sizes": list(self.config.prefix_sizes),
                "max_candidate_factors": self.config.max_candidate_factors,
                "interaction_search": self.config.interaction_search,
                "beam_width": self.config.beam_width,
                "max_interaction_size": self.config.max_interaction_size,
                "min_non_null_ratio": self.config.min_non_null_ratio,
                "min_unique_values": self.config.min_unique_values,
                "regime_filter": self.config.regime_filter,
                "return_weight": self.config.return_weight,
                "excess_weight": self.config.excess_weight,
                "drawdown_penalty": self.config.drawdown_penalty,
                "turnover_penalty": self.config.turnover_penalty,
                "cost_penalty": self.config.cost_penalty,
                "decision": self.config.decision.__dict__,
            },
            "n_trials": len(self.trials),
            "top_trials": [t.as_row() for t in sorted(self.trials, key=lambda item: -item.score)[:20]],
        }


@dataclass(frozen=True)
class StrictFactorEvaluation:
    """Materialised evaluation for one factor subset."""

    factors: tuple[str, ...]
    top_k: int
    score: float
    metrics: dict[str, object]
    decision_summary: dict[str, object]
    composite: pd.DataFrame
    target_weights: pd.DataFrame
    backtest: StrictBacktestArtifactSet | None


def infer_candidate_factors(
    frame: pd.DataFrame,
    *,
    explicit: Sequence[str] | None = None,
    max_factors: int = 64,
    min_non_null_ratio: float = 0.20,
    min_unique_values: int = 10,
) -> tuple[str, ...]:
    """Return numeric factor columns, excluding identifiers and labels."""
    if explicit:
        missing = [col for col in explicit if col not in frame.columns]
        if missing:
            raise ValueError(f"candidate factors missing from dataset: {missing}")
        return tuple(str(col) for col in explicit)
    candidates: list[tuple[str, float]] = []
    for col in frame.columns:
        if col in IDENTIFIER_COLUMNS or col.startswith(LABEL_PREFIXES):
            continue
        series = frame[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        non_null = float(series.notna().mean())
        if non_null < float(min_non_null_ratio):
            continue
        unique = int(series.nunique(dropna=True))
        if unique < int(min_unique_values):
            continue
        score = non_null * np.log1p(unique)
        candidates.append((str(col), float(score)))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    if max_factors > 0:
        candidates = candidates[: int(max_factors)]
    return tuple(col for col, _ in candidates)


def infer_factor_signs(factors: Sequence[str], overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    """Infer positive/negative rank direction for each factor."""
    signs = {}
    for factor in factors:
        lower = factor.lower()
        sign = -1.0 if any(hint in lower for hint in DEFAULT_NEGATIVE_FACTOR_HINTS) else 1.0
        signs[str(factor)] = sign
    if overrides:
        for factor, sign in overrides.items():
            signs[str(factor)] = 1.0 if float(sign) >= 0.0 else -1.0
    return signs


def build_factor_composite(
    factor_frame: pd.DataFrame,
    factors: Sequence[str],
    *,
    factor_signs: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Convert selected raw factor columns into a per-date rank composite."""
    if not factors:
        raise ValueError("at least one factor is required")
    required = {"trade_date", "symbol", *factors}
    missing = sorted(required.difference(factor_frame.columns))
    if missing:
        raise ValueError(f"factor frame is missing required columns: {missing}")
    signs = factor_signs or infer_factor_signs(factors)
    frame = factor_frame[["trade_date", "symbol", *factors]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["symbol"] = frame["symbol"].astype(str)
    ranked_parts: list[pd.Series] = []
    for factor in factors:
        values = pd.to_numeric(frame[factor], errors="coerce")
        ranks = values.groupby(frame["trade_date"], sort=False).rank(pct=True, method="average")
        centered = ranks.fillna(0.5) - 0.5
        ranked_parts.append(centered * float(signs.get(str(factor), 1.0)))
    score = pd.concat(ranked_parts, axis=1).mean(axis=1)
    out = frame[["trade_date", "symbol"]].copy()
    out["composite_score"] = score.astype(float)
    return out.dropna(subset=["trade_date", "symbol", "composite_score"])


def filter_factor_frame_by_regime(
    factor_frame: pd.DataFrame,
    market_panel: pd.DataFrame,
    regime_filter: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter factor rows to bull/neutral/bear dates using PIT market regime."""
    regime = compute_regime_family(market_panel)
    normalized = str(regime_filter or "all").lower()
    if normalized in {"all", "*"}:
        return factor_frame.copy(), regime
    if normalized not in {"bull", "neutral", "bear"}:
        raise ValueError("regime_filter must be one of all, bull, neutral, bear")
    dates = set(regime[regime == normalized].index)
    out = factor_frame.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    return out[out["trade_date"].isin(dates)].reset_index(drop=True), regime


def evaluate_strict_factor_subset(
    *,
    factor_frame: pd.DataFrame,
    factors: Sequence[str],
    top_k: int,
    market_panel: pd.DataFrame,
    sector_map: pd.DataFrame | None = None,
    sector_pool: pd.DataFrame | None = None,
    config: StrictFactorSearchConfig | None = None,
    factor_signs: Mapping[str, float] | None = None,
    write_backtest: bool = False,
) -> StrictFactorEvaluation:
    """Run factor composite -> decision chain -> strict backtest."""
    cfg = config or StrictFactorSearchConfig()
    composite = build_factor_composite(factor_frame, tuple(factors), factor_signs=factor_signs)
    if composite.empty:
        return StrictFactorEvaluation(tuple(factors), int(top_k), -np.inf, {}, {}, composite, pd.DataFrame(), None)
    decision_cfg = cfg.decision
    dc_cfg = DecisionChainConfig(
        top_k=int(top_k),
        candidate_pool_size=decision_cfg.candidate_pool_size,
        max_name_weight=decision_cfg.max_name_weight,
        max_sector_weight=decision_cfg.max_sector_weight,
        max_consecutive_limit_up=decision_cfg.max_consecutive_limit_up,
        min_avg_amount_yuan=decision_cfg.min_avg_amount_yuan,
        liquidity_window=decision_cfg.liquidity_window,
        sector_pool_top_n=decision_cfg.sector_pool_top_n,
        limit_up_position_cap=decision_cfg.limit_up_position_cap,
        block_one_word_limit_up=decision_cfg.block_one_word_limit_up,
        allow_limit_up_small_position=decision_cfg.allow_limit_up_small_position,
        old_dealer_risk_max=0.70,
        regime_position_scaling=False,
    )
    dc = run_decision_chain(
        composite=composite,
        market_panel=market_panel,
        sector_map=sector_map,
        sector_pool=sector_pool,
        config=dc_cfg,
    )
    target = dc.target_weights
    if target.empty or float(target.abs().sum(axis=1).sum()) <= 0.0:
        metrics = _empty_metrics()
        return StrictFactorEvaluation(
            tuple(factors),
            int(top_k),
            _score_factor_metrics(metrics, cfg),
            metrics,
            dc.summary,
            composite,
            target,
            None,
        )
    bt_start = pd.to_datetime(target.index.min())
    bt_end = pd.to_datetime(target.index.max())
    panel = market_panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    bt_panel = panel[
        (panel["trade_date"] >= bt_start)
        & (panel["trade_date"] <= bt_end)
        & (panel["symbol"].astype(str).isin(target.columns.astype(str)))
    ].reset_index(drop=True)
    for col in ("is_suspended", "is_st", "is_limit_up", "is_limit_down"):
        if col not in bt_panel.columns:
            bt_panel[col] = False
    bt = run_strict_backtest_v8(
        target,
        bt_panel,
        sector_map=sector_map,
        factor_weights={name: float(factor_signs.get(name, 1.0) if factor_signs else 1.0) for name in factors},
        config=AShareExecutionSimulationConfig(
            initial_cash=decision_cfg.initial_cash,
            slippage_bps=decision_cfg.slippage_bps,
        ),
    )
    metrics = bt.metrics.to_dict()
    bench = equal_weight_benchmark(market_panel, bt_start, bt_end)
    metrics["benchmark_equal_weight_ann"] = bench.get("ann", float("nan"))
    metrics["benchmark_equal_weight_total"] = bench.get("total_return", float("nan"))
    metrics["excess_return_ann"] = float(metrics["annualized_return"] - metrics["benchmark_equal_weight_ann"])
    score = _score_factor_metrics(metrics, cfg)
    return StrictFactorEvaluation(
        tuple(factors),
        int(top_k),
        score,
        metrics,
        dc.summary,
        composite,
        target,
        bt if write_backtest else None,
    )


def search_strict_factors(
    *,
    factor_frame: pd.DataFrame,
    market_panel: pd.DataFrame,
    sector_map: pd.DataFrame | None = None,
    sector_pool: pd.DataFrame | None = None,
    candidate_factors: Sequence[str] | None = None,
    factor_signs: Mapping[str, float] | None = None,
    config: StrictFactorSearchConfig | None = None,
) -> StrictFactorSearchResult:
    """Rank factors and search top-prefix subsets with strict backtest scoring."""
    cfg = config or StrictFactorSearchConfig()
    filtered, _ = filter_factor_frame_by_regime(factor_frame, market_panel, cfg.regime_filter)
    if filtered.empty:
        raise ValueError(f"no factor rows after regime_filter={cfg.regime_filter}")
    market_panel = prepare_decision_chain_panel(market_panel, cfg.decision, sector_map=sector_map)
    candidates = infer_candidate_factors(
        filtered,
        explicit=candidate_factors,
        max_factors=cfg.max_candidate_factors,
        min_non_null_ratio=cfg.min_non_null_ratio,
        min_unique_values=cfg.min_unique_values,
    )
    if not candidates:
        raise ValueError("no candidate factors available for strict search")
    signs = infer_factor_signs(candidates, factor_signs)
    trials: list[StrictFactorTrial] = []
    trial_id = 0

    def _eval(stage: str, factors: Sequence[str], top_k: int) -> StrictFactorEvaluation:
        nonlocal trial_id
        trial_id += 1
        ev = evaluate_strict_factor_subset(
            factor_frame=filtered,
            factors=tuple(factors),
            top_k=int(top_k),
            market_panel=market_panel,
            sector_map=sector_map,
            sector_pool=sector_pool,
            config=cfg,
            factor_signs=signs,
        )
        trials.append(StrictFactorTrial(
            trial_id=trial_id,
            stage=stage,
            top_k=int(top_k),
            factors=tuple(factors),
            score=ev.score,
            metrics=ev.metrics,
            decision_summary=ev.decision_summary,
        ))
        return ev

    best_eval: StrictFactorEvaluation | None = None
    single_best: dict[str, float] = {}
    for factor in candidates:
        factor_best = -np.inf
        for top_k in cfg.top_k_values:
            ev = _eval("single_factor", (factor,), int(top_k))
            factor_best = max(factor_best, ev.score)
            if best_eval is None or ev.score > best_eval.score:
                best_eval = ev
        single_best[factor] = float(factor_best)
    ranked = tuple(sorted(candidates, key=lambda name: (-single_best.get(name, -np.inf), name)))
    for size in cfg.prefix_sizes:
        subset = ranked[: min(int(size), len(ranked))]
        if not subset:
            continue
        for top_k in cfg.top_k_values:
            ev = _eval("ranked_prefix", subset, int(top_k))
            if best_eval is None or ev.score > best_eval.score:
                best_eval = ev
    if cfg.interaction_search and ranked:
        max_size = int(cfg.max_interaction_size) if int(cfg.max_interaction_size) > 0 else max(cfg.prefix_sizes)
        max_size = max(1, min(max_size, len(ranked)))
        beam: list[tuple[tuple[str, ...], float]] = [
            ((factor,), float(single_best.get(factor, -np.inf)))
            for factor in ranked[: max(1, int(cfg.beam_width))]
        ]
        seen: set[tuple[str, ...]] = {item[0] for item in beam}
        for size in range(2, max_size + 1):
            candidates_for_size: list[tuple[tuple[str, ...], float]] = []
            for subset, _ in beam:
                for factor in ranked:
                    if factor in subset:
                        continue
                    new_subset = tuple(sorted((*subset, factor), key=ranked.index))
                    if new_subset in seen:
                        continue
                    seen.add(new_subset)
                    subset_best = -np.inf
                    for top_k in cfg.top_k_values:
                        ev = _eval("interaction_beam", new_subset, int(top_k))
                        subset_best = max(subset_best, ev.score)
                        if best_eval is None or ev.score > best_eval.score:
                            best_eval = ev
                    candidates_for_size.append((new_subset, float(subset_best)))
            if not candidates_for_size:
                break
            candidates_for_size.sort(key=lambda item: (-item[1], item[0]))
            beam = candidates_for_size[: max(1, int(cfg.beam_width))]
    if best_eval is None:
        raise ValueError("strict factor search produced no valid trial")
    return StrictFactorSearchResult(
        best_factors=best_eval.factors,
        best_top_k=best_eval.top_k,
        best_score=best_eval.score,
        best_metrics=best_eval.metrics,
        trials=trials,
        candidate_factors=ranked,
        factor_signs={name: signs[name] for name in ranked},
        regime_filter=cfg.regime_filter,
        config=cfg,
    )


def write_strict_factor_search_result(
    result: StrictFactorSearchResult,
    *,
    factor_frame: pd.DataFrame,
    market_panel: pd.DataFrame,
    sector_map: pd.DataFrame | None,
    sector_pool: pd.DataFrame | None,
    output_dir: Path,
) -> dict[str, Path]:
    """Write search summary, trials, best composite, weights, and backtest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["summary"] = output_dir / "strict_factor_search.json"
    paths["summary"].write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["trials"] = output_dir / "strict_factor_trials.csv"
    pd.DataFrame([trial.as_row() for trial in result.trials]).to_csv(paths["trials"], index=False)
    filtered, _ = filter_factor_frame_by_regime(factor_frame, market_panel, result.regime_filter)
    ev = evaluate_strict_factor_subset(
        factor_frame=filtered,
        factors=result.best_factors,
        top_k=result.best_top_k,
        market_panel=prepare_decision_chain_panel(market_panel, result.config.decision, sector_map=sector_map),
        sector_map=sector_map,
        sector_pool=sector_pool,
        config=result.config,
        factor_signs=result.factor_signs,
        write_backtest=True,
    )
    paths["composite"] = output_dir / "best_composite.parquet"
    ev.composite.to_parquet(paths["composite"], index=False)
    paths["target_weights"] = output_dir / "best_target_weights.parquet"
    ev.target_weights.to_parquet(paths["target_weights"])
    if ev.backtest is not None:
        backtest_dir = output_dir / "best_backtest"
        ev.backtest.write(backtest_dir)
        paths["backtest"] = backtest_dir
    return paths


def _score_factor_metrics(metrics: Mapping[str, object], config: StrictFactorSearchConfig) -> float:
    proxy = StrictPolicySearchConfig(
        return_weight=config.return_weight,
        excess_weight=config.excess_weight,
        drawdown_penalty=config.drawdown_penalty,
        turnover_penalty=config.turnover_penalty,
        cost_penalty=config.cost_penalty,
        initial_cash=config.decision.initial_cash,
    )
    return _score_metrics(metrics, proxy)


def _empty_metrics() -> dict[str, float]:
    return {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "max_drawdown": 0.0,
        "sharpe": 0.0,
        "calmar": 0.0,
        "turnover": 0.0,
        "total_cost": 0.0,
        "benchmark_equal_weight_ann": 0.0,
        "benchmark_equal_weight_total": 0.0,
        "excess_return_ann": 0.0,
    }


__all__ = [
    "StrictFactorEvaluation",
    "StrictFactorSearchConfig",
    "StrictFactorSearchResult",
    "StrictFactorTrial",
    "build_factor_composite",
    "evaluate_strict_factor_subset",
    "filter_factor_frame_by_regime",
    "infer_candidate_factors",
    "infer_factor_signs",
    "search_strict_factors",
    "write_strict_factor_search_result",
]
