"""Legacy v8 GA training pipeline with evaluation-integrity guards.

Production training is the FT-Transformer sleeve path in ``cli/v8_deep.py``
and ``configs/production_blend.json``. This module remains available for
research and regression tests, but it must not manufacture optimistic metrics
from overlapping forward labels or silently replace a failed optimiser with a
uniform blend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import StrictBacktestArtifactSet, run_strict_backtest_v8
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.router import (
    MultiSourceDataRouter,
    RouterAllSourcesUnavailable,
    RouterResult,
)
from quantagent.diagnostics.daily_decision_report import (
    DailyDecisionInputs,
    DailyDecisionReport,
    build_daily_decision_report,
)
from quantagent.optimization.ga_weight_optimizer import (
    GAConfig,
    GAOptimizationResult,
    WalkForwardConfig,
    optimize_factor_weights_ga,
    save_optimisation_artifacts,
)
from quantagent.training.horizon_models import (
    DEFAULT_HORIZON_SPECS,
    HorizonBundle,
    HorizonClass,
    build_all_horizon_bundles,
)


@dataclass(frozen=True)
class V8TrainingConfig:
    horizon_class: HorizonClass = HorizonClass.SHORT
    factor_columns: tuple[str, ...] = ()
    top_k: int = 10
    ga_population: int = 12
    ga_generations: int = 6
    ga_random_seed: int = 17
    ga_min_label_coverage: float = 0.80
    ga_min_cohort_observations: int = 2
    walk_forward_folds: int = 3
    embargo_days: int = 5
    min_train_days: int = 60
    min_test_days: int = 20
    initial_cash: float = 1_000_000.0
    slippage_bps: float = 8.0
    allow_uniform_fallback: bool = False


@dataclass
class V8TrainingArtifacts:
    market_panel: pd.DataFrame
    forward_returns: pd.DataFrame
    factor_panel: pd.DataFrame
    horizon_bundles: dict[HorizonClass, HorizonBundle]
    ga_result: GAOptimizationResult | None
    target_weights: pd.DataFrame
    backtest: StrictBacktestArtifactSet
    daily_report: DailyDecisionReport
    router_diagnostics: dict[str, dict[str, object]] = field(default_factory=dict)


def _ensure_market_panel(router_result: RouterResult) -> pd.DataFrame:
    if router_result is None or router_result.frame is None or router_result.frame.empty:
        raise RouterAllSourcesUnavailable(
            "router returned no market panel rows; production path forbids synthetic fallback"
        )
    frame = router_result.frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"])
    for column in ("is_suspended", "is_st", "is_limit_up", "is_limit_down"):
        if column not in frame.columns:
            frame[column] = False
    return frame.reset_index(drop=True)


def build_forward_returns(
    market_panel: pd.DataFrame,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """Build symbol/date forward labels and record their primary horizon."""
    horizons = tuple(sorted({int(value) for value in horizons}))
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("horizons must contain positive integers")
    if market_panel is None or market_panel.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", "forward_return"])

    work = market_panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work = work.dropna(subset=["trade_date", "symbol", "close"]).sort_values(
        ["symbol", "trade_date"]
    )
    output = work[["symbol", "trade_date", "close"]].copy()
    output["available_at"] = work.get("available_at", work["trade_date"])
    grouped_close = work.groupby("symbol", sort=False)["close"]
    for horizon in horizons:
        output[f"forward_return_{horizon}d"] = grouped_close.shift(-horizon) / work["close"] - 1.0

    primary_horizon = max(horizons)
    output["forward_return"] = output[f"forward_return_{primary_horizon}d"]
    output["forward_horizon_days"] = primary_horizon
    output.attrs["forward_horizon_days"] = primary_horizon
    return output.reset_index(drop=True)


def build_default_factor_panel(market_panel: pd.DataFrame) -> pd.DataFrame:
    """Build two minimal price factors for legacy-pipeline smoke tests."""
    if market_panel is None or market_panel.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", "mr_5d", "mom_20d"])
    work = market_panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work = work.dropna(subset=["trade_date", "symbol", "close"]).sort_values(
        ["symbol", "trade_date"]
    )
    work["ret_1d"] = work.groupby("symbol", sort=False)["close"].pct_change()
    work["mr_5d"] = -(
        work.groupby("symbol", sort=False)["ret_1d"]
        .rolling(5)
        .sum()
        .reset_index(level=0, drop=True)
    )
    work["mom_20d"] = work.groupby("symbol", sort=False)["close"].pct_change(20)
    return work[["symbol", "trade_date", "mr_5d", "mom_20d"]].dropna().reset_index(drop=True)


def build_top_k_target_weights(
    predictions: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    """Create fully invested equal weights among the available daily top-K."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    work = predictions.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    if "alpha_score" not in work.columns:
        raise ValueError("predictions need an 'alpha_score' column")
    work["alpha_score"] = pd.to_numeric(work["alpha_score"], errors="coerce")
    work = work.dropna(subset=["trade_date", "symbol", "alpha_score"])
    if work.empty:
        return pd.DataFrame()
    if work.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("predictions contain duplicate trade_date/symbol keys")

    work = work.sort_values(
        ["trade_date", "alpha_score", "symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    work["rank"] = work.groupby("trade_date", sort=False).cumcount()
    selected = work[work["rank"] < top_k].copy()
    selected_count = selected.groupby("trade_date")["symbol"].transform("size")
    selected["weight"] = 1.0 / selected_count.astype(float)
    return selected.pivot(
        index="trade_date", columns="symbol", values="weight"
    ).fillna(0.0).sort_index()


def factor_blend_to_predictions(
    factor_panel: pd.DataFrame,
    factor_weights: Mapping[str, float],
) -> pd.DataFrame:
    if factor_panel is None or factor_panel.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "alpha_score"])
    columns = [name for name in factor_weights if name in factor_panel.columns]
    if not columns:
        raise ValueError("no factor columns in panel match the weight dict")
    work = factor_panel.copy()
    for column in columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["trade_date", "symbol", *columns])
    work["alpha_score"] = sum(
        work[column] * float(factor_weights[column]) for column in columns
    )
    return work[["trade_date", "symbol", "alpha_score"]].reset_index(drop=True)


def _build_horizon_bundles(
    forward: pd.DataFrame,
    factor_panel: pd.DataFrame,
    factor_names: tuple[str, ...],
    horizons: tuple[int, ...],
) -> dict[HorizonClass, HorizonBundle]:
    bundle_input = forward[["symbol", "trade_date", "available_at"]].copy()
    factor_slice = factor_panel[["symbol", "trade_date", *factor_names]]
    bundle_input = bundle_input.merge(
        factor_slice, on=["symbol", "trade_date"], how="left", validate="one_to_one"
    )
    label_columns = [f"forward_return_{horizon}d" for horizon in horizons]
    bundle_input = bundle_input.merge(
        forward[["symbol", "trade_date", *label_columns]],
        on=["symbol", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    return build_all_horizon_bundles(bundle_input)


def run_v8_training_pipeline(
    *,
    router: MultiSourceDataRouter,
    symbols: tuple[str, ...],
    start_date: str,
    end_date: str,
    config: V8TrainingConfig | None = None,
    sector_map: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
) -> V8TrainingArtifacts:
    cfg = config or V8TrainingConfig()
    router_result = router.daily_ohlcv(
        ProviderRequest(start_date=start_date, end_date=end_date, symbols=tuple(symbols))
    )
    market_panel = _ensure_market_panel(router_result)

    spec = DEFAULT_HORIZON_SPECS[cfg.horizon_class]
    horizons = tuple(int(value) for value in spec.horizons)
    primary_horizon = max(horizons)
    forward = build_forward_returns(market_panel, horizons)
    factor_panel = build_default_factor_panel(market_panel)

    factor_names = tuple(cfg.factor_columns) if cfg.factor_columns else ("mr_5d", "mom_20d")
    missing = [name for name in factor_names if name not in factor_panel.columns]
    if missing:
        raise ValueError(f"factor_panel missing requested columns: {missing}")
    bundles = _build_horizon_bundles(forward, factor_panel, factor_names, horizons)

    ga_result: GAOptimizationResult | None
    try:
        ga_result = optimize_factor_weights_ga(
            factor_panel=factor_panel,
            forward_returns=forward[["symbol", "trade_date", "forward_return"]].dropna(),
            factor_names=factor_names,
            ga_config=GAConfig(
                population_size=cfg.ga_population,
                generations=cfg.ga_generations,
                top_k=cfg.top_k,
                random_seed=cfg.ga_random_seed,
                label_horizon_days=primary_horizon,
                transaction_cost_bps=cfg.slippage_bps,
                min_label_coverage=cfg.ga_min_label_coverage,
                min_cohort_observations=cfg.ga_min_cohort_observations,
            ),
            wf_config=WalkForwardConfig(
                n_folds=cfg.walk_forward_folds,
                embargo_days=max(cfg.embargo_days, primary_horizon),
                label_horizon_days=primary_horizon,
                min_train_days=max(
                    cfg.min_train_days,
                    primary_horizon * cfg.ga_min_cohort_observations,
                ),
                min_test_days=max(
                    cfg.min_test_days,
                    primary_horizon * cfg.ga_min_cohort_observations,
                ),
            ),
        )
        factor_weights = ga_result.best_weights
    except (ValueError, RouterAllSourcesUnavailable) as error:
        if not cfg.allow_uniform_fallback:
            raise RuntimeError(
                "legacy v8 GA failed its integrity gates; refusing silent uniform fallback"
            ) from error
        ga_result = None
        factor_weights = {name: 1.0 / len(factor_names) for name in factor_names}

    predictions = factor_blend_to_predictions(factor_panel, factor_weights)
    target_weights = build_top_k_target_weights(predictions, top_k=cfg.top_k)
    backtest = run_strict_backtest_v8(
        target_weights,
        market_panel,
        sector_map=sector_map,
        factor_weights=factor_weights,
        config=AShareExecutionSimulationConfig(
            initial_cash=cfg.initial_cash,
            slippage_bps=cfg.slippage_bps,
        ),
    )

    last_date = market_panel["trade_date"].max()
    report = build_daily_decision_report(
        DailyDecisionInputs(
            as_of_date=pd.Timestamp(last_date),
            target_weights=(target_weights.iloc[-1] if not target_weights.empty else None),
            sector_map=sector_map,
            risk_events=list(backtest.risk_events),
            gross_exposure=(
                float(target_weights.iloc[-1].sum()) if not target_weights.empty else None
            ),
        )
    )
    artifacts = V8TrainingArtifacts(
        market_panel=market_panel,
        forward_returns=forward,
        factor_panel=factor_panel,
        horizon_bundles=bundles,
        ga_result=ga_result,
        target_weights=target_weights,
        backtest=backtest,
        daily_report=report,
        router_diagnostics=router_result.to_dict(),
    )
    if output_dir is not None:
        write_training_artifacts(artifacts, output_dir=output_dir)
    return artifacts


def write_training_artifacts(
    artifacts: V8TrainingArtifacts,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "market_panel": output / "market_panel.parquet",
        "forward_returns": output / "forward_returns.parquet",
        "factor_panel": output / "factor_panel.parquet",
        "target_weights": output / "target_weights.parquet",
        "backtest_dir": output / "backtest",
        "daily_report": output / "daily_report.md",
        "router_diagnostics": output / "router_diagnostics.json",
    }
    artifacts.market_panel.to_parquet(paths["market_panel"], index=False)
    artifacts.forward_returns.to_parquet(paths["forward_returns"], index=False)
    artifacts.factor_panel.to_parquet(paths["factor_panel"], index=False)
    if not artifacts.target_weights.empty:
        artifacts.target_weights.to_parquet(paths["target_weights"])
    if artifacts.ga_result is not None:
        ga_dir = output / "ga"
        save_optimisation_artifacts(artifacts.ga_result, output_dir=ga_dir)
        paths["ga_dir"] = ga_dir
    artifacts.backtest.write(paths["backtest_dir"])
    artifacts.daily_report.write(paths["daily_report"])
    paths["router_diagnostics"].write_text(
        json.dumps(artifacts.router_diagnostics, indent=2, default=str),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "V8TrainingArtifacts",
    "V8TrainingConfig",
    "build_default_factor_panel",
    "build_forward_returns",
    "build_top_k_target_weights",
    "factor_blend_to_predictions",
    "run_v8_training_pipeline",
    "write_training_artifacts",
]
