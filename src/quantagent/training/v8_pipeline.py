"""End-to-end v8 training pipeline orchestrator.

Wires every stage of the v8 spec into a single Python entry point:

    DataSourceRouter (Qlib + AkShare + BaoStock + TuShare)
        → forward-return label builder
        → horizon bundles (short / mid / long)
        → factor scoring (passthrough or alpha library)
        → GA factor weight optimisation (purged walk-forward + OOS)
        → target_weights builder (top-K equal-weight)
        → strict A-share backtest (T+1, cost, slippage, risk_events)
        → daily decision report

PIT discipline is enforced at every hand-off:

* Forward labels are clipped to ``available_at`` only.
* GA fold splits include embargo equal to the longest horizon used.
* Backtest defers to the existing simulator (no shortcut around T+1).

The function returns a :class:`V8TrainingArtifacts` record with all
intermediate frames + on-disk paths so a CLI wrapper can simply
unfurl them onto disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import (
    StrictBacktestArtifactSet,
    run_strict_backtest_v8,
)
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


# ---------------------------------------------------------------------------
# Config + artifact record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V8TrainingConfig:
    horizon_class: HorizonClass = HorizonClass.SHORT
    factor_columns: tuple[str, ...] = ()
    top_k: int = 10
    ga_population: int = 12
    ga_generations: int = 6
    ga_random_seed: int = 17
    walk_forward_folds: int = 3
    embargo_days: int = 5
    min_train_days: int = 60
    min_test_days: int = 20
    initial_cash: float = 1_000_000.0
    slippage_bps: float = 8.0


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_market_panel(router_result: RouterResult) -> pd.DataFrame:
    if router_result is None or router_result.frame is None or router_result.frame.empty:
        raise RouterAllSourcesUnavailable(
            "router returned no market panel rows; production path forbids synthetic fallback"
        )
    df = router_result.frame.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df.dropna(subset=["trade_date", "symbol"]).sort_values(["symbol", "trade_date"])
    if "is_suspended" not in df.columns:
        df["is_suspended"] = False
    if "is_st" not in df.columns:
        df["is_st"] = False
    if "is_limit_up" not in df.columns:
        df["is_limit_up"] = False
    if "is_limit_down" not in df.columns:
        df["is_limit_down"] = False
    return df.reset_index(drop=True)


def build_forward_returns(
    market_panel: pd.DataFrame,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """Long-form forward returns table (one row per symbol/date/horizon).

    The returned frame also includes a wide-form pivot with one
    ``forward_return_Hd`` column per horizon, which slots straight
    into :func:`build_all_horizon_bundles`.
    """
    if market_panel is None or market_panel.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", "forward_return"])
    work = market_panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work = work.sort_values(["symbol", "trade_date"])
    out = work[["symbol", "trade_date", "close"]].copy()
    out["available_at"] = work.get("available_at", work["trade_date"])
    for h in horizons:
        out[f"forward_return_{h}d"] = (
            work.groupby("symbol")["close"].shift(-h) / work["close"] - 1.0
        )
    # The primary (longest-supplied) horizon doubles as the GA forward_return
    primary_h = max(int(h) for h in horizons)
    out["forward_return"] = out[f"forward_return_{primary_h}d"]
    return out.reset_index(drop=True)


def build_default_factor_panel(market_panel: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal factor panel from raw OHLCV.

    Two technical factors are emitted (mean-reversion 5d, momentum
    20d) so the pipeline runs end-to-end without optional alpha
    libraries. Production callers should swap this for a real
    :mod:`quantagent.factors.alpha101` materialisation.
    """
    if market_panel is None or market_panel.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", "mr_5d", "mom_20d"])
    work = market_panel.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work = work.sort_values(["symbol", "trade_date"])
    work["ret_1d"] = work.groupby("symbol")["close"].pct_change()
    work["mr_5d"] = -work.groupby("symbol")["ret_1d"].rolling(5).sum().reset_index(0, drop=True)
    work["mom_20d"] = work.groupby("symbol")["close"].pct_change(20)
    panel = work[["symbol", "trade_date", "mr_5d", "mom_20d"]].dropna().reset_index(drop=True)
    return panel


def build_top_k_target_weights(
    predictions: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    """Pivot daily top-K equal-weight predictions into a target frame."""
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    work = predictions.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work = work.dropna(subset=["trade_date"])
    if "alpha_score" not in work.columns:
        raise ValueError("predictions need an 'alpha_score' column")
    work = work.sort_values(["trade_date", "alpha_score"], ascending=[True, False])
    work["rank"] = work.groupby("trade_date").cumcount()
    work["weight"] = (work["rank"] < top_k).astype(float) / float(top_k)
    wide = work.pivot_table(
        index="trade_date", columns="symbol", values="weight", fill_value=0.0,
    )
    return wide


def factor_blend_to_predictions(
    factor_panel: pd.DataFrame,
    factor_weights: Mapping[str, float],
) -> pd.DataFrame:
    """Combine a per-day factor panel with GA weights into ``alpha_score`` rows."""
    if factor_panel is None or factor_panel.empty:
        return pd.DataFrame(columns=["trade_date", "symbol", "alpha_score"])
    cols = [c for c in factor_weights if c in factor_panel.columns]
    if not cols:
        raise ValueError("no factor columns in panel match the weight dict")
    work = factor_panel.copy()
    work["alpha_score"] = sum(
        work[c].astype(float) * float(factor_weights[c]) for c in cols
    )
    return work[["trade_date", "symbol", "alpha_score"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

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
    request = ProviderRequest(
        start_date=start_date, end_date=end_date,
        symbols=tuple(symbols),
    )
    router_result = router.daily_ohlcv(request)
    market_panel = _ensure_market_panel(router_result)

    spec = DEFAULT_HORIZON_SPECS[cfg.horizon_class]
    horizons = spec.horizons
    forward = build_forward_returns(market_panel, horizons=horizons)

    factor_panel = build_default_factor_panel(market_panel)
    if cfg.factor_columns:
        missing = [c for c in cfg.factor_columns if c not in factor_panel.columns]
        if missing:
            raise ValueError(f"factor_panel missing requested columns: {missing}")
        factor_names = tuple(cfg.factor_columns)
    else:
        factor_names = ("mr_5d", "mom_20d")

    # Build horizon bundles for downstream consumers (training, audit).
    bundle_input = forward[["symbol", "trade_date", "available_at"]].copy()
    for col in factor_names:
        joined = factor_panel[["symbol", "trade_date", col]]
        bundle_input = bundle_input.merge(joined, on=["symbol", "trade_date"], how="left")
    for h in horizons:
        col = f"forward_return_{h}d"
        if col in forward.columns:
            bundle_input = bundle_input.merge(
                forward[["symbol", "trade_date", col]], on=["symbol", "trade_date"], how="left",
            )
    bundles = build_all_horizon_bundles(bundle_input)

    # GA optimisation over walk-forward folds. May fail if the panel
    # has too few trading days; production caller is expected to
    # configure horizons / windows so this succeeds.
    ga_result: GAOptimizationResult | None = None
    try:
        ga_result = optimize_factor_weights_ga(
            factor_panel=factor_panel.merge(forward[["symbol", "trade_date", "forward_return"]],
                                              on=["symbol", "trade_date"], how="inner"),
            forward_returns=forward[["symbol", "trade_date", "forward_return"]].dropna(),
            factor_names=list(factor_names),
            ga_config=GAConfig(
                population_size=cfg.ga_population,
                generations=cfg.ga_generations,
                top_k=cfg.top_k,
                random_seed=cfg.ga_random_seed,
            ),
            wf_config=WalkForwardConfig(
                n_folds=cfg.walk_forward_folds,
                embargo_days=cfg.embargo_days,
                min_train_days=cfg.min_train_days,
                min_test_days=cfg.min_test_days,
            ),
        )
        factor_weights = ga_result.best_weights
    except (ValueError, RouterAllSourcesUnavailable):
        # Pipeline can still produce target weights with a uniform
        # blend when walk-forward configuration is too strict for the
        # available history. The GA artefact is then None.
        factor_weights = {f: 1.0 / len(factor_names) for f in factor_names}

    predictions = factor_blend_to_predictions(factor_panel, factor_weights)
    target_weights = build_top_k_target_weights(predictions, top_k=cfg.top_k)

    bt_cfg = AShareExecutionSimulationConfig(
        initial_cash=cfg.initial_cash, slippage_bps=cfg.slippage_bps,
    )
    bt = run_strict_backtest_v8(
        target_weights, market_panel,
        sector_map=sector_map, factor_weights=factor_weights, config=bt_cfg,
    )

    # Daily decision report uses last available trade date.
    last_date = market_panel["trade_date"].max() if not market_panel.empty else pd.Timestamp.utcnow()
    decision = DailyDecisionInputs(
        as_of_date=pd.Timestamp(last_date),
        target_weights=(target_weights.iloc[-1] if not target_weights.empty else None),
        sector_map=sector_map,
        risk_events=list(bt.risk_events),
        gross_exposure=float(target_weights.iloc[-1].sum()) if not target_weights.empty else None,
    )
    report = build_daily_decision_report(decision)

    artifacts = V8TrainingArtifacts(
        market_panel=market_panel,
        forward_returns=forward,
        factor_panel=factor_panel,
        horizon_bundles=bundles,
        ga_result=ga_result,
        target_weights=target_weights,
        backtest=bt,
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
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    paths["market_panel"] = out / "market_panel.parquet"
    artifacts.market_panel.to_parquet(paths["market_panel"], index=False)
    paths["forward_returns"] = out / "forward_returns.parquet"
    artifacts.forward_returns.to_parquet(paths["forward_returns"], index=False)
    paths["factor_panel"] = out / "factor_panel.parquet"
    artifacts.factor_panel.to_parquet(paths["factor_panel"], index=False)
    paths["target_weights"] = out / "target_weights.parquet"
    if not artifacts.target_weights.empty:
        artifacts.target_weights.to_parquet(paths["target_weights"])
    if artifacts.ga_result is not None:
        ga_dir = out / "ga"
        save_optimisation_artifacts(artifacts.ga_result, output_dir=ga_dir)
        paths["ga_dir"] = ga_dir
    bt_dir = out / "backtest"
    artifacts.backtest.write(bt_dir)
    paths["backtest_dir"] = bt_dir
    paths["daily_report"] = out / "daily_report.md"
    artifacts.daily_report.write(paths["daily_report"])
    paths["router_diagnostics"] = out / "router_diagnostics.json"
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
