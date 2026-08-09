"""Trace-proven StrictBacktestV8 production evidence wrapper.

The legacy reporting module remains the single implementation of v8 metrics and
PnL attribution. This wrapper owns only production-evidence orchestration: it
retains both the strict input target schedule and verified execution trace,
emits both artifacts, and cross-stamps their canonical identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from quantagent.backtest.ashare_execution_simulator import (
    AShareExecutionSimulationConfig,
    AShareExecutionSimulationResult,
    simulate_ashare_target_weights,
)
from quantagent.backtest.execution_timing import (
    EXECUTION_TIMING_SEMANTICS,
    execution_trace_sha256,
    signal_schedule_sha256,
    validate_execution_trace,
)
from quantagent.backtest.strict_v8 import (
    METRIC_SEMANTICS_VERSION,
    StrictBacktestArtifactSet,
    _compute_metrics,
    _profit_by_sector,
    _profit_by_stock,
    _realized_round_trip_pnl,
    quarantine_trust_stamp,
)


TRACE_PROVEN_STRICT_SEMANTICS = "strict_v8_trace_proven_t1_v1"


@dataclass
class TraceProvenStrictBacktestArtifactSet(StrictBacktestArtifactSet):
    target_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    execution_trace: pd.DataFrame = field(default_factory=pd.DataFrame)

    def write(self, output_dir: str | Path) -> dict[str, Path]:
        paths = super().write(output_dir)
        out = Path(output_dir)

        target_path = out / "target_weights.csv"
        canonical_targets = _canonical_target_weights(self.target_weights)
        canonical_targets.to_csv(target_path, index=True, index_label="signal_date")
        paths["target_weights"] = target_path

        trace_path = out / "execution_trace.csv"
        self.execution_trace.to_csv(trace_path, index=False)
        paths["execution_trace"] = trace_path

        metrics_path = paths["metrics"]
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["execution_timing_semantics"] = EXECUTION_TIMING_SEMANTICS
        payload["execution_trace_sha256"] = execution_trace_sha256(self.execution_trace)
        payload["execution_trace_artifact"] = trace_path.name
        payload["strict_target_weights_artifact"] = target_path.name
        payload["strict_target_signal_schedule_sha256"] = signal_schedule_sha256(
            canonical_targets.index
        )
        payload["strict_evidence_semantics"] = TRACE_PROVEN_STRICT_SEMANTICS
        metrics_path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        return paths


def run_trace_proven_strict_backtest_v8(
    target_weights: pd.DataFrame,
    market_panel: pd.DataFrame,
    *,
    sector_map: pd.DataFrame | None = None,
    factor_weights: Mapping[str, float] | None = None,
    config: AShareExecutionSimulationConfig | None = None,
) -> TraceProvenStrictBacktestArtifactSet:
    """Run strict v8 while retaining complete input/timing evidence."""
    if target_weights is None or target_weights.empty:
        raise ValueError("trace-proven strict evidence requires non-empty target_weights")
    canonical_targets = _canonical_target_weights(target_weights)
    cfg = config or AShareExecutionSimulationConfig()
    trust_stamp = quarantine_trust_stamp(pd.to_datetime(canonical_targets.index))
    sim: AShareExecutionSimulationResult = simulate_ashare_target_weights(
        canonical_targets,
        market_panel,
        cfg,
    )
    timing = validate_execution_trace(sim.execution_trace)
    if not timing.ok:
        raise ValueError(
            "strict trace-proven execution rejected: " + "; ".join(timing.reasons)
        )

    trace_signals = pd.to_datetime(
        sim.execution_trace.loc[
            sim.execution_trace["record_type"].eq("session_mapping"), "signal_date"
        ],
        errors="coerce",
    )
    target_schedule_sha = signal_schedule_sha256(canonical_targets.index)
    trace_schedule_sha = signal_schedule_sha256(trace_signals)
    if target_schedule_sha != trace_schedule_sha:
        raise ValueError(
            "strict trace-proven signal schedule mismatch: "
            f"{trace_schedule_sha}!={target_schedule_sha}"
        )

    metrics = _compute_metrics(sim.nav, sim.order_audit, initial_nav=cfg.initial_cash)
    realized_trades = _realized_round_trip_pnl(sim.order_audit)
    by_stock = _profit_by_stock(realized_trades, sim.order_audit)
    by_sector = _profit_by_sector(by_stock, sector_map)

    nav_series = sim.nav.copy() if sim.nav is not None else pd.Series(dtype=float)
    if not nav_series.empty:
        daily_return = nav_series.pct_change()
        daily_return.iloc[0] = float(nav_series.iloc[0] / cfg.initial_cash - 1.0)
        daily_pnl = daily_return.rename("daily_return").to_frame()
        daily_pnl["nav"] = nav_series.values
        daily_pnl = daily_pnl.reset_index().rename(columns={"index": "trade_date"})
    else:
        daily_pnl = pd.DataFrame(columns=["trade_date", "daily_return", "nav"])

    if sim.order_audit is not None and not sim.order_audit.empty:
        filled = sim.order_audit[sim.order_audit["filled_quantity"].astype(float).abs() > 0]
        if not filled.empty:
            selected = (
                filled.groupby("symbol")
                .agg(
                    first_filled=("trade_date", "min"),
                    last_filled=("trade_date", "max"),
                    n_fills=("symbol", "size"),
                )
                .reset_index()
            )
        else:
            selected = pd.DataFrame(columns=["symbol", "first_filled", "last_filled", "n_fills"])
    else:
        selected = pd.DataFrame(columns=["symbol", "first_filled", "last_filled", "n_fills"])

    trades = sim.order_audit.copy() if sim.order_audit is not None else pd.DataFrame()
    failed = sim.failed_order_audit.copy() if sim.failed_order_audit is not None else pd.DataFrame()
    artifact_config = dict(sim.config or {})
    artifact_config["metric_semantics_version"] = METRIC_SEMANTICS_VERSION
    artifact_config["nav_baseline"] = "configured_initial_cash"
    artifact_config["strict_evidence_semantics"] = TRACE_PROVEN_STRICT_SEMANTICS
    artifact_config["execution_trace_sha256"] = execution_trace_sha256(sim.execution_trace)
    artifact_config["strict_target_signal_schedule_sha256"] = target_schedule_sha
    artifact_config["execution_timing_semantics"] = EXECUTION_TIMING_SEMANTICS

    return TraceProvenStrictBacktestArtifactSet(
        metrics=metrics,
        nav=nav_series,
        daily_pnl=daily_pnl,
        selected_stocks=selected,
        trades=trades,
        failed_orders=failed,
        risk_events=list(sim.risk_events) if sim.risk_events else [],
        profit_by_stock=by_stock,
        profit_by_sector=by_sector,
        realized_trades=realized_trades,
        factor_weights=dict(factor_weights or {}),
        config=artifact_config,
        trust_stamp=trust_stamp,
        target_weights=canonical_targets,
        execution_trace=sim.execution_trace.copy(),
    )


def _canonical_target_weights(target_weights: pd.DataFrame) -> pd.DataFrame:
    frame = target_weights.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    if frame.index.isna().any():
        raise ValueError("strict target_weights contain invalid signal dates")
    frame.index = frame.index.normalize()
    if frame.index.duplicated().any():
        raise ValueError("strict target_weights contain duplicate signal dates")
    if frame.empty:
        raise ValueError("strict target_weights are empty")
    return frame.sort_index()


__all__ = [
    "TRACE_PROVEN_STRICT_SEMANTICS",
    "TraceProvenStrictBacktestArtifactSet",
    "run_trace_proven_strict_backtest_v8",
]
