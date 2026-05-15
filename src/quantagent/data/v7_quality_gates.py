"""Hard quality and acceptance gates for V7 real-data training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class V7DataQualityGateConfig:
    min_rows: int = 1_000
    min_symbols: int = 50
    min_dates: int = 120
    require_real_data: bool = True
    max_single_factor_dominance: float = 0.60


@dataclass(frozen=True)
class V7ModelAcceptanceGateConfig:
    min_rank_ic_mean: float = 0.0
    min_rank_ic_stability: float = 0.0
    min_turnover_adjusted_return: float = 0.0
    max_drawdown: float = 0.25
    max_single_factor_dominance: float = 0.60
    require_adverse_regime: bool = True
    require_paper_report: bool = True


@dataclass(frozen=True)
class V7GateReport:
    passed: bool
    failures: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failures"] = list(self.failures)
        return data


def evaluate_data_quality_gates(frame: pd.DataFrame, config: V7DataQualityGateConfig | None = None) -> V7GateReport:
    config = config or V7DataQualityGateConfig()
    failures: list[str] = []
    metrics = {
        "row_count": int(0 if frame is None else len(frame)),
        "symbol_count": int(frame["symbol"].nunique()) if frame is not None and "symbol" in frame.columns else 0,
        "date_count": int(pd.to_datetime(frame["trade_date"], errors="coerce").nunique()) if frame is not None and "trade_date" in frame.columns else 0,
        "pit_violation_count": _pit_violations(frame),
        "uses_mock_or_synthetic": _uses_mock_or_synthetic(frame),
    }
    if metrics["pit_violation_count"] > 0:
        failures.append("pit_violations_present")
    if metrics["row_count"] < config.min_rows:
        failures.append("insufficient_training_rows")
    if metrics["symbol_count"] < config.min_symbols:
        failures.append("insufficient_symbol_coverage")
    if metrics["date_count"] < config.min_dates:
        failures.append("insufficient_date_coverage")
    if config.require_real_data and metrics["uses_mock_or_synthetic"]:
        failures.append("mock_or_synthetic_data_not_production_ready")
    return V7GateReport(not failures, tuple(failures), metrics)


def evaluate_model_acceptance_gates(
    metrics: dict[str, Any],
    config: V7ModelAcceptanceGateConfig | None = None,
    paper_report_path: str | Path | None = None,
) -> V7GateReport:
    config = config or V7ModelAcceptanceGateConfig()
    failures: list[str] = []
    if float(metrics.get("rank_ic_mean", 0.0)) <= config.min_rank_ic_mean:
        failures.append("rank_ic_mean_not_positive")
    if float(metrics.get("rank_ic_stability", 0.0)) <= config.min_rank_ic_stability:
        failures.append("rank_ic_stability_not_positive")
    if float(metrics.get("turnover_adjusted_net_return", 0.0)) <= config.min_turnover_adjusted_return:
        failures.append("turnover_adjusted_net_return_failed")
    if abs(float(metrics.get("max_drawdown", 0.0))) > config.max_drawdown:
        failures.append("max_drawdown_exceeded")
    if float(metrics.get("single_factor_dominance", 0.0)) > config.max_single_factor_dominance:
        failures.append("single_factor_dominance_too_high")
    if config.require_adverse_regime and not bool(metrics.get("adverse_regime_passed", False)):
        failures.append("adverse_regime_not_validated")
    if config.require_paper_report and not (paper_report_path and Path(paper_report_path).exists()):
        failures.append("paper_trading_report_missing")
    if bool(metrics.get("uses_mock_or_synthetic", False)):
        failures.append("mock_data_model_not_production_ready")
    return V7GateReport(not failures, tuple(failures), dict(metrics))


def _pit_violations(frame: pd.DataFrame | None) -> int:
    if frame is None or frame.empty or "available_at" not in frame.columns:
        return 0
    reference = "as_of_date" if "as_of_date" in frame.columns else "inference_date" if "inference_date" in frame.columns else ""
    if not reference:
        invalid = 0
        if "point_in_time_valid" in frame.columns:
            invalid = int((~frame["point_in_time_valid"].fillna(False).astype(bool)).sum())
        return invalid
    date_violations = int((pd.to_datetime(frame["available_at"], errors="coerce") > pd.to_datetime(frame[reference], errors="coerce")).sum())
    invalid = int((~frame["point_in_time_valid"].fillna(False).astype(bool)).sum()) if "point_in_time_valid" in frame.columns else 0
    return date_violations + invalid


def _uses_mock_or_synthetic(frame: pd.DataFrame | None) -> bool:
    if frame is None or frame.empty:
        return False
    for column in ("source", "source_name", "data_source"):
        if column in frame.columns:
            values = frame[column].astype(str).str.lower()
            if values.str.contains("mock|synthetic|demo").any():
                return True
    return False
