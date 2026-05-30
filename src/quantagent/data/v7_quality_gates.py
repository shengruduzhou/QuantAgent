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
    max_drawdown: float = 0.10
    max_single_factor_dominance: float = 0.60
    require_adverse_regime: bool = True
    require_paper_report: bool = True
    require_benchmark: bool = True
    min_excess_return_after_costs: float = 0.0
    min_selection_pressure: float = 3.0
    min_training_symbols: int = 50
    min_prediction_symbols: int = 50
    min_effective_universe_by_date: int = 50
    no_mock_or_synthetic: bool = True
    no_pit_violations: bool = True
    adverse_regime_min_rank_ic: float = -0.02
    adverse_regime_max_drawdown: float = 0.40


@dataclass(frozen=True)
class V7GateReport:
    passed: bool
    failures: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    gates: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["failures"] = list(self.failures)
        data["gates"] = list(self.gates)
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
    gates: list[dict[str, Any]] = []

    def add_gate(name: str, passed: bool, actual: Any, threshold: Any, reason: str) -> None:
        gates.append(
            {
                "name": name,
                "passed": bool(passed),
                "actual": actual,
                "threshold": threshold,
                "reason": "passed" if passed else reason,
            }
        )

    add_gate(
        "rank_ic_mean",
        float(metrics.get("rank_ic_mean", 0.0)) > config.min_rank_ic_mean,
        float(metrics.get("rank_ic_mean", 0.0)),
        f"> {config.min_rank_ic_mean}",
        "rank_ic_mean_not_positive",
    )
    add_gate(
        "rank_ic_stability",
        float(metrics.get("rank_ic_stability", metrics.get("ICIR", 0.0))) > config.min_rank_ic_stability,
        float(metrics.get("rank_ic_stability", metrics.get("ICIR", 0.0))),
        f"> {config.min_rank_ic_stability}",
        "rank_ic_stability_not_positive",
    )
    add_gate(
        "turnover_adjusted_net_return",
        float(metrics.get("turnover_adjusted_net_return", 0.0)) > config.min_turnover_adjusted_return,
        float(metrics.get("turnover_adjusted_net_return", 0.0)),
        f"> {config.min_turnover_adjusted_return}",
        "turnover_adjusted_net_return_failed",
    )
    add_gate(
        "max_drawdown",
        abs(float(metrics.get("max_drawdown", 0.0))) <= config.max_drawdown,
        float(metrics.get("max_drawdown", 0.0)),
        f"abs(drawdown) <= {config.max_drawdown}",
        "max_drawdown_exceeded",
    )
    add_gate(
        "single_factor_dominance",
        float(metrics.get("single_factor_dominance", 0.0)) <= config.max_single_factor_dominance,
        float(metrics.get("single_factor_dominance", 0.0)),
        f"<= {config.max_single_factor_dominance}",
        "single_factor_dominance_too_high",
    )
    adverse_actual = bool(metrics.get("adverse_regime_passed", False))
    add_gate(
        "adverse_regime",
        (not config.require_adverse_regime) or adverse_actual,
        adverse_actual,
        f"required={config.require_adverse_regime}, min_rank_ic={config.adverse_regime_min_rank_ic}",
        "adverse_regime_not_validated",
    )
    add_gate(
        "paper_report",
        (not config.require_paper_report) or bool(paper_report_path and Path(paper_report_path).exists()),
        str(paper_report_path) if paper_report_path else None,
        f"required={config.require_paper_report}",
        "paper_trading_report_missing",
    )
    has_benchmark = bool(metrics.get("benchmark_symbol")) or metrics.get("benchmark_return") is not None
    add_gate(
        "benchmark",
        (not config.require_benchmark) or has_benchmark,
        metrics.get("benchmark_symbol") or metrics.get("benchmark_return"),
        f"required={config.require_benchmark}",
        "benchmark_missing_quant_alpha_not_validated",
    )
    def _safe_float(key_chain: tuple[str, ...], default: float = 0.0) -> float:
        for key in key_chain:
            value = metrics.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return float(default)

    def _safe_int(key_chain: tuple[str, ...], default: int = 0) -> int:
        for key in key_chain:
            value = metrics.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return int(default)

    excess_after_costs = _safe_float(("excess_return_after_costs", "excess_return"))
    add_gate(
        "excess_return_after_costs",
        excess_after_costs > config.min_excess_return_after_costs,
        excess_after_costs,
        f"> {config.min_excess_return_after_costs}",
        "excess_return_after_costs_failed",
    )
    selection_pressure_actual = _safe_float(("selection_pressure_min", "selection_pressure"))
    add_gate(
        "selection_pressure",
        selection_pressure_actual >= config.min_selection_pressure,
        selection_pressure_actual,
        f">= {config.min_selection_pressure}",
        "selection_pressure_too_low",
    )
    training_symbols_actual = _safe_int((
        "training_dataset_symbol_count",
        "training_symbol_count",
        "symbol_count",
    ))
    add_gate(
        "training_symbols",
        training_symbols_actual >= config.min_training_symbols,
        training_symbols_actual,
        f">= {config.min_training_symbols}",
        "insufficient_training_symbols",
    )
    prediction_symbols_actual = _safe_int(("prediction_symbol_count",))
    add_gate(
        "prediction_symbols",
        prediction_symbols_actual >= config.min_prediction_symbols,
        prediction_symbols_actual,
        f">= {config.min_prediction_symbols}",
        "insufficient_prediction_symbols",
    )
    effective_universe = _safe_int(("effective_universe_min", "eligible_symbol_count_min"))
    add_gate(
        "effective_universe_by_date",
        effective_universe >= config.min_effective_universe_by_date,
        effective_universe,
        f">= {config.min_effective_universe_by_date}",
        "insufficient_effective_universe_by_date",
    )
    uses_mock = bool(metrics.get("uses_mock_or_synthetic", False))
    add_gate(
        "no_mock_or_synthetic",
        (not config.no_mock_or_synthetic) or not uses_mock,
        uses_mock,
        "False",
        "mock_data_model_not_production_ready",
    )
    pit_violations = int(metrics.get("pit_violation_count", 0))
    add_gate(
        "no_pit_violations",
        (not config.no_pit_violations) or pit_violations == 0,
        pit_violations,
        "0",
        "pit_violations_present",
    )
    failures = [str(gate["reason"]) for gate in gates if not gate["passed"]]
    return V7GateReport(not failures, tuple(failures), dict(metrics), tuple(gates))


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


def evaluate_adverse_regime(
    predictions: pd.DataFrame | None,
    market_panel: pd.DataFrame | None = None,
    label_column: str = "forward_return_1d",
    config: V7ModelAcceptanceGateConfig | None = None,
) -> dict[str, Any]:
    """Score the model in adverse regimes.

    Adverse regime is defined as trading days where the cross-sectional
    market return is in the bottom-quartile of the prediction window.
    We compute the rank-IC inside that subset and compare with the
    ``adverse_regime_*`` thresholds in ``V7ModelAcceptanceGateConfig``.

    Falls back to ``passed=False`` (not ``True``) when there is not
    enough data to evaluate — silent passes are no longer allowed.
    """
    config = config or V7ModelAcceptanceGateConfig()
    report: dict[str, Any] = {
        "passed": False,
        "reason": "insufficient_data",
        "adverse_dates_count": 0,
        "adverse_rank_ic_mean": 0.0,
    }
    if predictions is None or predictions.empty:
        return report
    if label_column not in predictions.columns or "prediction" not in predictions.columns:
        report["reason"] = "missing_prediction_or_label_columns"
        return report
    data = predictions.copy()
    data["trade_date"] = pd.to_datetime(data.get("trade_date"), errors="coerce")
    data = data.dropna(subset=["trade_date", "prediction", label_column])
    if data.empty:
        return report
    daily_return = data.groupby("trade_date")[label_column].mean()
    if daily_return.empty:
        return report
    threshold = daily_return.quantile(0.25)
    adverse_dates = daily_return[daily_return <= threshold].index
    if len(adverse_dates) == 0:
        report["reason"] = "no_adverse_dates"
        return report
    subset = data[data["trade_date"].isin(adverse_dates)]
    by_date_ic = subset.groupby("trade_date").apply(
        lambda f: float(f["prediction"].rank().corr(f[label_column].rank()))
        if len(f) >= 2 and f["prediction"].nunique() >= 2 and f[label_column].nunique() >= 2
        else float("nan")
    ).dropna()
    rank_ic_mean = float(by_date_ic.mean()) if not by_date_ic.empty else 0.0
    report["adverse_dates_count"] = int(len(adverse_dates))
    report["adverse_rank_ic_mean"] = rank_ic_mean
    report["passed"] = bool(rank_ic_mean >= config.adverse_regime_min_rank_ic)
    report["reason"] = "evaluated"
    return report
