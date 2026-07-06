"""Daily V7 paper loop: evidence refresh, alpha, target weights, report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
import json
from pathlib import Path

import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig, simulate_ashare_target_weights
from quantagent.backtest.paper_report import PaperReportConfig, write_paper_report
from quantagent.cli._utils import read_frame, write_frame
from quantagent.config.paths import quant_paths
from quantagent.data.ingestion.daily_evidence_job import DailyEvidenceJob, DailyEvidenceJobConfig
from quantagent.portfolio.multi_horizon_blender import MultiHorizonBlendConfig, blend_multi_horizon_predictions
from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights, write_v7_target_weights
from quantagent.training.v7_predictor import predict_v7_alpha


@dataclass(frozen=True)
class DailyPaperLoopConfig:
    as_of_date: str
    model_dir: str = field(default_factory=lambda: str(quant_paths().models / "v7_alpha"))
    feature_dataset_path: str = field(default_factory=lambda: str(quant_paths().data_root / "v7" / "gold" / "training_dataset" / "training_dataset.parquet"))
    market_panel_path: str = field(default_factory=lambda: str(quant_paths().data_root / "v7" / "silver" / "market_panel" / "market_panel.parquet"))
    sector_map_path: str | None = None
    output_root: str = field(default_factory=lambda: str(quant_paths().reports / "v7" / "paper"))
    paper_book_path: str = field(default_factory=lambda: str(quant_paths().home / "paper" / "paper_book.parquet"))
    primary_horizon: int = 5
    top_k: int = 30
    selection_mode: str = "ai_threshold"
    alpha_threshold: float = 0.0
    confidence_floor: float = 0.55
    selection_top_k_min: int = 5
    selection_top_k_max: int = 100
    max_weight_per_name: float = 0.10
    max_sector_weight: float = 0.30
    max_turnover: float = 0.40
    cost_bps: float = 12.0
    initial_cash: float = 1_000_000.0
    min_order_value_yuan: float = 100.0
    dry_run_evidence: bool = True


@dataclass(frozen=True)
class DailyPaperLoopResult:
    status: str
    as_of_date: str
    evidence_rows: int
    predictions_path: str
    target_weights_path: str
    paper_report_dir: str
    paper_book_path: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_once(config: DailyPaperLoopConfig) -> DailyPaperLoopResult:
    as_of = _normalise_date(config.as_of_date)
    paths = quant_paths().ensure()
    evidence = DailyEvidenceJob().run(
        DailyEvidenceJobConfig(as_of_date=as_of, dry_run=config.dry_run_evidence)
    )
    feature_path = Path(config.feature_dataset_path)
    if not feature_path.exists():
        raise FileNotFoundError(
            f"feature dataset not found: {feature_path}. The legacy default "
            "training_dataset.parquet was deleted 2026-07-06 (deletion manifest "
            "runtime/archives/deletion_manifests/batch4_20260706.json). Rebuild via "
            "`quantagent v7-data build-training-dataset-v7` or point "
            "feature_dataset_path at a current gold dataset."
        )
    feature_dataset = read_frame(feature_path)
    market_panel = read_frame(Path(config.market_panel_path))
    features_asof = _asof_slice(feature_dataset, as_of)
    if features_asof.empty:
        raise ValueError(f"no feature rows available at or before {as_of}")
    prediction_result = predict_v7_alpha(config.model_dir, features_asof, primary_horizon=config.primary_horizon)
    blend_result = blend_multi_horizon_predictions(
        prediction_result.predictions,
        config=MultiHorizonBlendConfig(primary_horizon=config.primary_horizon),
    )
    predictions = blend_result.blended if not blend_result.blended.empty else prediction_result.predictions
    day_dir = Path(config.output_root) / as_of
    predictions_path = write_frame(predictions, day_dir / "predictions.parquet")
    sector = read_frame(Path(config.sector_map_path)) if config.sector_map_path else None
    weights = build_v7_target_weights(
        predictions,
        market_panel,
        sector_map=sector,
        config=V7TargetWeightsConfig(
            top_k=config.top_k,
            selection_mode=config.selection_mode,
            alpha_threshold=config.alpha_threshold,
            confidence_floor=config.confidence_floor,
            selection_top_k_min=config.selection_top_k_min,
            selection_top_k_max=config.selection_top_k_max,
            max_weight_per_name=config.max_weight_per_name,
            max_sector_weight=config.max_sector_weight,
            max_turnover=config.max_turnover,
            cost_bps=config.cost_bps,
            shrink_on_small_universe=True,
            min_selection_pressure=1.0,
        ),
    )
    weights_path = write_v7_target_weights(weights, day_dir / "target_weights.parquet")
    sim = simulate_ashare_target_weights(
        _weights_for_sim(weights.target_weights),
        _market_for_dates(market_panel, weights.target_weights),
        AShareExecutionSimulationConfig(
            initial_cash=config.initial_cash,
            min_order_value_yuan=config.min_order_value_yuan,
            audit_log_dir=str(day_dir / "audit"),
        ),
    )
    report = write_paper_report(
        sim,
        market_panel=market_panel,
        config=PaperReportConfig(
            initial_cash=config.initial_cash,
            output_dir=day_dir,
            target_weights_path=str(weights_path),
        ),
    )
    paper_book_path = _append_paper_book(config.paper_book_path, as_of, weights.target_weights, report.summary)
    summary = {
        "config": asdict(config),
        "evidence_rows": int(len(evidence.frame)),
        "evidence_warnings": list(evidence.warnings),
        "blend_diagnostics": blend_result.diagnostics,
        "target_weight_diagnostics": weights.diagnostics,
        "paper_report": report.files,
        "paper_book_path": str(paper_book_path),
    }
    (day_dir / "daily_loop_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return DailyPaperLoopResult(
        status="passed",
        as_of_date=as_of,
        evidence_rows=int(len(evidence.frame)),
        predictions_path=str(predictions_path),
        target_weights_path=str(weights_path),
        paper_report_dir=str(day_dir),
        paper_book_path=str(paper_book_path),
        warnings=tuple(evidence.warnings),
    )


def _normalise_date(value: str) -> str:
    if value.lower() == "today":
        return date.today().isoformat()
    return pd.Timestamp(value).date().isoformat()


def _asof_slice(frame: pd.DataFrame, as_of: str) -> pd.DataFrame:
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    cutoff = pd.Timestamp(as_of)
    eligible = data[data["trade_date"] <= cutoff]
    if eligible.empty:
        return eligible
    latest = eligible["trade_date"].max()
    return eligible[eligible["trade_date"] == latest].reset_index(drop=True)


def _weights_for_sim(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        return weights
    frame = weights.copy()
    if "trade_date" in frame.columns:
        frame = frame.set_index("trade_date")
    return frame


def _market_for_dates(market: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty or "trade_date" not in weights.columns:
        return market
    dates = set(pd.to_datetime(weights["trade_date"], errors="coerce").dropna())
    data = market.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    return data[data["trade_date"].isin(dates)].reset_index(drop=True)


def _append_paper_book(path: str, as_of: str, weights: pd.DataFrame, summary: dict[str, object]) -> Path:
    book_path = Path(path)
    book_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "as_of_date": as_of,
        "target_weight_rows": int(len(weights)),
        "net_return_after_estimated_costs": summary.get("net_return_after_estimated_costs"),
        "max_drawdown": summary.get("max_drawdown"),
    }
    existing = pd.DataFrame()
    if book_path.exists():
        try:
            existing = pd.read_parquet(book_path)
        except Exception:
            csv = book_path.with_suffix(".csv")
            if csv.exists():
                existing = pd.read_csv(csv)
    output = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    return write_frame(output, book_path)


__all__ = ["DailyPaperLoopConfig", "DailyPaperLoopResult", "run_once"]
