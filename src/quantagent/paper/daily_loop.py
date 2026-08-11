"""Daily V7 paper loop: evidence refresh, alpha, and pending target signal.

This loop runs at/after the signal session close.  The target generated from T
information is therefore *pending* until the next observed market session under
``signal_t_close_next_session_close_v1``.  It must not be handed a future bar,
executed same-session, entered into the paper performance book, or reported as
completed paper PnL.

A later execution stage will consume the durable pending artifact against a
continuous recovered PaperBroker account.  Before a non-empty target is frozen,
this module also replays that same canonical account and reconciles the desired
target against the shares/cash that were actually filled.  Desired targets are
never treated as holdings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.cli._utils import read_frame, write_frame
from quantagent.config.paths import quant_paths
from quantagent.data.ingestion.daily_evidence_job import DailyEvidenceJob, DailyEvidenceJobConfig
from quantagent.paper.account_target_state import (
    recover_paper_account_target_state,
    reconcile_target_to_canonical_account,
)
from quantagent.paper.pending_signal import PendingPaperSignalStore
from quantagent.paper.runtime_paths import paper_runtime_paths
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
    paper_book_path: str = field(default_factory=lambda: str(paper_runtime_paths().paper_book))
    pending_signal_dir: str = field(default_factory=lambda: str(paper_runtime_paths().pending_signals))
    canonical_ledger_path: str = field(default_factory=lambda: str(paper_runtime_paths().canonical_ledger))
    portfolio_id: str = "v7-paper"
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
    pending_signal_path: str = ""
    execution_timing_semantics: str = EXECUTION_TIMING_SEMANTICS
    executed_fill_count: int = 0
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_once(config: DailyPaperLoopConfig) -> DailyPaperLoopResult:
    """Generate one T-close target and persist it as pending execution evidence.

    This function intentionally performs **zero economic fills**. That is the
    correct result when the next session has not yet been observed. The
    historical same-call strict simulation was not a shadow execution: it either
    right-censored the signal or required future data. Pending intent and
    executed paper evidence are now distinct states.

    For a non-empty desired target, the final frozen target is additionally
    constrained against the canonical account as it actually exists at T close.
    This makes partial fills, rejections, cash fees and dropped holdings visible
    to the next turnover decision.
    """

    as_of = _normalise_date(config.as_of_date)
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
    market_path = Path(config.market_panel_path)
    if not market_path.exists():
        raise FileNotFoundError(f"market panel not found: {market_path}")

    feature_dataset = read_frame(feature_path)
    market_panel = read_frame(market_path)
    features_asof = _asof_slice(feature_dataset, as_of)
    if features_asof.empty:
        raise ValueError(f"no feature rows available at or before {as_of}")

    prediction_result = predict_v7_alpha(
        config.model_dir,
        features_asof,
        primary_horizon=config.primary_horizon,
    )
    blend_result = blend_multi_horizon_predictions(
        prediction_result.predictions,
        config=MultiHorizonBlendConfig(primary_horizon=config.primary_horizon),
    )
    predictions = (
        blend_result.blended
        if not blend_result.blended.empty
        else prediction_result.predictions
    )
    day_dir = Path(config.output_root) / as_of
    predictions_path = write_frame(predictions, day_dir / "predictions.parquet")
    sector = read_frame(Path(config.sector_map_path)) if config.sector_map_path else None
    desired_weights = build_v7_target_weights(
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

    account_state = None
    if desired_weights.target_weights is not None and not desired_weights.target_weights.empty:
        account_state = recover_paper_account_target_state(
            canonical_ledger_path=config.canonical_ledger_path,
            market_panel=market_panel,
            as_of_date=as_of,
            portfolio_id=config.portfolio_id,
            initial_cash=config.initial_cash,
        )
        weights = reconcile_target_to_canonical_account(
            desired_weights,
            account_state=account_state,
            max_turnover=config.max_turnover,
        )
    else:
        weights = desired_weights

    weights_path = write_v7_target_weights(
        weights,
        day_dir / "target_weights.parquet",
    )

    if weights.target_weights is None or weights.target_weights.empty:
        warnings = tuple([*evidence.warnings, "paper_no_target_generated"])
        summary = {
            "config": asdict(config),
            "status": "no_target_generated",
            "evidence_rows": int(len(evidence.frame)),
            "evidence_warnings": list(evidence.warnings),
            "blend_diagnostics": blend_result.diagnostics,
            "target_weight_diagnostics": weights.diagnostics,
            "execution": {
                "status": "no_target_generated",
                "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
                "signal_date": as_of,
                "pending_signal_path": "",
                "executed_fill_count": 0,
                "paper_report_written": False,
                "paper_book_appended": False,
                "reason": (
                    "portfolio construction produced no target; absence of a target "
                    "is not reinterpreted as an all-zero liquidation instruction"
                ),
            },
            "paper_report": None,
            "paper_book_path": str(config.paper_book_path),
        }
        _write_daily_summary(day_dir, summary)
        return DailyPaperLoopResult(
            status="no_target_generated",
            as_of_date=as_of,
            evidence_rows=int(len(evidence.frame)),
            predictions_path=str(predictions_path),
            target_weights_path=str(weights_path),
            paper_report_dir=str(day_dir),
            paper_book_path=str(config.paper_book_path),
            pending_signal_path="",
            execution_timing_semantics=EXECUTION_TIMING_SEMANTICS,
            executed_fill_count=0,
            warnings=warnings,
        )

    if account_state is None:  # defensive invariant; non-empty targets require account proof
        raise RuntimeError("non-empty paper target lacks canonical account state")

    predictions_file = Path(predictions_path)
    target_weights_file = Path(weights_path)
    account_evidence = account_state.evidence()
    pending, pending_path = PendingPaperSignalStore(config.pending_signal_dir).record(
        signal_date=as_of,
        target_weights=weights.target_weights,
        source_lineage={
            "model_dir": config.model_dir,
            "feature_dataset_path": str(feature_path),
            "market_panel_path": str(market_path),
            "predictions_path": str(predictions_file),
            "predictions_file_sha256": _file_sha256(predictions_file),
            "target_weights_path": str(target_weights_file),
            "target_weights_file_sha256": _file_sha256(target_weights_file),
            "primary_horizon": str(config.primary_horizon),
            "canonical_account_state_sha256": str(account_evidence["account_state_sha256"]),
            "canonical_ledger_head_hash": str(account_evidence["canonical_head_hash"]),
            "canonical_ledger_records": str(account_evidence["canonical_records"]),
            "canonical_account_nav": str(account_evidence["nav"]),
        },
    )

    warnings = tuple(
        [*evidence.warnings, "paper_signal_pending_next_observed_session"]
    )
    summary = {
        "config": asdict(config),
        "status": "signal_recorded_pending_execution",
        "evidence_rows": int(len(evidence.frame)),
        "evidence_warnings": list(evidence.warnings),
        "blend_diagnostics": blend_result.diagnostics,
        "target_weight_diagnostics": weights.diagnostics,
        "account_state": account_evidence,
        "execution": {
            "status": "pending_next_observed_session",
            "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
            "signal_date": as_of,
            "pending_signal_path": str(pending_path),
            "pending_payload_sha256": pending.payload_sha256,
            "target_weights_sha256": pending.target_weights_sha256,
            "predictions_file_sha256": pending.source_lineage["predictions_file_sha256"],
            "target_weights_file_sha256": pending.source_lineage["target_weights_file_sha256"],
            "canonical_account_state_sha256": pending.source_lineage["canonical_account_state_sha256"],
            "canonical_ledger_head_hash": pending.source_lineage["canonical_ledger_head_hash"],
            "executed_fill_count": 0,
            "paper_report_written": False,
            "paper_book_appended": False,
            "reason": (
                "T-close target is an intent only; no next observed market session "
                "has been executed by this signal-generation call"
            ),
        },
        "paper_report": None,
        "paper_book_path": str(config.paper_book_path),
    }
    _write_daily_summary(day_dir, summary)
    return DailyPaperLoopResult(
        status="signal_recorded_pending_execution",
        as_of_date=as_of,
        evidence_rows=int(len(evidence.frame)),
        predictions_path=str(predictions_file),
        target_weights_path=str(target_weights_file),
        paper_report_dir=str(day_dir),
        paper_book_path=str(config.paper_book_path),
        pending_signal_path=str(pending_path),
        execution_timing_semantics=EXECUTION_TIMING_SEMANTICS,
        executed_fill_count=0,
        warnings=warnings,
    )


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"cannot bind generated artifact bytes: {path}")
    return sha256(path.read_bytes()).hexdigest()


def _write_daily_summary(day_dir: Path, summary: dict[str, object]) -> None:
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / "daily_loop_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
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


__all__ = ["DailyPaperLoopConfig", "DailyPaperLoopResult", "run_once"]
