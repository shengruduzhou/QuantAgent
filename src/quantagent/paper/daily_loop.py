"""Daily V7 paper loop: evidence refresh, alpha, and pending target signal.

This loop runs at/after the signal session close. The target generated from T
information is therefore *pending* until the next observed market session under
``signal_t_close_next_session_close_v1``. It must not be handed a future bar,
executed same-session, entered into the paper performance book, or reported as
completed paper PnL.

A later execution stage will consume the durable pending artifact against a
continuous recovered PaperBroker account. Target construction itself starts
from that same canonical executed account: recovered marked NAV drives
liquidity capacity and recovered actual weights seed first-date timing,
holding-period and turnover state. Before any daily decision is persisted, the
canonical account is recovered again while holding the cross-process account
lock. Any ledger/state drift makes the stale decision fail closed, every older
pending signal and account-wide execution history must already have a
canonically verified terminal/reconciliation outcome, and same-date evidence
files are written only inside that account-wide critical section. Desired
targets are never treated as holdings. The account genesis itself is immutable:
every run must match the persisted portfolio_id/initial_cash identity before any
target is produced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd

from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.cli._utils import read_frame, write_frame
from quantagent.config.paths import quant_paths
from quantagent.data.ingestion.daily_evidence_job import DailyEvidenceJob, DailyEvidenceJobConfig
from quantagent.domain.ledger import CanonicalLedger
from quantagent.paper.account_identity import (
    account_identity_path_for_canonical,
    ensure_paper_account_identity,
)
from quantagent.paper.account_lock import paper_account_lock
from quantagent.paper.account_target_state import (
    PaperAccountStateRefused,
    recover_paper_account_target_state,
    reconcile_target_to_canonical_account,
)
from quantagent.paper.canonical_receipt import (
    CanonicalPrefixReceiptError,
    build_canonical_prefix_index,
    verify_canonical_prefix_receipt,
)
from quantagent.paper.execution_journal import (
    DAILY_DECISION_STATUS,
    TERMINAL_OUTCOMES,
    PendingExecutionJournal,
)
from quantagent.paper.legacy_terminal_binding import (
    LegacyTerminalBindingError,
    verify_legacy_terminal_binding,
)
from quantagent.paper.pending_signal import (
    PENDING_COMMIT_PROTOCOL,
    PendingPaperSignalStore,
)
from quantagent.paper.runtime_paths import paper_runtime_paths
from quantagent.portfolio.multi_horizon_blender import MultiHorizonBlendConfig, blend_multi_horizon_predictions
from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights, write_v7_target_weights
from quantagent.training.v7_predictor import predict_v7_alpha


DAILY_SUMMARY_COMMIT_PROTOCOL = "daily_summary_bound_daily_decision_v1"


_RESOLVED_PRIOR_TERMINAL_STATUSES = frozenset(
    {
        "execution_observed",
        "execution_blocked",
        "missed_execution_session",
    }
)


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
    account_identity_path: str | None = None
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
    # Appended after the historical public positional fields so introducing the
    # journal gate cannot silently reinterpret existing positional callers.
    execution_journal_path: str | None = None


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

    The immutable paper account identity is verified before evidence refresh or
    prediction. The canonical account is recovered and marked before target
    construction. Immediately before either a target or a no-target decision is
    persisted, the account is recovered again under the same cross-process lock
    used by the continuous execution worker. Older pending signals and orphaned
    journal attempts must already be resolved with valid canonical evidence; a
    changed state/head/count invalidates the computed decision instead of
    silently applying it to different holdings.

    Prediction/target files that become pending-signal lineage are also written
    inside that lock. Once one same-date pending signal is frozen, a second
    writer is refused before it can overwrite those hash-bound evidence files.
    """

    as_of = _normalise_date(config.as_of_date)
    resolved_identity_path = (
        Path(config.account_identity_path)
        if config.account_identity_path is not None
        else account_identity_path_for_canonical(config.canonical_ledger_path)
    )
    account_identity = ensure_paper_account_identity(
        canonical_ledger_path=config.canonical_ledger_path,
        portfolio_id=config.portfolio_id,
        initial_cash=config.initial_cash,
        identity_path=config.account_identity_path,
    )
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
    account_state = recover_paper_account_target_state(
        canonical_ledger_path=config.canonical_ledger_path,
        market_panel=market_panel,
        as_of_date=as_of,
        portfolio_id=account_identity.portfolio_id,
        initial_cash=account_identity.initial_cash,
    )
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
            capital_yuan=float(account_state.nav),
            shrink_on_small_universe=True,
            min_selection_pressure=1.0,
        ),
        initial_weights=account_state.current_weights,
    )

    if desired_weights.target_weights is not None and not desired_weights.target_weights.empty:
        weights = reconcile_target_to_canonical_account(
            desired_weights,
            account_state=account_state,
            max_turnover=config.max_turnover,
        )
    else:
        weights = desired_weights

    identity_evidence = {
        "schema_version": account_identity.schema_version,
        "account_instance_id": account_identity.account_instance_id,
        "portfolio_id": account_identity.portfolio_id,
        "initial_cash_cny": account_identity.initial_cash_cny,
        "payload_sha256": account_identity.payload_sha256,
        "identity_path": str(resolved_identity_path),
    }
    account_evidence = account_state.evidence()

    if weights.target_weights is None or weights.target_weights.empty:
        with paper_account_lock(config.canonical_ledger_path):
            _assert_current_signal_not_frozen(
                config,
                as_of,
                paper_account_identity_sha256=account_identity.payload_sha256,
            )
            _assert_prior_pending_signals_resolved(
                config,
                as_of,
                paper_account_identity_sha256=account_identity.payload_sha256,
            )
            fresh_account_state = recover_paper_account_target_state(
                canonical_ledger_path=config.canonical_ledger_path,
                market_panel=market_panel,
                as_of_date=as_of,
                portfolio_id=account_identity.portfolio_id,
                initial_cash=account_identity.initial_cash,
            )
            _assert_account_snapshot_unchanged(account_state, fresh_account_state)
            account_state = fresh_account_state
            account_evidence = account_state.evidence()
            _assert_exact_signal_date(
                predictions,
                as_of,
                evidence_name="no-target predictions",
            )
            # Stage every fallible artifact before the irreversible no-target
            # marker. A failed writer leaves the date safely rerunnable.
            predictions_path = write_frame(predictions, day_dir / "predictions.parquet")
            weights_path = write_v7_target_weights(
                weights,
                day_dir / "target_weights.parquet",
            )
            warnings = tuple([*evidence.warnings, "paper_no_target_generated"])
            summary = {
                "daily_decision_commit_protocol": DAILY_SUMMARY_COMMIT_PROTOCOL,
                "paper_account_owner": _paper_account_owner(
                    config, account_identity.payload_sha256
                ),
                "config": asdict(config),
                "status": "no_target_generated",
                "evidence_rows": int(len(evidence.frame)),
                "evidence_warnings": list(evidence.warnings),
                "account_identity": identity_evidence,
                "account_state": account_evidence,
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
                    "paper_account_identity_sha256": account_identity.payload_sha256,
                    "canonical_account_state_sha256": account_state.account_state_sha256,
                    "reason": (
                        "portfolio construction produced no target; absence of a target "
                        "is not reinterpreted as an all-zero liquidation instruction"
                    ),
                },
                "paper_report": None,
                "paper_book_path": str(config.paper_book_path),
            }
            summary_path = _write_daily_summary(day_dir, summary)
            summary_sha = _file_sha256(summary_path)
            try:
                _freeze_daily_decision(
                    config,
                    as_of,
                    decision_kind="no_target",
                    paper_account_identity_sha256=account_identity.payload_sha256,
                    account_evidence=account_evidence,
                    daily_summary_path=summary_path,
                    daily_summary_sha256=summary_sha,
                )
            except Exception:
                decision = PendingExecutionJournal(
                    _execution_journal_path(config)
                ).daily_decision(as_of)
                if decision is None:
                    _discard_staged_daily_summary(
                        summary_path,
                        expected_sha256=summary_sha,
                        expected_owner=_paper_account_owner(
                            config, account_identity.payload_sha256
                        ),
                    )
                raise
        summary = {
            "config": asdict(config),
            "status": "no_target_generated",
            "evidence_rows": int(len(evidence.frame)),
            "evidence_warnings": list(evidence.warnings),
            "account_identity": identity_evidence,
            "account_state": account_evidence,
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
                "paper_account_identity_sha256": account_identity.payload_sha256,
                "canonical_account_state_sha256": account_state.account_state_sha256,
                "reason": (
                    "portfolio construction produced no target; absence of a target "
                    "is not reinterpreted as an all-zero liquidation instruction"
                ),
            },
            "paper_report": None,
            "paper_book_path": str(config.paper_book_path),
        }
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

    with paper_account_lock(config.canonical_ledger_path):
        _assert_current_signal_not_frozen(
            config,
            as_of,
            paper_account_identity_sha256=account_identity.payload_sha256,
        )
        _assert_prior_pending_signals_resolved(
            config,
            as_of,
            paper_account_identity_sha256=account_identity.payload_sha256,
        )
        fresh_account_state = recover_paper_account_target_state(
            canonical_ledger_path=config.canonical_ledger_path,
            market_panel=market_panel,
            as_of_date=as_of,
            portfolio_id=account_identity.portfolio_id,
            initial_cash=account_identity.initial_cash,
        )
        _assert_account_snapshot_unchanged(account_state, fresh_account_state)
        account_state = fresh_account_state
        account_evidence = account_state.evidence()
        # Persist and read-verify every fallible artifact, including pending
        # target date/weight validation, before the append-only daily freeze.
        predictions_path = write_frame(predictions, day_dir / "predictions.parquet")
        weights_path = write_v7_target_weights(
            weights,
            day_dir / "target_weights.parquet",
        )
        predictions_file = Path(predictions_path)
        target_weights_file = Path(weights_path)
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
                "paper_account_identity_sha256": account_identity.payload_sha256,
                "canonical_ledger_path": str(
                    Path(config.canonical_ledger_path).resolve(strict=False)
                ),
                "execution_journal_path": str(
                    Path(_execution_journal_path(config)).resolve(strict=False)
                ),
                "canonical_account_state_sha256": str(account_evidence["account_state_sha256"]),
                "canonical_ledger_head_hash": str(account_evidence["canonical_head_hash"]),
                "canonical_ledger_records": str(account_evidence["canonical_records"]),
                "canonical_account_nav": str(account_evidence["nav"]),
                "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,
            },
        )
        warnings = tuple(
            [*evidence.warnings, "paper_signal_pending_next_observed_session"]
        )
        summary = {
            "daily_decision_commit_protocol": DAILY_SUMMARY_COMMIT_PROTOCOL,
            "paper_account_owner": _paper_account_owner(
                config, account_identity.payload_sha256
            ),
            "config": asdict(config),
            "status": "signal_recorded_pending_execution",
            "evidence_rows": int(len(evidence.frame)),
            "evidence_warnings": list(evidence.warnings),
            "account_identity": identity_evidence,
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
                "paper_account_identity_sha256": pending.source_lineage["paper_account_identity_sha256"],
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
        summary_path = _write_daily_summary(day_dir, summary)
        summary_sha = _file_sha256(summary_path)
        try:
            _freeze_daily_decision(
                config,
                as_of,
                decision_kind="target",
                paper_account_identity_sha256=account_identity.payload_sha256,
                account_evidence=account_evidence,
                pending_payload_sha256=pending.payload_sha256,
                target_weights_sha256=pending.target_weights_sha256,
                daily_summary_path=summary_path,
                daily_summary_sha256=summary_sha,
            )
        except Exception:
            # If append definitively did not commit, remove the current-protocol
            # staging artifact. If a marker is visible, the transaction committed
            # even if the caller did not receive success; preserve it.
            decision = PendingExecutionJournal(
                _execution_journal_path(config)
            ).daily_decision(as_of)
            if decision is None:
                PendingPaperSignalStore(config.pending_signal_dir).discard_staged(
                    as_of, expected_payload_sha256=pending.payload_sha256
                )
                _discard_staged_daily_summary(
                    summary_path,
                    expected_sha256=summary_sha,
                    expected_owner=_paper_account_owner(
                        config, account_identity.payload_sha256
                    ),
                )
            raise

    warnings = tuple(
        [*evidence.warnings, "paper_signal_pending_next_observed_session"]
    )
    summary = {
        "config": asdict(config),
        "status": "signal_recorded_pending_execution",
        "evidence_rows": int(len(evidence.frame)),
        "evidence_warnings": list(evidence.warnings),
        "account_identity": identity_evidence,
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
            "paper_account_identity_sha256": pending.source_lineage["paper_account_identity_sha256"],
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


def _assert_account_snapshot_unchanged(expected, observed) -> None:
    if (
        observed.account_state_sha256 != expected.account_state_sha256
        or observed.canonical_records != expected.canonical_records
        or observed.canonical_head_hash != expected.canonical_head_hash
    ):
        raise PaperAccountStateRefused(
            "canonical paper account changed during target construction; stale "
            "daily decision was discarded before persistence. Rerun target "
            "generation from the freshly executed account state"
        )


def _assert_exact_signal_date(
    frame: pd.DataFrame,
    as_of: str,
    *,
    evidence_name: str,
) -> None:
    if "trade_date" not in frame.columns or frame.empty:
        raise PaperAccountStateRefused(
            f"{evidence_name} lacks a non-empty trade_date column for {as_of}"
        )
    parsed = pd.to_datetime(frame["trade_date"], errors="coerce")
    if parsed.isna().any():
        raise PaperAccountStateRefused(
            f"{evidence_name} contains invalid trade_date values"
        )
    dates = {pd.Timestamp(value).date().isoformat() for value in parsed}
    if dates != {str(as_of)}:
        raise PaperAccountStateRefused(
            f"{evidence_name} must belong exactly to signal_date={as_of}; "
            f"observed_dates={sorted(dates)}"
        )


def _execution_journal_path(config: DailyPaperLoopConfig) -> str:
    if config.execution_journal_path is not None:
        return str(config.execution_journal_path)

    canonical = Path(config.canonical_ledger_path).resolve(strict=False)
    runtime = paper_runtime_paths()
    if canonical == runtime.canonical_ledger.resolve(strict=False):
        return str(runtime.execution_journal)

    # A custom canonical account must never share the default account journal
    # merely because both ledgers live in the same runtime directory.
    suffix = canonical.suffix or ".jsonl"
    stem = canonical.stem if canonical.suffix else canonical.name
    return str(canonical.with_name(f"{stem}.execution_journal{suffix}"))


def _assert_current_signal_not_frozen(
    config: DailyPaperLoopConfig,
    as_of: str,
    *,
    paper_account_identity_sha256: str,
) -> None:
    """Fail closed if any durable evidence already owns this signal date."""

    store = PendingPaperSignalStore(config.pending_signal_dir)
    journal = PendingExecutionJournal(_execution_journal_path(config))
    if not journal.verify():
        raise PaperAccountStateRefused(
            "pending execution journal verification failed before same-date freeze check"
        )
    if journal.daily_decision(as_of) is not None:
        raise PaperAccountStateRefused(
            "daily paper decision is already durably frozen for the current signal date"
        )
    summary_path = Path(config.output_root) / as_of / "daily_loop_summary.json"
    if summary_path.exists():
        try:
            summary_payload = _read_strict_json_object(summary_path)
        except Exception as exc:
            raise PaperAccountStateRefused(
                "uncommitted daily summary is unreadable/ambiguous; operator "
                "reconciliation is required before same-date reuse"
            ) from exc
        if (
            isinstance(summary_payload, dict)
            and summary_payload.get("daily_decision_commit_protocol")
            == DAILY_SUMMARY_COMMIT_PROTOCOL
            and summary_payload.get("paper_account_owner")
            == _paper_account_owner(config, paper_account_identity_sha256)
        ):
            _discard_staged_daily_summary(
                summary_path,
                expected_sha256=_file_sha256(summary_path),
                expected_owner=_paper_account_owner(
                    config, paper_account_identity_sha256
                ),
            )
        else:
            raise PaperAccountStateRefused(
                "legacy/ambiguous daily summary exists without a committed marker; "
                "refusing same-date overwrite"
            )
    existing = store.read(as_of)
    if existing is not None:
        if (
            existing.source_lineage.get("daily_decision_commit_protocol")
            == PENDING_COMMIT_PROTOCOL
            and existing.source_lineage.get("paper_account_identity_sha256")
            == paper_account_identity_sha256
            and existing.source_lineage.get("canonical_ledger_path")
            == str(Path(config.canonical_ledger_path).resolve(strict=False))
            and existing.source_lineage.get("execution_journal_path")
            == str(Path(_execution_journal_path(config)).resolve(strict=False))
        ):
            # Current-protocol pending without a freeze is staging left by a
            # crash before commit. The execution consumer rejects it, so under
            # the same account lock it is safe to remove and recompute.
            store.discard_staged(
                as_of, expected_payload_sha256=existing.payload_sha256
            )
        else:
            raise PaperAccountStateRefused(
                "legacy/ambiguous pending paper signal exists without a committed "
                "daily-decision marker; operator reconciliation is required"
            )
    same_date_economic = [
        record
        for record in journal.records()
        if record.signal_date == as_of
        and (record.status == "execution_started" or record.status in TERMINAL_OUTCOMES)
    ]
    if same_date_economic:
        raise PaperAccountStateRefused(
            "legacy same-date execution evidence already exists; refusing to reuse "
            "the signal date without rewriting append-only history"
        )


def _freeze_daily_decision(
    config: DailyPaperLoopConfig,
    as_of: str,
    *,
    decision_kind: str,
    paper_account_identity_sha256: str,
    account_evidence: dict[str, object],
    pending_payload_sha256: str | None = None,
    target_weights_sha256: str | None = None,
    daily_summary_path: Path | None = None,
    daily_summary_sha256: str | None = None,
) -> None:
    """Append the irreversible commit marker for one validated daily decision."""

    if daily_summary_path is None or not Path(daily_summary_path).is_file():
        raise PaperAccountStateRefused(
            "daily decision requires a durable pre-freeze summary artifact"
        )
    if len(str(daily_summary_sha256 or "")) != 64:
        raise PaperAccountStateRefused(
            "daily decision requires exact summary digest binding"
        )
    if _file_sha256(Path(daily_summary_path)) != str(daily_summary_sha256):
        raise PaperAccountStateRefused(
            "daily summary changed before decision freeze"
        )

    if decision_kind == "target":
        if len(str(pending_payload_sha256 or "")) != 64:
            raise PaperAccountStateRefused(
                "target daily decision requires exact pending payload binding"
            )
        if len(str(target_weights_sha256 or "")) != 64:
            raise PaperAccountStateRefused(
                "target daily decision requires exact target-weight binding"
            )
        marker_identity = str(pending_payload_sha256)
    else:
        material = (
            "quantagent.paper.daily_decision.v1|"
            f"{as_of}|{paper_account_identity_sha256}"
        ).encode("utf-8")
        marker_identity = sha256(material).hexdigest()

    journal = PendingExecutionJournal(_execution_journal_path(config))
    if not journal.verify():
        raise PaperAccountStateRefused(
            "pending execution journal verification failed before daily decision freeze"
        )
    journal.append(
        pending_payload_sha256=marker_identity,
        signal_date=as_of,
        execution_date=as_of,
        status=DAILY_DECISION_STATUS,
        details={
            "decision_kind": str(decision_kind),
            "paper_account_identity_sha256": str(paper_account_identity_sha256),
            "canonical_account_state_sha256": str(account_evidence["account_state_sha256"]),
            "canonical_records": int(account_evidence["canonical_records"]),
            "canonical_head": str(account_evidence["canonical_head_hash"]),
            "assurance": "canonical_account_daily_decision_freeze_v1",
            "commit_protocol": (
                PENDING_COMMIT_PROTOCOL if decision_kind == "target" else "no_target_v1"
            ),
            "target_weights_sha256": (
                str(target_weights_sha256) if decision_kind == "target" else ""
            ),
            "daily_summary_path": str(Path(daily_summary_path).resolve(strict=False)),
            "daily_summary_sha256": str(daily_summary_sha256),
            "daily_summary_commit_protocol": DAILY_SUMMARY_COMMIT_PROTOCOL,
        },
    )


def _valid_indeterminate_reconciliation(
    *,
    terminal,
    reconciliation,
    prefix_index,
    paper_account_identity_sha256: str,
) -> bool:
    if reconciliation is None:
        return False
    details = dict(reconciliation.details or {})
    if str(details.get("indeterminate_record_sha256") or "") != terminal.record_sha256:
        return False
    if str(details.get("paper_account_identity_sha256") or "") != str(
        paper_account_identity_sha256
    ):
        return False
    try:
        record_count = int(details["canonical_records"])
        head = str(details["canonical_head"])
        return prefix_index.head_at(record_count) == head
    except (CanonicalPrefixReceiptError, KeyError, TypeError, ValueError):
        return False


def _assert_terminal_bound_to_account(
    *,
    terminal,
    legacy_binding,
    prefix_index,
    paper_account_identity_sha256: str,
    expected_target_weights_sha256: str | None = None,
    signal_date: str | None = None,
) -> None:
    receipt = dict(terminal.details or {}).get("canonical_prefix_receipt")
    try:
        verification = verify_canonical_prefix_receipt(
            receipt,
            prefix_index=prefix_index,
            expected_target_weights_sha256=expected_target_weights_sha256,
            expected_paper_account_identity_sha256=(
                paper_account_identity_sha256 if receipt is not None else None
            ),
        )
    except CanonicalPrefixReceiptError as exc:
        location = f"signal_date={signal_date}: " if signal_date else ""
        raise PaperAccountStateRefused(
            "paper execution terminal no longer matches the canonical account: "
            f"{location}{exc}"
        ) from exc
    if verification.bound:
        return
    try:
        verify_legacy_terminal_binding(
            terminal,
            legacy_binding,
            prefix_index=prefix_index,
            expected_paper_account_identity_sha256=paper_account_identity_sha256,
            expected_target_weights_sha256=expected_target_weights_sha256,
        )
    except LegacyTerminalBindingError as exc:
        location = f"signal_date={signal_date}; " if signal_date else ""
        raise PaperAccountStateRefused(
            "paper execution terminal lacks valid execution-time receipt or "
            f"operator-reconciled legacy binding: {location}{exc}"
        ) from exc


def _assert_execution_journal_resolved(
    journal: PendingExecutionJournal,
    *,
    prefix_index,
    paper_account_identity_sha256: str,
) -> None:
    """Freeze target construction on unresolved account-wide journal economics.

    Pending JSON files are not authoritative lifecycle state. A missing/deleted
    artifact must not erase an ``execution_started`` or uncertain terminal from
    the append-only journal, so this scan deliberately runs before looking at
    the pending directory.
    """

    by_payload: dict[str, list[object]] = {}
    for record in journal.records():
        by_payload.setdefault(record.pending_payload_sha256, []).append(record)

    # Resolve account-wide uncertainty before lower-assurance lineage
    # migration checks. Journal insertion order must not let an older unbound
    # legacy terminal mask a later indeterminate account state.
    terminals_by_payload: dict[str, object] = {}
    for payload, history in by_payload.items():
        starts = [row for row in history if row.status == "execution_started"]
        terminals = [row for row in history if row.status in TERMINAL_OUTCOMES]
        if starts and not terminals:
            raise PaperAccountStateRefused(
                "paper account has an unresolved execution_started record for "
                f"{payload}; pending artifact presence is irrelevant. Explicit "
                "account reconciliation is required before a new target can freeze"
            )
        if len(terminals) > 1:
            raise PaperAccountStateRefused(
                f"paper account has multiple terminal outcomes for {payload}"
            )
        if not terminals:
            continue
        terminal = terminals[0]
        terminals_by_payload[payload] = terminal
        if terminal.status == "execution_indeterminate":
            reconciliation = journal.reconciliation(payload)
            if not _valid_indeterminate_reconciliation(
                terminal=terminal,
                reconciliation=reconciliation,
                prefix_index=prefix_index,
                paper_account_identity_sha256=paper_account_identity_sha256,
            ):
                raise PaperAccountStateRefused(
                    "paper account has an unreconciled execution_indeterminate outcome; "
                    "explicit canonical/operational reconciliation is required before "
                    "a new target can freeze"
                )

    for payload, terminal in terminals_by_payload.items():
        _assert_terminal_bound_to_account(
            terminal=terminal,
            legacy_binding=journal.legacy_binding(payload),
            prefix_index=prefix_index,
            paper_account_identity_sha256=paper_account_identity_sha256,
        )


def _assert_prior_pending_signals_resolved(
    config: DailyPaperLoopConfig,
    as_of: str,
    *,
    paper_account_identity_sha256: str,
) -> None:
    """Require previous economic attempts to resolve before a new daily decision.

    The cross-process account lock must already be held by the caller. The
    append-only execution journal is scanned first, independently of pending
    artifact presence. Then surviving prior pending signals are cross-bound to
    their exact target digest, terminal receipt and reconciliation evidence.
    """

    journal = PendingExecutionJournal(_execution_journal_path(config))
    if not journal.verify():
        raise PaperAccountStateRefused(
            "pending execution journal verification failed before target freeze"
        )
    cutoff = _strict_date(as_of, evidence_name="paper decision as_of")
    later_records = []
    for record in journal.records():
        signal_date = _strict_date(
            record.signal_date, evidence_name="journal signal_date"
        )
        execution_date = _strict_date(
            record.execution_date, evidence_name="journal execution_date"
        )
        if signal_date > cutoff or execution_date > cutoff:
            later_records.append(record)
    if later_records:
        first = min(
            later_records,
            key=lambda row: max(
                _strict_date(row.signal_date, evidence_name="journal signal_date"),
                _strict_date(
                    row.execution_date, evidence_name="journal execution_date"
                ),
            ),
        )
        raise PaperAccountStateRefused(
            "paper decision chronology regression refused: later durable journal "
            "evidence already exists for "
            f"signal_date={first.signal_date}, execution_date={first.execution_date}"
        )

    canonical = CanonicalLedger(config.canonical_ledger_path)
    canonical_verification = canonical.verify()
    if (
        not bool(canonical_verification.get("valid"))
        or bool(canonical_verification.get("tornTail"))
    ):
        raise PaperAccountStateRefused(
            "canonical ledger chronology cannot be certified because the chain/tail "
            "is not fully verifiable"
        )
    for record in canonical.read():
        if record.trade_date is None:
            continue
        trade_date = _strict_date(
            record.trade_date, evidence_name="canonical ledger trade_date"
        )
        if trade_date > cutoff:
            raise PaperAccountStateRefused(
                "paper decision chronology regression refused: canonical economic "
                f"history already reached trade_date={trade_date.isoformat()}"
            )

    prefix_index = build_canonical_prefix_index(config.canonical_ledger_path)
    _assert_execution_journal_resolved(
        journal,
        prefix_index=prefix_index,
        paper_account_identity_sha256=paper_account_identity_sha256,
    )

    root = Path(config.pending_signal_dir)
    if not root.exists():
        return
    store = PendingPaperSignalStore(root)
    for path in sorted(root.glob("*.json")):
        signal_date = _strict_date(
            path.stem, evidence_name=f"pending-signal filename {path.name}"
        )
        if signal_date > cutoff:
            raise PaperAccountStateRefused(
                "paper decision chronology regression refused: later pending signal "
                f"already exists for signal_date={signal_date.isoformat()}"
            )
        if signal_date == cutoff:
            continue
        signal = store.read(signal_date.isoformat())
        if signal is None:
            continue
        terminal = journal.terminal(signal.payload_sha256)
        if terminal is None:
            raise PaperAccountStateRefused(
                "prior pending paper signal is unresolved before current target "
                f"freeze: signal_date={signal.signal_date}, status=none; "
                "run/reconcile the execution stage first"
            )
        _assert_terminal_bound_to_account(
            terminal=terminal,
            legacy_binding=journal.legacy_binding(signal.payload_sha256),
            prefix_index=prefix_index,
            paper_account_identity_sha256=paper_account_identity_sha256,
            expected_target_weights_sha256=signal.target_weights_sha256,
            signal_date=signal.signal_date,
        )

        if terminal.status in _RESOLVED_PRIOR_TERMINAL_STATUSES:
            continue
        if terminal.status == "execution_indeterminate":
            reconciliation = journal.reconciliation(signal.payload_sha256)
            if _valid_indeterminate_reconciliation(
                terminal=terminal,
                reconciliation=reconciliation,
                prefix_index=prefix_index,
                paper_account_identity_sha256=paper_account_identity_sha256,
            ):
                continue
        raise PaperAccountStateRefused(
            "prior pending paper signal is unresolved before current target "
            f"freeze: signal_date={signal.signal_date}, status={terminal.status}; "
            "run/reconcile the execution stage first"
        )


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"cannot bind generated artifact bytes: {path}")
    return sha256(path.read_bytes()).hexdigest()


def _write_daily_summary(day_dir: Path, summary: dict[str, object]) -> Path:
    day_dir.mkdir(parents=True, exist_ok=True)
    target = day_dir / "daily_loop_summary.json"
    tmp = day_dir / f".{target.name}.{uuid4().hex}.tmp"
    payload = (
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        if os.name != "nt":
            directory_fd = os.open(day_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            if tmp.exists() or tmp.is_symlink():
                tmp.unlink()
        except OSError:
            pass
    return target


def _discard_staged_daily_summary(
    path: Path,
    *,
    expected_sha256: str,
    expected_owner: dict[str, str],
) -> None:
    if not path.exists():
        return
    if _file_sha256(path) != str(expected_sha256):
        raise PaperAccountStateRefused(
            "refusing to discard staged daily summary with mismatched digest"
        )
    try:
        payload = _read_strict_json_object(path)
    except Exception as exc:
        raise PaperAccountStateRefused(
            "refusing to discard unreadable staged daily summary"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("daily_decision_commit_protocol")
        != DAILY_SUMMARY_COMMIT_PROTOCOL
        or payload.get("paper_account_owner") != expected_owner
    ):
        raise PaperAccountStateRefused(
            "refusing to discard legacy/ambiguous daily summary staging"
        )
    path.unlink()
    if os.name != "nt":
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _normalise_date(value: str) -> str:
    if value.lower() == "today":
        return date.today().isoformat()
    return _strict_date(value, evidence_name="as_of_date").isoformat()


def _strict_json_object(pairs):
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _read_strict_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_strict_json_object
    )
    if not isinstance(payload, dict):
        raise ValueError("JSON evidence root must be an object")
    return payload


def _strict_date(value: object, *, evidence_name: str):
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise PaperAccountStateRefused(
            f"{evidence_name} is not a valid finite date: {value!r}"
        ) from exc
    if pd.isna(timestamp):
        raise PaperAccountStateRefused(
            f"{evidence_name} is not a valid finite date: {value!r}"
        )
    return timestamp.date()


def _paper_account_owner(
    config: DailyPaperLoopConfig,
    paper_account_identity_sha256: str,
) -> dict[str, str]:
    return {
        "paper_account_identity_sha256": str(paper_account_identity_sha256),
        "canonical_ledger_path": str(
            Path(config.canonical_ledger_path).resolve(strict=False)
        ),
        "execution_journal_path": str(
            Path(_execution_journal_path(config)).resolve(strict=False)
        ),
    }


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
