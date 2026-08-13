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
holding-period and turnover state. Before a non-empty target is frozen, the
canonical account is recovered again while holding the cross-process account
lock. Any ledger/state drift makes the stale target fail closed, every older
pending signal must already have a canonically verified terminal execution
outcome, and same-date evidence files are written only inside that account-wide
critical section. Desired targets are never treated as holdings. The account
genesis itself is immutable: every run must match the persisted
portfolio_id/initial_cash identity before any target is produced.
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
from quantagent.paper.execution_journal import PendingExecutionJournal
from quantagent.paper.pending_signal import PendingPaperSignalStore
from quantagent.paper.runtime_paths import paper_runtime_paths
from quantagent.portfolio.multi_horizon_blender import MultiHorizonBlendConfig, blend_multi_horizon_predictions
from quantagent.portfolio.v7_target_weights import V7TargetWeightsConfig, build_v7_target_weights, write_v7_target_weights
from quantagent.training.v7_predictor import predict_v7_alpha


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
    execution_journal_path: str = field(default_factory=lambda: str(paper_runtime_paths().execution_journal))
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
    construction. Immediately before a non-empty pending intent is frozen, the
    account is recovered again under the same cross-process lock used by the
    continuous execution worker. Older pending signals must already be resolved
    with valid canonical evidence; a changed state/head/count invalidates the
    computed target instead of silently executing it against different holdings.

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
            _assert_current_signal_not_frozen(config, as_of)
            predictions_path = write_frame(predictions, day_dir / "predictions.parquet")
            weights_path = write_v7_target_weights(
                weights,
                day_dir / "target_weights.parquet",
            )
        warnings = tuple([*evidence.warnings, "paper_no_target_generated"])
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

    with paper_account_lock(config.canonical_ledger_path):
        _assert_current_signal_not_frozen(config, as_of)
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
        if (
            fresh_account_state.account_state_sha256 != account_state.account_state_sha256
            or fresh_account_state.canonical_records != account_state.canonical_records
            or fresh_account_state.canonical_head_hash != account_state.canonical_head_hash
        ):
            raise PaperAccountStateRefused(
                "canonical paper account changed during target construction; "
                "stale target was discarded before pending freeze. Rerun target "
                "generation from the freshly executed account state"
            )
        account_state = fresh_account_state
        account_evidence = account_state.evidence()

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


def _assert_current_signal_not_frozen(
    config: DailyPaperLoopConfig,
    as_of: str,
) -> None:
    """Prevent a second same-date writer from mutating hash-bound evidence files.

    The account lock must already be held. A frozen pending signal is immutable
    economic evidence; its referenced prediction/target files must therefore be
    immutable too. Re-running inference for the same signal date is allowed only
    after an explicit operator workflow has dealt with the existing frozen
    intent, rather than silently replacing its lineage artifacts.
    """

    existing = PendingPaperSignalStore(config.pending_signal_dir).read(as_of)
    if existing is not None:
        raise PaperAccountStateRefused(
            "pending paper signal is already frozen for the current signal date; "
            "refusing to overwrite its hash-bound predictions/target artifacts"
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


def _assert_prior_pending_signals_resolved(
    config: DailyPaperLoopConfig,
    as_of: str,
    *,
    paper_account_identity_sha256: str,
) -> None:
    """Require previous signal dates to reach a durable terminal before freeze.

    The cross-process account lock must already be held by the caller. This turns
    the operational ordering contract into a fail-closed invariant: on session T,
    a T-1 pending signal cannot still be waiting for execution while a new T
    target is frozen from the pre-execution account. Terminals and indeterminate
    reconciliations are independently bound back to the canonical ledger before
    they are accepted as resolved.
    """

    root = Path(config.pending_signal_dir)
    if not root.exists():
        return
    journal = PendingExecutionJournal(config.execution_journal_path)
    if not journal.verify():
        raise PaperAccountStateRefused(
            "pending execution journal verification failed before target freeze"
        )
    prefix_index = build_canonical_prefix_index(config.canonical_ledger_path)
    store = PendingPaperSignalStore(root)
    cutoff = pd.Timestamp(as_of).date()
    for path in sorted(root.glob("*.json")):
        try:
            signal_date = pd.Timestamp(path.stem).date()
        except (TypeError, ValueError) as exc:
            raise PaperAccountStateRefused(
                f"unexpected pending-signal filename before target freeze: {path.name}"
            ) from exc
        if signal_date >= cutoff:
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
        receipt = dict(terminal.details or {}).get("canonical_prefix_receipt")
        try:
            verification = verify_canonical_prefix_receipt(
                receipt,
                prefix_index=prefix_index,
                expected_target_weights_sha256=signal.target_weights_sha256,
                expected_paper_account_identity_sha256=(
                    paper_account_identity_sha256 if receipt is not None else None
                ),
            )
        except CanonicalPrefixReceiptError as exc:
            raise PaperAccountStateRefused(
                "prior pending terminal no longer matches the canonical account: "
                f"signal_date={signal.signal_date}: {exc}"
            ) from exc
        if not verification.bound:
            raise PaperAccountStateRefused(
                "prior pending terminal lacks a canonical-prefix/account-identity "
                f"receipt: signal_date={signal.signal_date}; explicitly reconcile "
                "legacy evidence before freezing a new target"
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