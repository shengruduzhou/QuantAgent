from __future__ import annotations

from pathlib import Path
from hashlib import sha256

import pandas as pd

from quantagent.domain.ledger import CanonicalLedger
from quantagent.paper.account_identity import ensure_paper_account_identity
from quantagent.paper.canonical_receipt import (
    build_canonical_prefix_receipt,
    canonical_snapshot,
)
from quantagent.paper.execution_journal import PendingExecutionJournal
from quantagent.paper.continuous_execution import (
    ContinuousPaperExecutionConfig,
    bind_legacy_terminal_account,
)
from quantagent.paper.execution_journal import DAILY_DECISION_STATUS
from quantagent.paper.pending_signal import PENDING_COMMIT_PROTOCOL, PendingPaperSignalStore
from quantagent.paper.runtime_paths import paper_runtime_paths
from services.quant_api.config import ApiSettings
from services.quant_api.services.paper_execution_evidence import (
    PaperExecutionEvidenceService,
)


def _settings(tmp_path: Path) -> ApiSettings:
    runtime = tmp_path / "runtime"
    return ApiSettings(
        project_root=tmp_path,
        runtime_root=runtime,
        cache_root=runtime / "cache" / "quant_ui",
        jobs_root=runtime / "jobs" / "quant_ui",
    ).ensure()


def _write_bound_terminal(settings: ApiSettings) -> tuple[str, str]:
    paths = paper_runtime_paths(settings.runtime_root)
    identity = ensure_paper_account_identity(
        canonical_ledger_path=paths.canonical_ledger,
        portfolio_id="v7-paper",
        initial_cash=1_000_000.0,
        identity_path=paths.account_identity,
    )
    ledger = CanonicalLedger(paths.canonical_ledger)
    before_records, before_head = canonical_snapshot(ledger)
    receipt = build_canonical_prefix_receipt(
        ledger=ledger,
        canonical_before_records=before_records,
        canonical_before_head=before_head,
        target_weights_sha256="target-sha",
        paper_account_identity_sha256=identity.payload_sha256,
    )
    PendingExecutionJournal(paths.execution_journal).append(
        pending_payload_sha256="payload-sha",
        signal_date="2026-08-07",
        execution_date="2026-08-10",
        status="missed_execution_session",
        details={
            "target_weights_sha256": "target-sha",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_prefix_receipt": receipt,
            "calendar_assurance": "caller_supplied_session_set_unverified",
            "shadow_acceptance_calendar_eligible": False,
            "production_pretrade_risk_certified": False,
        },
    )
    return identity.account_instance_id, identity.payload_sha256


def _write_frozen_target(settings: ApiSettings):
    paths = paper_runtime_paths(settings.runtime_root)
    identity = ensure_paper_account_identity(
        canonical_ledger_path=paths.canonical_ledger,
        portfolio_id="v7-paper",
        initial_cash=1_000_000.0,
        identity_path=paths.account_identity,
    )
    records, head = canonical_snapshot(paths.canonical_ledger)
    pending = PendingPaperSignalStore(paths.pending_signals).record(
        signal_date="2026-08-07",
        target_weights=pd.DataFrame(
            [{"trade_date": pd.Timestamp("2026-08-07"), "600000.SH": 0.25}]
        ),
        source_lineage={
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_ledger_head_hash": head,
            "canonical_ledger_records": str(records),
            "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,
        },
    )[0]
    summary_path = paths.root / "summary.json"
    summary_path.write_text('{"status":"pending"}\n', encoding="utf-8")
    PendingExecutionJournal(paths.execution_journal).append(
        pending_payload_sha256=pending.payload_sha256,
        signal_date=pending.signal_date,
        execution_date=pending.signal_date,
        status=DAILY_DECISION_STATUS,
        details={
            "decision_kind": "target",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_records": records,
            "canonical_head": head,
            "assurance": "canonical_account_daily_decision_freeze_v1",
            "commit_protocol": PENDING_COMMIT_PROTOCOL,
            "target_weights_sha256": pending.target_weights_sha256,
            "daily_summary_path": str(summary_path.resolve()),
            "daily_summary_sha256": sha256(summary_path.read_bytes()).hexdigest(),
        },
    )
    return identity, pending


def test_api_projects_verified_account_identity_and_bound_terminal_prefix(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    instance_id, identity_sha = _write_bound_terminal(settings)

    result = PaperExecutionEvidenceService(settings).status()

    assert result["journal"]["state"] == "valid"
    assert result["accountIdentity"]["state"] == "valid"
    assert result["accountIdentity"]["verified"] is True
    assert result["accountIdentity"]["accountInstanceId"] == instance_id
    assert result["accountIdentity"]["portfolioId"] == "v7-paper"
    assert result["accountIdentity"]["initialCashCny"] == "1000000.00"
    assert result["accountIdentity"]["payloadSha256"] == identity_sha
    assert result["canonicalPrefix"]["state"] == "valid"
    assert result["canonicalPrefix"]["boundTerminalCount"] == 1
    assert result["canonicalPrefix"]["legacyUnboundTerminalCount"] == 0
    assert result["canonicalPrefix"]["latestTerminalBound"] is True
    assert result["summary"]["accountIdentityVerified"] is True
    assert result["summary"]["latestTerminalCanonicalPrefixBound"] is True
    assert result["operatorTruth"]["accountIdentityVerified"] is True
    assert result["operatorTruth"]["canonicalExecutionPrefixCertified"] is True
    assert result["operatorTruth"]["productionLiveCertified"] is False


def test_invalid_account_identity_is_operator_critical_even_without_journal(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    paths = paper_runtime_paths(settings.runtime_root)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.account_identity.write_text("{not-valid-json", encoding="utf-8")

    result = PaperExecutionEvidenceService(settings).status()

    assert result["accountIdentity"]["state"] == "invalid"
    assert result["accountIdentity"]["verified"] is False
    assert result["summary"]["attention"] == "critical"
    assert result["operatorTruth"]["accountIdentityVerified"] is False
    assert result["operatorTruth"]["productionLiveCertified"] is False


def test_tampered_canonical_prefix_receipt_invalidates_operator_evidence(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    paths = paper_runtime_paths(settings.runtime_root)
    identity = ensure_paper_account_identity(
        canonical_ledger_path=paths.canonical_ledger,
        portfolio_id="v7-paper",
        initial_cash=1_000_000.0,
        identity_path=paths.account_identity,
    )
    ledger = CanonicalLedger(paths.canonical_ledger)
    before_records, before_head = canonical_snapshot(ledger)
    receipt = build_canonical_prefix_receipt(
        ledger=ledger,
        canonical_before_records=before_records,
        canonical_before_head=before_head,
        target_weights_sha256="target-sha",
        paper_account_identity_sha256=identity.payload_sha256,
    )
    receipt["canonical_after_head"] = "f" * 64
    PendingExecutionJournal(paths.execution_journal).append(
        pending_payload_sha256="payload-sha",
        signal_date="2026-08-07",
        execution_date="2026-08-10",
        status="execution_observed",
        details={
            "target_weights_sha256": "target-sha",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_prefix_receipt": receipt,
        },
    )

    result = PaperExecutionEvidenceService(settings).status()

    assert result["journal"]["state"] == "invalid"
    assert result["canonicalPrefix"]["state"] == "invalid"
    assert result["summary"]["attention"] == "critical"
    assert result["operatorTruth"]["paperExecutionEvidence"] is False
    assert result["operatorTruth"]["canonicalExecutionPrefixCertified"] is False
    assert result["operatorTruth"]["productionLiveCertified"] is False


def test_structurally_malformed_canonical_json_is_projected_invalid_not_raised(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    paths = paper_runtime_paths(settings.runtime_root)
    identity = ensure_paper_account_identity(
        canonical_ledger_path=paths.canonical_ledger,
        portfolio_id="v7-paper",
        initial_cash=1_000_000.0,
        identity_path=paths.account_identity,
    )
    before_records, before_head = canonical_snapshot(paths.canonical_ledger)
    receipt = build_canonical_prefix_receipt(
        ledger=paths.canonical_ledger,
        canonical_before_records=before_records,
        canonical_before_head=before_head,
        target_weights_sha256="target-sha",
        paper_account_identity_sha256=identity.payload_sha256,
    )
    PendingExecutionJournal(paths.execution_journal).append(
        pending_payload_sha256="payload-sha",
        signal_date="2026-08-07",
        execution_date="2026-08-10",
        status="execution_observed",
        details={
            "target_weights_sha256": "target-sha",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_prefix_receipt": receipt,
        },
    )
    # Valid JSON, invalid canonical record structure: this used to escape the
    # evidence service as KeyError/TypeError instead of fail-closed projection.
    paths.canonical_ledger.write_text('{"schema_version":"bad"}\n', encoding="utf-8")

    result = PaperExecutionEvidenceService(settings).status()

    assert result["journal"]["state"] == "invalid"
    assert result["canonicalPrefix"]["state"] == "invalid"
    assert result["summary"]["attention"] == "critical"
    assert result["operatorTruth"]["productionLiveCertified"] is False


def test_committed_target_is_pending_not_ok_or_executed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_frozen_target(settings)

    result = PaperExecutionEvidenceService(settings).status()

    assert result["journal"]["committedPendingCount"] == 1
    assert result["journal"]["stagedUncommittedCount"] == 0
    assert result["summary"]["latestStatus"] == DAILY_DECISION_STATUS
    assert result["summary"]["latestDecisionKind"] == "target"
    assert result["summary"]["attention"] == "pending"
    assert result["summary"]["executionEvidenceAvailable"] is False
    assert result["operatorTruth"]["paperExecutionEvidence"] is False
    assert result["canonicalPrefix"]["currentRecordCount"] == 0
    assert len(result["canonicalPrefix"]["currentHeadHash"]) == 64


def test_committed_no_target_is_explicitly_not_execution_evidence(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    paths = paper_runtime_paths(settings.runtime_root)
    identity = ensure_paper_account_identity(
        canonical_ledger_path=paths.canonical_ledger,
        portfolio_id="v7-paper",
        initial_cash=1_000_000.0,
        identity_path=paths.account_identity,
    )
    records, head = canonical_snapshot(paths.canonical_ledger)
    PendingExecutionJournal(paths.execution_journal).append(
        pending_payload_sha256="a" * 64,
        signal_date="2026-08-07",
        execution_date="2026-08-07",
        status=DAILY_DECISION_STATUS,
        details={
            "decision_kind": "no_target",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_records": records,
            "canonical_head": head,
            "assurance": "canonical_account_daily_decision_freeze_v1",
        },
    )

    result = PaperExecutionEvidenceService(settings).status()
    assert result["summary"]["latestDecisionKind"] == "no_target"
    assert result["summary"]["attention"] == "ok"
    assert result["operatorTruth"]["paperExecutionEvidence"] is False
    assert "no-target" in result["operatorTruth"]["message"]


def test_staged_uncommitted_target_is_visible_but_never_execution_evidence(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    paths = paper_runtime_paths(settings.runtime_root)
    PendingPaperSignalStore(paths.pending_signals).record(
        signal_date="2026-08-07",
        target_weights=pd.DataFrame(
            [{"trade_date": pd.Timestamp("2026-08-07"), "600000.SH": 0.25}]
        ),
        source_lineage={"test": "staged"},
    )

    result = PaperExecutionEvidenceService(settings).status()
    assert result["journal"]["state"] == "unavailable"
    assert result["journal"]["stagedUncommittedCount"] == 1
    assert result["summary"]["latestStatus"] == "staged_uncommitted"
    assert result["summary"]["attention"] == "warning"
    assert result["operatorTruth"]["paperExecutionEvidence"] is False


def test_valid_legacy_binding_is_projected_as_lower_assurance_bound(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    paths = paper_runtime_paths(settings.runtime_root)
    PendingExecutionJournal(paths.execution_journal).append(
        pending_payload_sha256="b" * 64,
        signal_date="2026-08-07",
        execution_date="2026-08-10",
        status="execution_observed",
        details={"target_weights_sha256": "c" * 64},
    )
    bind_legacy_terminal_account(
        config=ContinuousPaperExecutionConfig(
            pending_signal_dir=str(paths.pending_signals),
            execution_journal_path=str(paths.execution_journal),
            canonical_ledger_path=str(paths.canonical_ledger),
            operational_ledger_path=str(paths.operational_ledger),
            idempotency_path=str(paths.idempotency),
            account_identity_path=str(paths.account_identity),
        ),
        pending_payload_sha256="b" * 64,
        as_of_date="2026-08-11",
        reason="operator verified canonical account",
    )

    result = PaperExecutionEvidenceService(settings).status()
    assert result["canonicalPrefix"]["boundTerminalCount"] == 1
    assert result["canonicalPrefix"]["legacyBoundTerminalCount"] == 1
    assert result["canonicalPrefix"]["legacyUnboundTerminalCount"] == 0
    assert result["canonicalPrefix"]["latestTerminalBound"] is True
    assert result["canonicalPrefix"]["latestTerminalBindingAssurance"] == "operator_bound_legacy"
    assert result["summary"]["latestStatus"] == "legacy_terminal_bound"
    assert result["summary"]["attention"] == "warning"
