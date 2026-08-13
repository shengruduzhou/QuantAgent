from __future__ import annotations

import pandas as pd
import pytest

from quantagent.domain.ledger import CanonicalLedger
from quantagent.paper.account_identity import ensure_paper_account_identity
from quantagent.paper.canonical_receipt import (
    CanonicalPrefixReceiptError,
    build_canonical_prefix_index,
    build_canonical_prefix_receipt,
    canonical_snapshot,
    verify_canonical_prefix_receipt,
)
from quantagent.paper.continuous_execution import (
    ContinuousPaperExecutionBlocked,
    ContinuousPaperExecutionConfig,
    execute_pending_for_session,
    reconcile_indeterminate_account,
)
from quantagent.paper.execution_journal import (
    DAILY_DECISION_STATUS,
    RECONCILIATION_STATUS,
    PendingExecutionJournal,
)
from quantagent.paper.pending_signal import (
    PENDING_COMMIT_PROTOCOL,
    PendingPaperSignalStore,
)


FRIDAY = "2026-08-07"
MONDAY = "2026-08-10"
TUESDAY = "2026-08-11"
SYMBOL = "600000.SH"


def _config(tmp_path, *, canonical, identity_path, journal_path):
    return ContinuousPaperExecutionConfig(
        pending_signal_dir=str(tmp_path / "pending"),
        execution_journal_path=str(journal_path),
        canonical_ledger_path=str(canonical),
        operational_ledger_path=str(tmp_path / "operational.jsonl"),
        idempotency_path=str(tmp_path / "idempotency.jsonl"),
        account_identity_path=str(identity_path),
        portfolio_id="v7-paper",
        initial_cash=100_000.0,
    )


def test_prefix_receipt_survives_later_canonical_appends(tmp_path) -> None:
    ledger = CanonicalLedger(tmp_path / "canonical.jsonl")
    before_records, before_head = canonical_snapshot(ledger)
    receipt = build_canonical_prefix_receipt(
        ledger=ledger,
        canonical_before_records=before_records,
        canonical_before_head=before_head,
        target_weights_sha256="target-a",
        paper_account_identity_sha256="identity-a",
    )

    first = verify_canonical_prefix_receipt(
        receipt,
        ledger_or_path=ledger,
        expected_target_weights_sha256="target-a",
        expected_paper_account_identity_sha256="identity-a",
    )
    assert first.bound is True
    assert first.valid is True
    assert first.canonical_after_records == 0

    ledger.append(None, trade_date=MONDAY)
    later = verify_canonical_prefix_receipt(
        receipt,
        ledger_or_path=ledger,
        expected_target_weights_sha256="target-a",
        expected_paper_account_identity_sha256="identity-a",
    )
    assert later.bound is True
    assert later.current_records == 1
    assert later.canonical_after_records == 0


def test_terminal_receipt_path_reopens_durable_ledger_instead_of_cached_instance(tmp_path) -> None:
    path = tmp_path / "canonical.jsonl"
    stale = CanonicalLedger(path)
    before_records, before_head = canonical_snapshot(path)

    # Simulate a different economic writer after the long-lived consumer loaded
    # its CanonicalLedger object. The terminal seal must see this durable record.
    CanonicalLedger(path).append(None, trade_date=MONDAY)
    assert len(stale) == 0

    receipt = build_canonical_prefix_receipt(
        ledger=path,
        canonical_before_records=before_records,
        canonical_before_head=before_head,
        target_weights_sha256="target-a",
        paper_account_identity_sha256="identity-a",
    )
    assert receipt["canonical_after_records"] == 1
    assert receipt["canonical_after_head"] == canonical_snapshot(path)[1]


def test_prefix_index_supports_many_receipts_without_reloading_ledger(tmp_path) -> None:
    path = tmp_path / "canonical.jsonl"
    ledger = CanonicalLedger(path)
    ledger.append(None, trade_date=MONDAY)
    index = build_canonical_prefix_index(path)
    receipt = {
        "schema_version": "quantagent.paper.canonical_prefix_receipt.v1",
        "canonical_before_records": 0,
        "canonical_before_head": index.head_at(0),
        "canonical_after_records": 1,
        "canonical_after_head": index.head_at(1),
        "target_weights_sha256": "target-a",
        "paper_account_identity_sha256": "identity-a",
    }
    for _ in range(100):
        verified = verify_canonical_prefix_receipt(receipt, prefix_index=index)
        assert verified.valid is True
        assert verified.current_records == 1


def test_claimed_prefix_receipt_with_wrong_head_fails_closed(tmp_path) -> None:
    ledger = CanonicalLedger(tmp_path / "canonical.jsonl")
    before_records, before_head = canonical_snapshot(ledger)
    receipt = build_canonical_prefix_receipt(
        ledger=ledger,
        canonical_before_records=before_records,
        canonical_before_head=before_head,
        target_weights_sha256="target-a",
        paper_account_identity_sha256="identity-a",
    )
    tampered = dict(receipt)
    tampered["canonical_after_head"] = "f" * 64

    with pytest.raises(CanonicalPrefixReceiptError, match="post-execution prefix"):
        verify_canonical_prefix_receipt(tampered, ledger_or_path=ledger)


def test_legacy_terminal_without_receipt_is_explicitly_unbound(tmp_path) -> None:
    ledger = CanonicalLedger(tmp_path / "canonical.jsonl")
    verification = verify_canonical_prefix_receipt(None, ledger_or_path=ledger)
    assert verification.valid is True
    assert verification.bound is False
    assert verification.reason == "legacy_terminal_without_canonical_prefix_receipt"


def test_missed_session_terminal_is_bound_to_same_canonical_prefix(tmp_path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    identity_path = tmp_path / "account_identity.json"
    identity = ensure_paper_account_identity(
        canonical_ledger_path=canonical,
        portfolio_id="v7-paper",
        initial_cash=100_000.0,
        identity_path=identity_path,
    )
    pending_dir = tmp_path / "pending"
    summary_path = tmp_path / "daily_loop_summary.json"
    summary_path.write_text('{"test":"committed-summary"}\n', encoding="utf-8")
    summary_sha = __import__("hashlib").sha256(summary_path.read_bytes()).hexdigest()
    pending = PendingPaperSignalStore(pending_dir).record(
        signal_date=FRIDAY,
        target_weights=pd.DataFrame(
            {"trade_date": [pd.Timestamp(FRIDAY)], SYMBOL: [0.5]}
        ),
        source_lineage={
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_ledger_head_hash": "0" * 64,
            "canonical_ledger_records": "0",
            "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,
        },
        created_at=f"{FRIDAY}T07:00:00+00:00",
    )[0]
    journal_path = tmp_path / "execution.jsonl"
    PendingExecutionJournal(journal_path).append(
        pending_payload_sha256=pending.payload_sha256,
        signal_date=FRIDAY,
        execution_date=FRIDAY,
        status=DAILY_DECISION_STATUS,
        details={
            "decision_kind": "target",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_records": 0,
            "canonical_head": "0" * 64,
            "assurance": "canonical_account_daily_decision_freeze_v1",
            "commit_protocol": PENDING_COMMIT_PROTOCOL,
            "target_weights_sha256": pending.target_weights_sha256,
            "daily_summary_path": str(summary_path.resolve()),
            "daily_summary_sha256": summary_sha,
            "daily_summary_commit_protocol": "daily_summary_bound_daily_decision_v1",
        },
    )
    config = _config(
        tmp_path,
        canonical=canonical,
        identity_path=identity_path,
        journal_path=journal_path,
    )

    result = execute_pending_for_session(
        TUESDAY,
        pd.DataFrame({"trade_date": [FRIDAY, MONDAY, TUESDAY]}),
        config=config,
        authoritative_sessions=[FRIDAY, MONDAY, TUESDAY],
    )
    assert len(result) == 1
    assert result[0].status == "missed_execution_session"

    terminal = PendingExecutionJournal(journal_path).terminal(pending.payload_sha256)
    assert terminal is not None
    receipt = terminal.details["canonical_prefix_receipt"]
    verified = verify_canonical_prefix_receipt(
        receipt,
        ledger_or_path=canonical,
        expected_target_weights_sha256=pending.target_weights_sha256,
        expected_paper_account_identity_sha256=identity.payload_sha256,
    )
    assert verified.bound is True
    assert verified.canonical_before_records == verified.canonical_after_records == 0


def test_unresolved_start_freezes_account_even_when_pending_artifact_is_missing(tmp_path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    identity_path = tmp_path / "account_identity.json"
    identity = ensure_paper_account_identity(
        canonical_ledger_path=canonical,
        portfolio_id="v7-paper",
        initial_cash=100_000.0,
        identity_path=identity_path,
    )
    before_records, before_head = canonical_snapshot(canonical)
    journal_path = tmp_path / "execution.jsonl"
    PendingExecutionJournal(journal_path).append(
        pending_payload_sha256="deleted-pending-payload",
        signal_date=FRIDAY,
        execution_date=MONDAY,
        status="execution_started",
        details={
            "target_weights_sha256": "target-sha",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_records_before": before_records,
            "canonical_head_before": before_head,
        },
    )
    config = _config(
        tmp_path,
        canonical=canonical,
        identity_path=identity_path,
        journal_path=journal_path,
    )

    with pytest.raises(ContinuousPaperExecutionBlocked, match="unresolved execution_started"):
        execute_pending_for_session(
            MONDAY,
            pd.DataFrame({"trade_date": [MONDAY]}),
            config=config,
            authoritative_sessions=[MONDAY],
        )


def test_explicit_reconciliation_is_append_only_and_clears_indeterminate_freeze(tmp_path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    identity_path = tmp_path / "account_identity.json"
    identity = ensure_paper_account_identity(
        canonical_ledger_path=canonical,
        portfolio_id="v7-paper",
        initial_cash=100_000.0,
        identity_path=identity_path,
    )
    before_records, before_head = canonical_snapshot(canonical)
    journal_path = tmp_path / "execution.jsonl"
    journal = PendingExecutionJournal(journal_path)
    journal.append(
        pending_payload_sha256="crashed-payload",
        signal_date=FRIDAY,
        execution_date=MONDAY,
        status="execution_started",
        details={
            "target_weights_sha256": "target-sha",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_records_before": before_records,
            "canonical_head_before": before_head,
        },
    )
    config = _config(
        tmp_path,
        canonical=canonical,
        identity_path=identity_path,
        journal_path=journal_path,
    )

    appended = reconcile_indeterminate_account(
        config=config,
        as_of_date=MONDAY,
        reason="operator verified canonical and operational paper state",
    )
    assert len(appended) == 1
    assert appended[0]["status"] == RECONCILIATION_STATUS

    history = journal.history("crashed-payload")
    assert [row.status for row in history] == [
        "execution_started",
        "execution_indeterminate",
        RECONCILIATION_STATUS,
    ]
    terminal = journal.terminal("crashed-payload")
    reconciliation = journal.reconciliation("crashed-payload")
    assert terminal is not None and reconciliation is not None
    assert reconciliation.details["indeterminate_record_sha256"] == terminal.record_sha256

    # No pending files exist, but a successfully reconciled incident no longer
    # permanently bricks the account.
    assert execute_pending_for_session(
        MONDAY,
        pd.DataFrame({"trade_date": [MONDAY]}),
        config=config,
        authoritative_sessions=[MONDAY],
    ) == []


def test_indeterminate_terminal_freezes_entire_account(tmp_path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    identity_path = tmp_path / "account_identity.json"
    ensure_paper_account_identity(
        canonical_ledger_path=canonical,
        portfolio_id="v7-paper",
        initial_cash=100_000.0,
        identity_path=identity_path,
    )
    journal_path = tmp_path / "execution.jsonl"
    PendingExecutionJournal(journal_path).append(
        pending_payload_sha256="legacy-crash",
        signal_date=FRIDAY,
        execution_date=MONDAY,
        status="execution_indeterminate",
        details={"reason": "crash during broker interaction"},
    )
    config = _config(
        tmp_path,
        canonical=canonical,
        identity_path=identity_path,
        journal_path=journal_path,
    )

    with pytest.raises(ContinuousPaperExecutionBlocked, match="unreconciled execution_indeterminate"):
        execute_pending_for_session(
            MONDAY,
            pd.DataFrame({"trade_date": [MONDAY]}),
            config=config,
            authoritative_sessions=[MONDAY],
        )
