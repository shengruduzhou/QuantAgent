from __future__ import annotations

import pandas as pd
import pytest

from quantagent.domain.ledger import CanonicalLedger
from quantagent.paper.account_identity import ensure_paper_account_identity
from quantagent.paper.canonical_receipt import (
    CanonicalPrefixReceiptError,
    build_canonical_prefix_receipt,
    canonical_snapshot,
    verify_canonical_prefix_receipt,
)
from quantagent.paper.continuous_execution import (
    ContinuousPaperExecutionBlocked,
    ContinuousPaperExecutionConfig,
    execute_pending_for_session,
)
from quantagent.paper.execution_journal import PendingExecutionJournal
from quantagent.paper.pending_signal import PendingPaperSignalStore


FRIDAY = "2026-08-07"
MONDAY = "2026-08-10"
TUESDAY = "2026-08-11"
SYMBOL = "600000.SH"


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

    # Later economic history may extend the chain. The old terminal evidence is
    # still valid because it binds a prefix, not the current mutable tail.
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
    pending = PendingPaperSignalStore(pending_dir).record(
        signal_date=FRIDAY,
        target_weights=pd.DataFrame(
            {"trade_date": [pd.Timestamp(FRIDAY)], SYMBOL: [0.5]}
        ),
        source_lineage={
            "paper_account_identity_sha256": identity.payload_sha256,
        },
        created_at=f"{FRIDAY}T07:00:00+00:00",
    )[0]
    journal_path = tmp_path / "execution.jsonl"
    config = ContinuousPaperExecutionConfig(
        pending_signal_dir=str(pending_dir),
        execution_journal_path=str(journal_path),
        canonical_ledger_path=str(canonical),
        operational_ledger_path=str(tmp_path / "operational.jsonl"),
        idempotency_path=str(tmp_path / "idempotency.jsonl"),
        account_identity_path=str(identity_path),
        portfolio_id="v7-paper",
        initial_cash=100_000.0,
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


def test_indeterminate_terminal_freezes_entire_account(tmp_path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    journal_path = tmp_path / "execution.jsonl"
    journal = PendingExecutionJournal(journal_path)
    journal.append(
        pending_payload_sha256="legacy-crash",
        signal_date=FRIDAY,
        execution_date=MONDAY,
        status="execution_indeterminate",
        details={"reason": "crash during broker interaction"},
    )
    config = ContinuousPaperExecutionConfig(
        pending_signal_dir=str(tmp_path / "pending"),
        execution_journal_path=str(journal_path),
        canonical_ledger_path=str(canonical),
        operational_ledger_path=str(tmp_path / "operational.jsonl"),
        idempotency_path=str(tmp_path / "idempotency.jsonl"),
        account_identity_path=str(tmp_path / "account_identity.json"),
        portfolio_id="v7-paper",
        initial_cash=100_000.0,
    )

    with pytest.raises(ContinuousPaperExecutionBlocked, match="execution_indeterminate"):
        execute_pending_for_session(
            MONDAY,
            pd.DataFrame({"trade_date": [MONDAY]}),
            config=config,
            authoritative_sessions=[MONDAY],
        )
