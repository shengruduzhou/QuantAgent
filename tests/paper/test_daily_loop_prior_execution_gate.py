from __future__ import annotations

import pandas as pd
import pytest

from quantagent.paper.account_target_state import PaperAccountStateRefused
from quantagent.paper.canonical_receipt import (
    build_canonical_prefix_index,
    build_canonical_prefix_receipt,
)
from quantagent.paper.daily_loop import (
    DailyPaperLoopConfig,
    _assert_prior_pending_signals_resolved,
)
from quantagent.paper.execution_journal import PendingExecutionJournal
from quantagent.paper.pending_signal import PendingPaperSignalStore


_IDENTITY_SHA = "f" * 64


def _config(tmp_path, as_of: str) -> DailyPaperLoopConfig:
    return DailyPaperLoopConfig(
        as_of_date=as_of,
        pending_signal_dir=str(tmp_path / "pending"),
        execution_journal_path=str(tmp_path / "execution.jsonl"),
        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),
        account_identity_path=str(tmp_path / "identity.json"),
    )


def _record(store: PendingPaperSignalStore, signal_date: str):
    return store.record(
        signal_date=signal_date,
        target_weights=pd.DataFrame(
            [{"trade_date": pd.Timestamp(signal_date), "600000.SH": 0.25}]
        ),
        source_lineage={"test": "prior-execution-gate"},
        created_at="2026-08-13T00:00:00+00:00",
    )[0]


def _receipt(config: DailyPaperLoopConfig, prior) -> dict[str, object]:
    prefix = build_canonical_prefix_index(config.canonical_ledger_path)
    return build_canonical_prefix_receipt(
        ledger=config.canonical_ledger_path,
        canonical_before_records=prefix.record_count,
        canonical_before_head=prefix.current_head,
        target_weights_sha256=prior.target_weights_sha256,
        paper_account_identity_sha256=_IDENTITY_SHA,
    )


def _assert_resolved(config: DailyPaperLoopConfig, as_of: str) -> None:
    _assert_prior_pending_signals_resolved(
        config,
        as_of,
        paper_account_identity_sha256=_IDENTITY_SHA,
    )


def test_prior_unresolved_pending_signal_blocks_current_target_freeze(tmp_path) -> None:
    config = _config(tmp_path, "2026-08-11")
    prior = _record(PendingPaperSignalStore(config.pending_signal_dir), "2026-08-10")

    with pytest.raises(PaperAccountStateRefused, match="prior pending paper signal is unresolved"):
        _assert_resolved(config, "2026-08-11")

    journal = PendingExecutionJournal(config.execution_journal_path)
    journal.append(
        pending_payload_sha256=prior.payload_sha256,
        signal_date=prior.signal_date,
        execution_date="2026-08-11",
        status="execution_observed",
        details={
            "paper_account_identity_sha256": _IDENTITY_SHA,
            "target_weights_sha256": prior.target_weights_sha256,
            "canonical_prefix_receipt": _receipt(config, prior),
        },
        recorded_at="2026-08-13T00:01:00+00:00",
    )

    _assert_resolved(config, "2026-08-11")


def test_prior_indeterminate_signal_requires_explicit_bound_reconciliation(tmp_path) -> None:
    config = _config(tmp_path, "2026-08-11")
    prior = _record(PendingPaperSignalStore(config.pending_signal_dir), "2026-08-10")
    journal = PendingExecutionJournal(config.execution_journal_path)
    terminal = journal.append(
        pending_payload_sha256=prior.payload_sha256,
        signal_date=prior.signal_date,
        execution_date="2026-08-11",
        status="execution_indeterminate",
        details={
            "paper_account_identity_sha256": _IDENTITY_SHA,
            "target_weights_sha256": prior.target_weights_sha256,
            "canonical_prefix_receipt": _receipt(config, prior),
        },
        recorded_at="2026-08-13T00:01:00+00:00",
    )

    with pytest.raises(PaperAccountStateRefused, match="unreconciled execution_indeterminate"):
        _assert_resolved(config, "2026-08-11")

    prefix = build_canonical_prefix_index(config.canonical_ledger_path)
    journal.append(
        pending_payload_sha256=prior.payload_sha256,
        signal_date=prior.signal_date,
        execution_date="2026-08-11",
        status="execution_reconciled",
        details={
            "indeterminate_record_sha256": terminal.record_sha256,
            "paper_account_identity_sha256": _IDENTITY_SHA,
            "canonical_records": prefix.record_count,
            "canonical_head": prefix.current_head,
        },
        recorded_at="2026-08-13T00:02:00+00:00",
    )

    _assert_resolved(config, "2026-08-11")


def test_orphan_execution_started_blocks_even_after_pending_artifact_deleted(tmp_path) -> None:
    config = _config(tmp_path, "2026-08-11")
    journal = PendingExecutionJournal(config.execution_journal_path)
    journal.append(
        pending_payload_sha256="a" * 64,
        signal_date="2026-08-10",
        execution_date="2026-08-11",
        status="execution_started",
        details={
            "paper_account_identity_sha256": _IDENTITY_SHA,
            "target_weights_sha256": "b" * 64,
            "canonical_records_before": 0,
            "canonical_head_before": "0" * 64,
        },
        recorded_at="2026-08-13T00:01:00+00:00",
    )
    assert not (tmp_path / "pending").exists()

    with pytest.raises(PaperAccountStateRefused, match="unresolved execution_started"):
        _assert_resolved(config, "2026-08-11")


def test_orphan_indeterminate_blocks_even_after_pending_artifact_deleted(tmp_path) -> None:
    config = _config(tmp_path, "2026-08-11")
    prefix = build_canonical_prefix_index(config.canonical_ledger_path)
    receipt = build_canonical_prefix_receipt(
        ledger=config.canonical_ledger_path,
        canonical_before_records=prefix.record_count,
        canonical_before_head=prefix.current_head,
        target_weights_sha256="b" * 64,
        paper_account_identity_sha256=_IDENTITY_SHA,
    )
    journal = PendingExecutionJournal(config.execution_journal_path)
    journal.append(
        pending_payload_sha256="a" * 64,
        signal_date="2026-08-10",
        execution_date="2026-08-11",
        status="execution_indeterminate",
        details={
            "paper_account_identity_sha256": _IDENTITY_SHA,
            "target_weights_sha256": "b" * 64,
            "canonical_prefix_receipt": receipt,
        },
        recorded_at="2026-08-13T00:01:00+00:00",
    )
    assert not (tmp_path / "pending").exists()

    with pytest.raises(PaperAccountStateRefused, match="unreconciled execution_indeterminate"):
        _assert_resolved(config, "2026-08-11")


def test_current_date_pending_signal_is_not_treated_as_prior_execution(tmp_path) -> None:
    config = _config(tmp_path, "2026-08-11")
    _record(PendingPaperSignalStore(config.pending_signal_dir), "2026-08-11")

    _assert_resolved(config, "2026-08-11")


def test_execution_journal_field_preserves_legacy_positional_config_order() -> None:
    config = DailyPaperLoopConfig(
        "2026-08-11",
        "model",
        "features",
        "market",
        "sector",
        "output",
        "book",
        "pending",
        "canonical",
        "identity",
    )

    assert config.pending_signal_dir == "pending"
    assert config.canonical_ledger_path == "canonical"
    assert config.account_identity_path == "identity"
    assert config.execution_journal_path not in {"canonical", "identity"}


def test_backdated_decision_refuses_later_pending_signal(tmp_path) -> None:
    later = _config(tmp_path, "2026-08-12")
    _record(PendingPaperSignalStore(later.pending_signal_dir), "2026-08-12")
    earlier = _config(tmp_path, "2026-08-11")
    with pytest.raises(PaperAccountStateRefused, match="chronology regression.*later pending"):
        _assert_prior_pending_signals_resolved(
            earlier,
            "2026-08-11",
            paper_account_identity_sha256=_IDENTITY_SHA,
        )


def test_backdated_decision_refuses_later_durable_journal_record(tmp_path) -> None:
    earlier = _config(tmp_path, "2026-08-11")
    PendingExecutionJournal(earlier.execution_journal_path).append(
        pending_payload_sha256="f" * 64,
        signal_date="2026-08-12",
        execution_date="2026-08-12",
        status="execution_started",
        details={"paper_account_identity_sha256": _IDENTITY_SHA},
    )
    with pytest.raises(PaperAccountStateRefused, match="chronology regression.*later durable"):
        _assert_prior_pending_signals_resolved(
            earlier,
            "2026-08-11",
            paper_account_identity_sha256=_IDENTITY_SHA,
        )
