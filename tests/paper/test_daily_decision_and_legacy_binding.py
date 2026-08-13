from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import quantagent.paper.continuous_execution as continuous_execution
import quantagent.paper.daily_loop as daily_loop
import pytest

from quantagent.paper.account_target_state import PaperAccountStateRefused
from quantagent.paper.canonical_receipt import build_canonical_prefix_index
from quantagent.paper.continuous_execution import (
    ContinuousPaperExecutionBlocked,
    ContinuousPaperExecutionConfig,
    bind_legacy_terminal_account,
    reconcile_indeterminate_account,
)
from quantagent.paper.daily_loop import (
    DailyPaperLoopConfig,
    _assert_current_signal_not_frozen,
    _assert_prior_pending_signals_resolved,
    _freeze_daily_decision,
)
from quantagent.paper.execution_journal import PendingExecutionJournal
from quantagent.paper.pending_signal import PendingPaperSignalStore


def _daily_config(tmp_path, as_of: str = "2026-08-11") -> DailyPaperLoopConfig:
    return DailyPaperLoopConfig(
        as_of_date=as_of,
        output_root=str(tmp_path / "reports"),
        pending_signal_dir=str(tmp_path / "pending"),
        execution_journal_path=str(tmp_path / "execution.jsonl"),
        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),
        account_identity_path=str(tmp_path / "identity.json"),
    )


def _continuous_config(tmp_path) -> ContinuousPaperExecutionConfig:
    return ContinuousPaperExecutionConfig(
        pending_signal_dir=str(tmp_path / "pending"),
        execution_journal_path=str(tmp_path / "execution.jsonl"),
        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),
        operational_ledger_path=str(tmp_path / "operational.jsonl"),
        idempotency_path=str(tmp_path / "idempotency.json"),
        account_identity_path=str(tmp_path / "identity.json"),
    )


def _pending(config: DailyPaperLoopConfig, signal_date: str):
    return PendingPaperSignalStore(config.pending_signal_dir).record(
        signal_date=signal_date,
        target_weights=pd.DataFrame(
            [{"trade_date": pd.Timestamp(signal_date), "600000.SH": 0.25}]
        ),
        source_lineage={"test": "legacy-migration"},
        created_at="2026-08-13T00:00:00+00:00",
    )[0]


def test_daily_decision_marker_survives_without_pending_json(tmp_path) -> None:
    config = _daily_config(tmp_path)
    prefix = build_canonical_prefix_index(config.canonical_ledger_path)
    summary_path = daily_loop._write_daily_summary(
        Path(config.output_root) / "2026-08-11",
        {
            "daily_decision_commit_protocol": daily_loop.DAILY_SUMMARY_COMMIT_PROTOCOL,
            "status": "no_target_generated",
        },
    )
    summary_sha = daily_loop._file_sha256(summary_path)
    _freeze_daily_decision(
        config,
        "2026-08-11",
        decision_kind="no_target",
        paper_account_identity_sha256="f" * 64,
        account_evidence={
            "account_state_sha256": "a" * 64,
            "canonical_records": prefix.record_count,
            "canonical_head_hash": prefix.current_head,
        },
        daily_summary_path=summary_path,
        daily_summary_sha256=summary_sha,
    )
    assert not (tmp_path / "pending").exists()
    with pytest.raises(PaperAccountStateRefused, match="already durably frozen"):
        _assert_current_signal_not_frozen(
            config,
            "2026-08-11",
            paper_account_identity_sha256="f" * 64,
        )


def test_legacy_same_date_terminal_blocks_artifact_reuse_after_pending_deleted(tmp_path) -> None:
    config = _daily_config(tmp_path)
    PendingExecutionJournal(config.execution_journal_path).append(
        pending_payload_sha256="b" * 64,
        signal_date="2026-08-11",
        execution_date="2026-08-12",
        status="execution_observed",
        details={"target_weights_sha256": "c" * 64},
        recorded_at="2026-08-13T00:00:00+00:00",
    )
    with pytest.raises(PaperAccountStateRefused, match="legacy same-date execution evidence"):
        _assert_current_signal_not_frozen(
            config,
            "2026-08-11",
            paper_account_identity_sha256="f" * 64,
        )


def test_operator_legacy_binding_is_append_only_and_accepted_by_daily_gate(tmp_path) -> None:
    daily = _daily_config(tmp_path, "2026-08-12")
    prior = _pending(daily, "2026-08-10")
    journal = PendingExecutionJournal(daily.execution_journal_path)
    terminal = journal.append(
        pending_payload_sha256=prior.payload_sha256,
        signal_date=prior.signal_date,
        execution_date="2026-08-11",
        status="execution_observed",
        details={"target_weights_sha256": prior.target_weights_sha256},
        recorded_at="2026-08-13T00:01:00+00:00",
    )
    with pytest.raises(PaperAccountStateRefused, match="operator-reconciled legacy binding"):
        _assert_prior_pending_signals_resolved(
            daily, "2026-08-12", paper_account_identity_sha256="f" * 64
        )
    binding = bind_legacy_terminal_account(
        config=_continuous_config(tmp_path),
        pending_payload_sha256=prior.payload_sha256,
        as_of_date="2026-08-12",
        reason="operator verified canonical and operational account parity",
    )
    identity_sha = str(binding["details"]["paper_account_identity_sha256"])
    assert binding["status"] == "legacy_terminal_bound"
    assert binding["details"]["assurance"] == "operator_bound_canonical_only_legacy_terminal_v1"
    assert (
        binding["details"]["operational_economic_reconstruction"]
        == "not_claimed_canonical_is_record_of_account"
    )
    assert journal.terminal(prior.payload_sha256).record_sha256 == terminal.record_sha256
    _assert_prior_pending_signals_resolved(
        daily, "2026-08-12", paper_account_identity_sha256=identity_sha
    )


def test_legacy_indeterminate_binding_does_not_clear_uncertainty(tmp_path) -> None:
    daily = _daily_config(tmp_path, "2026-08-12")
    prior = _pending(daily, "2026-08-10")
    PendingExecutionJournal(daily.execution_journal_path).append(
        pending_payload_sha256=prior.payload_sha256,
        signal_date=prior.signal_date,
        execution_date="2026-08-11",
        status="execution_indeterminate",
        details={"target_weights_sha256": prior.target_weights_sha256},
        recorded_at="2026-08-13T00:01:00+00:00",
    )
    continuous = _continuous_config(tmp_path)
    binding = bind_legacy_terminal_account(
        config=continuous,
        pending_payload_sha256=prior.payload_sha256,
        as_of_date="2026-08-12",
        reason="bind legacy terminal before uncertainty reconciliation",
    )
    identity_sha = str(binding["details"]["paper_account_identity_sha256"])
    with pytest.raises(PaperAccountStateRefused, match="unreconciled execution_indeterminate"):
        _assert_prior_pending_signals_resolved(
            daily, "2026-08-12", paper_account_identity_sha256=identity_sha
        )
    appended = reconcile_indeterminate_account(
        config=continuous,
        as_of_date="2026-08-12",
        reason="operator reconciled uncertain account economics",
    )
    assert len(appended) == 1
    assert appended[0]["status"] == "execution_reconciled"
    _assert_prior_pending_signals_resolved(
        daily, "2026-08-12", paper_account_identity_sha256=identity_sha
    )


def test_custom_canonical_ledger_derives_ledger_specific_journal(tmp_path) -> None:
    canonical = tmp_path / "paper" / "custom.jsonl"
    config = DailyPaperLoopConfig(
        as_of_date="2026-08-11",
        canonical_ledger_path=str(canonical),
        execution_journal_path=None,
    )
    resolved = Path(daily_loop._execution_journal_path(config))
    assert resolved == canonical.with_name("custom.execution_journal.jsonl")
    assert resolved != canonical.with_name("execution_journal.jsonl")


def test_indeterminate_account_prepass_outranks_legacy_binding_order(tmp_path) -> None:
    daily = _daily_config(tmp_path, "2026-08-12")
    journal = PendingExecutionJournal(daily.execution_journal_path)
    journal.append(
        pending_payload_sha256="a" * 64,
        signal_date="2026-08-08",
        execution_date="2026-08-09",
        status="execution_observed",
        details={"target_weights_sha256": "c" * 64},
    )
    journal.append(
        pending_payload_sha256="b" * 64,
        signal_date="2026-08-10",
        execution_date="2026-08-11",
        status="execution_indeterminate",
        details={"target_weights_sha256": "d" * 64},
    )
    identity_sha = "f" * 64

    with pytest.raises(PaperAccountStateRefused, match="unreconciled execution_indeterminate"):
        daily_loop._assert_execution_journal_resolved(
            journal,
            prefix_index=build_canonical_prefix_index(daily.canonical_ledger_path),
            paper_account_identity_sha256=identity_sha,
        )
    with pytest.raises(
        continuous_execution.ContinuousPaperExecutionBlocked,
        match="unreconciled execution_indeterminate",
    ):
        continuous_execution._assert_account_execution_state_resolved(
            journal,
            canonical_ledger_path=daily.canonical_ledger_path,
            paper_account_identity_sha256=identity_sha,
        )


def test_recovered_consistency_allows_operational_lifecycle_only_state() -> None:
    held = SimpleNamespace(total=100.0, is_flat=False)
    canonical = SimpleNamespace(
        portfolio=SimpleNamespace(
            positions={"600000.SH": held}, cash=99_000.0, initial_cash=100_000.0
        ),
        orders={},
        fills=[],
    )
    operational = SimpleNamespace(
        portfolio=SimpleNamespace(positions={}, cash=100_000.0, initial_cash=100_000.0),
        orders={},
        fills=[],
    )
    assert continuous_execution._operational_has_reconstructable_economics(operational) is False
    continuous_execution._assert_recovered_account_consistent(canonical, operational)


def test_recovered_consistency_rejects_conflicting_operational_economics() -> None:
    held = SimpleNamespace(total=100.0, is_flat=False)
    wrong = SimpleNamespace(total=50.0, is_flat=False)
    canonical = SimpleNamespace(
        portfolio=SimpleNamespace(
            positions={"600000.SH": held}, cash=99_000.0, initial_cash=100_000.0
        ),
        orders={},
        fills=[],
    )
    operational = SimpleNamespace(
        portfolio=SimpleNamespace(
            positions={"600000.SH": wrong}, cash=99_000.0, initial_cash=100_000.0
        ),
        orders={},
        fills=[],
    )
    assert continuous_execution._operational_has_reconstructable_economics(operational) is True
    with pytest.raises(
        continuous_execution.ContinuousPaperExecutionBlocked,
        match="position reconciliation failed",
    ):
        continuous_execution._assert_recovered_account_consistent(canonical, operational)


def test_legacy_binding_never_upgrades_to_state_only_operational_parity(tmp_path) -> None:
    daily = _daily_config(tmp_path, "2026-08-12")
    prior = _pending(daily, "2026-08-10")
    journal = PendingExecutionJournal(daily.execution_journal_path)
    journal.append(
        pending_payload_sha256=prior.payload_sha256,
        signal_date=prior.signal_date,
        execution_date="2026-08-11",
        status="execution_observed",
        details={"target_weights_sha256": prior.target_weights_sha256},
    )
    binding = bind_legacy_terminal_account(
        config=_continuous_config(tmp_path),
        pending_payload_sha256=prior.payload_sha256,
        as_of_date="2026-08-12",
        reason="bind canonical record without historical parity overclaim",
    )
    assert binding["details"]["assurance"] == "operator_bound_canonical_only_legacy_terminal_v1"
    assert (
        binding["details"]["operational_economic_reconstruction"]
        == "not_claimed_canonical_is_record_of_account"
    )


def test_custom_journal_path_canonicalizes_file_symlink_alias(tmp_path) -> None:
    real = tmp_path / "real_custom.jsonl"
    real.write_text("", encoding="utf-8")
    alias = tmp_path / "different_alias.jsonl"
    try:
        alias.symlink_to(real)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    real_cfg = DailyPaperLoopConfig(
        as_of_date="2026-08-11", canonical_ledger_path=str(real), execution_journal_path=None
    )
    alias_cfg = DailyPaperLoopConfig(
        as_of_date="2026-08-11", canonical_ledger_path=str(alias), execution_journal_path=None
    )
    assert daily_loop._execution_journal_path(real_cfg) == daily_loop._execution_journal_path(alias_cfg)


def test_legacy_binding_rejects_terminal_with_conflicting_explicit_identity(
    tmp_path,
) -> None:
    daily = _daily_config(tmp_path, "2026-08-12")
    prior = _pending(daily, "2026-08-10")
    PendingExecutionJournal(daily.execution_journal_path).append(
        pending_payload_sha256=prior.payload_sha256,
        signal_date=prior.signal_date,
        execution_date="2026-08-11",
        status="execution_observed",
        details={
            "target_weights_sha256": prior.target_weights_sha256,
            "paper_account_identity_sha256": "0" * 64,
        },
    )

    with pytest.raises(
        ContinuousPaperExecutionBlocked, match="conflicting paper-account identity"
    ):
        bind_legacy_terminal_account(
            config=_continuous_config(tmp_path),
            pending_payload_sha256=prior.payload_sha256,
            as_of_date="2026-08-12",
            reason="must not rebind another account",
        )


def test_other_account_staged_summary_is_preserved_and_refused(tmp_path) -> None:
    shared_reports = tmp_path / "reports"
    shared_pending = tmp_path / "pending"
    first = DailyPaperLoopConfig(
        as_of_date="2026-08-11",
        output_root=str(shared_reports),
        pending_signal_dir=str(shared_pending),
        canonical_ledger_path=str(tmp_path / "first-canonical.jsonl"),
        execution_journal_path=str(tmp_path / "first-journal.jsonl"),
    )
    second = DailyPaperLoopConfig(
        as_of_date="2026-08-11",
        output_root=str(shared_reports),
        pending_signal_dir=str(shared_pending),
        canonical_ledger_path=str(tmp_path / "second-canonical.jsonl"),
        execution_journal_path=str(tmp_path / "second-journal.jsonl"),
    )
    summary_path = daily_loop._write_daily_summary(
        shared_reports / "2026-08-11",
        {
            "daily_decision_commit_protocol": daily_loop.DAILY_SUMMARY_COMMIT_PROTOCOL,
            "paper_account_owner": daily_loop._paper_account_owner(first, "a" * 64),
        },
    )

    with pytest.raises(PaperAccountStateRefused, match="legacy/ambiguous daily summary"):
        _assert_current_signal_not_frozen(
            second,
            "2026-08-11",
            paper_account_identity_sha256="b" * 64,
        )
    assert summary_path.exists()


def test_other_account_staged_pending_is_preserved_and_refused(tmp_path) -> None:
    shared_pending = tmp_path / "pending"
    first = _daily_config(tmp_path / "first")
    second = _daily_config(tmp_path / "second")
    second = DailyPaperLoopConfig(
        **{
            **daily_loop.asdict(second),
            "pending_signal_dir": str(shared_pending),
            "output_root": str(tmp_path / "shared-reports"),
        }
    )
    first = DailyPaperLoopConfig(
        **{
            **daily_loop.asdict(first),
            "pending_signal_dir": str(shared_pending),
            "output_root": str(tmp_path / "shared-reports"),
        }
    )
    pending, pending_path = PendingPaperSignalStore(shared_pending).record(
        signal_date="2026-08-11",
        target_weights=pd.DataFrame(
            [{"trade_date": pd.Timestamp("2026-08-11"), "600000.SH": 0.25}]
        ),
        source_lineage={
            "daily_decision_commit_protocol": daily_loop.PENDING_COMMIT_PROTOCOL,
            "paper_account_identity_sha256": "a" * 64,
            "canonical_ledger_path": str(
                Path(first.canonical_ledger_path).resolve(strict=False)
            ),
            "execution_journal_path": str(
                Path(daily_loop._execution_journal_path(first)).resolve(strict=False)
            ),
        },
    )

    with pytest.raises(PaperAccountStateRefused, match="legacy/ambiguous pending"):
        _assert_current_signal_not_frozen(
            second,
            "2026-08-11",
            paper_account_identity_sha256="b" * 64,
        )
    assert pending_path.exists()
    assert PendingPaperSignalStore(shared_pending).read("2026-08-11") == pending
