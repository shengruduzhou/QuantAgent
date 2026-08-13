from __future__ import annotations

from contextlib import contextmanager

import pandas as pd
import pytest

import quantagent.paper.account_lock as account_lock

import quantagent.paper.continuous_execution as continuous_execution
from quantagent.paper.continuous_execution import ContinuousPaperExecutionConfig


def _config(tmp_path) -> ContinuousPaperExecutionConfig:
    return ContinuousPaperExecutionConfig(
        pending_signal_dir=str(tmp_path / "pending"),
        execution_journal_path=str(tmp_path / "execution.jsonl"),
        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),
        operational_ledger_path=str(tmp_path / "operational.jsonl"),
        idempotency_path=str(tmp_path / "idempotency.json"),
        account_identity_path=str(tmp_path / "identity.json"),
    )


def test_execute_pending_for_session_owns_account_lock(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    state = {"held": False, "enters": 0}

    @contextmanager
    def fake_lock(path, **_kwargs):
        assert str(path) == config.canonical_ledger_path
        state["enters"] += 1
        state["held"] = True
        try:
            yield path
        finally:
            state["held"] = False

    def fake_locked(as_of_date, market_panel, *, config, authoritative_sessions):
        assert state["held"] is True
        assert as_of_date == "2026-08-11"
        assert market_panel.empty
        assert authoritative_sessions is None
        return []

    monkeypatch.setattr(continuous_execution, "paper_account_lock", fake_lock)
    monkeypatch.setattr(
        continuous_execution,
        "_execute_pending_for_session_locked",
        fake_locked,
    )

    assert continuous_execution.execute_pending_for_session(
        "2026-08-11",
        pd.DataFrame(),
        config=config,
        authoritative_sessions=None,
    ) == []
    assert state == {"held": False, "enters": 1}


def test_reconcile_indeterminate_account_owns_account_lock(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    state = {"held": False, "enters": 0}

    @contextmanager
    def fake_lock(path, **_kwargs):
        assert str(path) == config.canonical_ledger_path
        state["enters"] += 1
        state["held"] = True
        try:
            yield path
        finally:
            state["held"] = False

    def fake_locked(*, config, as_of_date, reason):
        assert state["held"] is True
        assert as_of_date == "2026-08-11"
        assert reason == "operator reconciliation"
        return [{"status": "execution_reconciled"}]

    monkeypatch.setattr(continuous_execution, "paper_account_lock", fake_lock)
    monkeypatch.setattr(
        continuous_execution,
        "_reconcile_indeterminate_account_locked",
        fake_locked,
    )

    assert continuous_execution.reconcile_indeterminate_account(
        config=config,
        as_of_date="2026-08-11",
        reason="operator reconciliation",
    ) == [{"status": "execution_reconciled"}]
    assert state == {"held": False, "enters": 1}


def test_account_lock_path_canonicalizes_symlink_alias(tmp_path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    ledger = real_dir / "canonical.jsonl"
    ledger.write_text("", encoding="utf-8")
    alias_dir = tmp_path / "alias"
    try:
        alias_dir.symlink_to(real_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    assert account_lock.paper_account_lock_path(ledger) == account_lock.paper_account_lock_path(
        alias_dir / "canonical.jsonl"
    )


def test_account_lock_path_collapses_hardlink_aliases(tmp_path) -> None:
    ledger = tmp_path / "canonical.jsonl"
    ledger.write_text("", encoding="utf-8")
    alias = tmp_path / "canonical-hardlink.jsonl"
    try:
        alias.hardlink_to(ledger)
    except OSError as exc:
        pytest.skip(f"hardlink creation unavailable: {exc}")

    assert ledger.stat().st_ino == alias.stat().st_ino
    assert account_lock.paper_account_lock_path(ledger) == account_lock.paper_account_lock_path(alias)
    with account_lock.paper_account_lock(ledger):
        with pytest.raises(account_lock.PaperAccountLockTimeout):
            with account_lock.paper_account_lock(alias, timeout_seconds=0.0):
                pytest.fail("hardlink alias acquired a second account lock")
