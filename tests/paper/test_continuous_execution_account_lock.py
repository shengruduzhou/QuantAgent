from __future__ import annotations

from contextlib import contextmanager

import pandas as pd

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