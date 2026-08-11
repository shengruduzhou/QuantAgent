from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.ingestion.daily_evidence_job import DailyEvidenceJob
from quantagent.paper.account_identity import (
    PaperAccountIdentityStore,
    ensure_paper_account_identity,
)
from quantagent.paper.continuous_execution import (
    ContinuousPaperExecutionConfig,
    execute_pending_for_session,
)
from quantagent.paper.daily_loop import DailyPaperLoopConfig, run_once
from quantagent.paper.execution_journal import PendingExecutionJournal
from quantagent.paper.pending_signal import PendingPaperSignalStore


FRIDAY = "2026-08-07"
MONDAY = "2026-08-10"
SYMBOL = "600000.SH"


def test_same_second_accounts_with_same_genesis_have_distinct_identity_hashes(tmp_path) -> None:
    created_at = "2026-08-11T00:00:00+00:00"
    first = PaperAccountIdentityStore(tmp_path / "a" / "account_identity.json").ensure(
        portfolio_id="v7-paper",
        initial_cash=100_000.0,
        created_at=created_at,
    )
    second = PaperAccountIdentityStore(tmp_path / "b" / "account_identity.json").ensure(
        portfolio_id="v7-paper",
        initial_cash=100_000.0,
        created_at=created_at,
    )

    assert first.account_instance_id != second.account_instance_id
    assert first.payload_sha256 != second.payload_sha256


def test_daily_loop_custom_canonical_ledger_derives_sibling_identity(tmp_path, monkeypatch) -> None:
    canonical = tmp_path / "isolated" / "canonical_ledger.jsonl"
    config = DailyPaperLoopConfig(
        as_of_date=FRIDAY,
        canonical_ledger_path=str(canonical),
        account_identity_path=None,
        initial_cash=100_000.0,
    )

    def stop_after_identity(self, job_config):
        del self, job_config
        raise RuntimeError("stop-after-identity")

    monkeypatch.setattr(DailyEvidenceJob, "run", stop_after_identity)
    with pytest.raises(RuntimeError, match="stop-after-identity"):
        run_once(config)

    identity_path = canonical.with_name("account_identity.json")
    assert identity_path.exists()
    identity = PaperAccountIdentityStore(identity_path).read()
    assert identity is not None
    assert identity.portfolio_id == config.portfolio_id
    assert identity.initial_cash == pytest.approx(config.initial_cash)


def test_terminal_legacy_signal_without_identity_lineage_does_not_block_new_runtime(tmp_path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    identity_path = tmp_path / "account_identity.json"
    ensure_paper_account_identity(
        canonical_ledger_path=canonical,
        portfolio_id="v7-paper",
        initial_cash=100_000.0,
        identity_path=identity_path,
    )

    pending = PendingPaperSignalStore(tmp_path / "pending").record(
        signal_date=FRIDAY,
        target_weights=pd.DataFrame(
            {
                "trade_date": [pd.Timestamp(FRIDAY)],
                SYMBOL: [0.50],
            }
        ),
        source_lineage={"legacy_model": "pre-account-identity"},
        created_at=f"{FRIDAY}T07:00:00+00:00",
    )[0]

    journal_path = tmp_path / "execution.jsonl"
    PendingExecutionJournal(journal_path).append(
        pending_payload_sha256=pending.payload_sha256,
        signal_date=FRIDAY,
        execution_date=MONDAY,
        status="execution_observed",
        details={"legacy_terminal_evidence": True},
        recorded_at=f"{MONDAY}T07:00:00+00:00",
    )

    config = ContinuousPaperExecutionConfig(
        pending_signal_dir=str(tmp_path / "pending"),
        execution_journal_path=str(journal_path),
        canonical_ledger_path=str(canonical),
        operational_ledger_path=str(tmp_path / "operational.jsonl"),
        idempotency_path=str(tmp_path / "idempotency.jsonl"),
        account_identity_path=str(identity_path),
        portfolio_id="v7-paper",
        initial_cash=100_000.0,
    )
    market = pd.DataFrame({"trade_date": [MONDAY]})

    result = execute_pending_for_session(
        MONDAY,
        market,
        config=config,
        authoritative_sessions=[FRIDAY, MONDAY],
    )
    assert result == []
