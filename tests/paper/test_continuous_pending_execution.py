from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from quantagent.paper.account_identity import ensure_paper_account_identity
from quantagent.paper.canonical_receipt import canonical_snapshot
from quantagent.paper.broker import PaperBroker
from quantagent.domain.ledger import CanonicalLedger
import quantagent.paper.execution_journal as execution_journal_module
from quantagent.paper.continuous_execution import (
    ContinuousPaperExecutionBlocked,
    ContinuousPaperExecutionConfig,
    execute_pending_for_session,
)
from quantagent.paper.execution_journal import DAILY_DECISION_STATUS, PendingExecutionJournal
from quantagent.paper.pending_signal import PENDING_COMMIT_PROTOCOL, PendingPaperSignalStore
from quantagent.paper.recovery import recover_from_canonical


FRIDAY = "2026-08-07"
MONDAY = "2026-08-10"
TUESDAY = "2026-08-11"
SESSIONS = [FRIDAY, MONDAY, TUESDAY]
SYMBOL = "600000.SH"


def _market() -> pd.DataFrame:
    rows = []
    for date in SESSIONS:
        rows.append(
            {
                "trade_date": date,
                "symbol": SYMBOL,
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 10_000_000.0,
                "amount": 100_000_000.0,
                "is_suspended": False,
                "is_st": False,
                "price_adjustment": "raw",
                "execution_eligible": True,
            }
        )
    return pd.DataFrame(rows)


def _target(signal_date: str, weight: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [pd.Timestamp(signal_date)],
            SYMBOL: [weight],
        }
    )


def _config(tmp_path) -> ContinuousPaperExecutionConfig:
    return ContinuousPaperExecutionConfig(
        pending_signal_dir=str(tmp_path / "pending"),
        execution_journal_path=str(tmp_path / "execution.jsonl"),
        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),
        operational_ledger_path=str(tmp_path / "operational.jsonl"),
        idempotency_path=str(tmp_path / "idempotency.jsonl"),
        account_identity_path=str(tmp_path / "account_identity.json"),
        initial_cash=100_000.0,
        max_participation_rate=0.05,
    )


def _identity(tmp_path):
    return ensure_paper_account_identity(
        canonical_ledger_path=tmp_path / "canonical.jsonl",
        portfolio_id="v7-paper",
        initial_cash=100_000.0,
        identity_path=tmp_path / "account_identity.json",
    )


def _record(tmp_path, signal_date: str, weight: float):
    identity = _identity(tmp_path)
    canonical_records, canonical_head = canonical_snapshot(tmp_path / "canonical.jsonl")
    pending = PendingPaperSignalStore(tmp_path / "pending").record(
        signal_date=signal_date,
        target_weights=_target(signal_date, weight),
        source_lineage={
            "model": "test-model",
            "target_weights_file_sha256": f"sha-{signal_date}-{weight}",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_ledger_head_hash": canonical_head,
            "canonical_ledger_records": str(canonical_records),
            "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,
        },
        created_at=f"{signal_date}T07:00:00+00:00",
    )[0]
    summary_path = tmp_path / f"summary-{signal_date}.json"
    summary_path.write_text('{"committed":true}\n', encoding="utf-8")
    summary_sha = __import__("hashlib").sha256(summary_path.read_bytes()).hexdigest()
    PendingExecutionJournal(tmp_path / "execution.jsonl").append(
        pending_payload_sha256=pending.payload_sha256,
        signal_date=signal_date,
        execution_date=signal_date,
        status=DAILY_DECISION_STATUS,
        details={
            "decision_kind": "target",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_records": canonical_records,
            "canonical_head": canonical_head,
            "assurance": "canonical_account_daily_decision_freeze_v1",
            "commit_protocol": PENDING_COMMIT_PROTOCOL,
            "target_weights_sha256": pending.target_weights_sha256,
            "daily_summary_path": str(summary_path.resolve()),
            "daily_summary_sha256": summary_sha,
            "daily_summary_commit_protocol": "daily_summary_bound_daily_decision_v1",
        },
    )
    return pending


def test_friday_signal_executes_monday_once_on_persistent_account(tmp_path) -> None:
    config = _config(tmp_path)
    pending = _record(tmp_path, FRIDAY, 0.50)

    first = execute_pending_for_session(
        MONDAY,
        _market(),
        config=config,
        authoritative_sessions=SESSIONS,
    )
    assert len(first) == 1
    result = first[0]
    assert result.signal_date == FRIDAY
    assert result.execution_date == MONDAY
    assert result.status == "execution_observed"
    assert result.order_count == 1
    assert result.fill_count == 1
    assert result.calendar_assurance == "caller_supplied_session_set_unverified"
    assert result.shadow_acceptance_calendar_eligible is False

    repeated = execute_pending_for_session(
        MONDAY,
        _market(),
        config=config,
        authoritative_sessions=SESSIONS,
    )
    assert repeated == []
    terminal = PendingExecutionJournal(config.execution_journal_path).terminal(
        pending.payload_sha256
    )
    assert terminal is not None
    assert terminal.status == "execution_observed"
    assert terminal.details["session_closed"] is True
    assert terminal.details["production_pretrade_risk_certified"] is False
    assert terminal.details["paper_account_identity_sha256"] == _identity(tmp_path).payload_sha256


def test_next_day_liquidation_uses_recovered_position_and_t_plus_one_sellability(tmp_path) -> None:
    config = _config(tmp_path)
    _record(tmp_path, FRIDAY, 0.50)
    buy = execute_pending_for_session(
        MONDAY,
        _market(),
        config=config,
        authoritative_sessions=SESSIONS,
    )
    assert buy[0].fill_count == 1

    after_buy = recover_from_canonical(
        config.canonical_ledger_path,
        portfolio_id=config.portfolio_id,
        initial_cash=config.initial_cash,
        as_of_session=MONDAY,
    )
    assert after_buy.portfolio.positions[SYMBOL].sellable == 0

    _record(tmp_path, MONDAY, 0.0)
    sell = execute_pending_for_session(
        TUESDAY,
        _market(),
        config=config,
        authoritative_sessions=SESSIONS,
    )
    assert len(sell) == 1
    assert sell[0].execution_date == TUESDAY
    assert sell[0].order_count == 1
    assert sell[0].fill_count == 1

    after_sell = recover_from_canonical(
        config.canonical_ledger_path,
        portfolio_id=config.portfolio_id,
        initial_cash=config.initial_cash,
        as_of_session=TUESDAY,
    )
    position = after_sell.portfolio.positions.get(SYMBOL)
    assert position is None or position.total == 0


def test_missed_next_session_is_not_retroactively_filled(tmp_path) -> None:
    config = _config(tmp_path)
    pending = _record(tmp_path, FRIDAY, 0.50)
    result = execute_pending_for_session(
        TUESDAY,
        _market(),
        config=config,
        authoritative_sessions=SESSIONS,
    )
    assert len(result) == 1
    assert result[0].status == "missed_execution_session"
    assert result[0].execution_date == MONDAY
    assert result[0].fill_count == 0
    assert not (tmp_path / "canonical.jsonl").exists()
    assert PendingExecutionJournal(config.execution_journal_path).terminal(
        pending.payload_sha256
    ).status == "missed_execution_session"


def test_unresolved_started_attempt_blocks_retry_until_explicit_reconciliation(tmp_path) -> None:
    config = _config(tmp_path)
    pending = _record(tmp_path, FRIDAY, 0.50)
    journal = PendingExecutionJournal(config.execution_journal_path)
    journal.append(
        pending_payload_sha256=pending.payload_sha256,
        signal_date=FRIDAY,
        execution_date=MONDAY,
        status="execution_started",
        details={
            "simulated_crash": True,
            "paper_account_identity_sha256": _identity(tmp_path).payload_sha256,
        },
    )

    with pytest.raises(ContinuousPaperExecutionBlocked, match="unresolved execution_started"):
        execute_pending_for_session(
            MONDAY,
            _market(),
            config=config,
            authoritative_sessions=SESSIONS,
        )
    assert journal.terminal(pending.payload_sha256) is None
    assert journal.has_unresolved_start(pending.payload_sha256)
    assert not (tmp_path / "canonical.jsonl").exists()


def test_observed_panel_calendar_can_execute_operationally_but_not_certify_shadow_calendar(tmp_path) -> None:
    config = _config(tmp_path)
    _record(tmp_path, FRIDAY, 0.50)
    result = execute_pending_for_session(
        MONDAY,
        _market(),
        config=config,
        authoritative_sessions=None,
    )
    assert len(result) == 1
    assert result[0].status == "execution_observed"
    assert result[0].calendar_assurance == "observed_market_panel_only"
    assert result[0].shadow_acceptance_calendar_eligible is False


def test_configured_execution_clock_is_recorded_in_terminal_evidence(tmp_path) -> None:
    config = replace(_config(tmp_path), execution_clock="14:58:30+08:00")
    pending = _record(tmp_path, FRIDAY, 0.50)
    result = execute_pending_for_session(
        MONDAY,
        _market(),
        config=config,
        authoritative_sessions=SESSIONS,
    )
    assert result[0].status == "execution_observed"
    terminal = PendingExecutionJournal(config.execution_journal_path).terminal(
        pending.payload_sha256
    )
    assert terminal is not None
    assert terminal.details["execution_clock"] == "14:58:30+08:00"


def test_close_session_failure_never_publishes_false_terminal(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path)
    pending = _record(tmp_path, FRIDAY, 0.50)

    def fail_close(self, trade_date):
        del self, trade_date
        raise RuntimeError("simulated close-session crash")

    monkeypatch.setattr(PaperBroker, "close_session", fail_close)
    with pytest.raises(RuntimeError, match="simulated close-session crash"):
        execute_pending_for_session(
            MONDAY,
            _market(),
            config=config,
            authoritative_sessions=SESSIONS,
        )

    journal = PendingExecutionJournal(config.execution_journal_path)
    assert journal.terminal(pending.payload_sha256) is None
    assert journal.has_unresolved_start(pending.payload_sha256)


def test_missing_execution_critical_flag_fails_before_economic_attempt(tmp_path) -> None:
    config = _config(tmp_path)
    pending = _record(tmp_path, FRIDAY, 0.50)
    market = _market().drop(columns=["is_st"])

    with pytest.raises(ContinuousPaperExecutionBlocked, match="missing columns:.*is_st"):
        execute_pending_for_session(
            MONDAY,
            market,
            config=config,
            authoritative_sessions=SESSIONS,
        )

    assert not (tmp_path / "canonical.jsonl").exists()
    assert PendingExecutionJournal(config.execution_journal_path).terminal(
        pending.payload_sha256
    ) is None


def test_unknown_execution_critical_flag_fails_closed(tmp_path) -> None:
    config = _config(tmp_path)
    pending = _record(tmp_path, FRIDAY, 0.50)
    market = _market()
    market.loc[market["trade_date"] == MONDAY, "is_suspended"] = None

    with pytest.raises(ContinuousPaperExecutionBlocked, match="missing explicit is_suspended"):
        execute_pending_for_session(
            MONDAY,
            market,
            config=config,
            authoritative_sessions=SESSIONS,
        )

    assert not (tmp_path / "canonical.jsonl").exists()
    assert PendingExecutionJournal(config.execution_journal_path).terminal(
        pending.payload_sha256
    ) is None


def test_string_false_flags_are_not_truthy_python_strings(tmp_path) -> None:
    config = _config(tmp_path)
    _record(tmp_path, FRIDAY, 0.50)
    market = _market()
    market["is_suspended"] = "false"
    market["is_st"] = "0"

    result = execute_pending_for_session(
        MONDAY,
        market,
        config=config,
        authoritative_sessions=SESSIONS,
    )
    assert result[0].status == "execution_observed"
    assert result[0].fill_count == 1


def test_adjusted_execution_prices_fail_before_economic_attempt(tmp_path) -> None:
    config = _config(tmp_path)
    pending = _record(tmp_path, FRIDAY, 0.50)
    market = _market()
    market["price_adjustment"] = "qfq"
    market["execution_eligible"] = False

    with pytest.raises(ContinuousPaperExecutionBlocked, match="raw/unadjusted"):
        execute_pending_for_session(
            MONDAY,
            market,
            config=config,
            authoritative_sessions=SESSIONS,
        )

    assert not (tmp_path / "canonical.jsonl").exists()
    assert PendingExecutionJournal(config.execution_journal_path).terminal(
        pending.payload_sha256
    ) is None


def test_missing_price_provenance_fails_closed(tmp_path) -> None:
    config = _config(tmp_path)
    _record(tmp_path, FRIDAY, 0.50)
    market = _market().drop(columns=["price_adjustment", "execution_eligible"])

    with pytest.raises(ContinuousPaperExecutionBlocked, match="price provenance failed"):
        execute_pending_for_session(
            MONDAY,
            market,
            config=config,
            authoritative_sessions=SESSIONS,
        )


def test_worker_initial_cash_mismatch_fails_before_execution(tmp_path) -> None:
    good = _config(tmp_path)
    pending = _record(tmp_path, FRIDAY, 0.50)
    bad = replace(good, initial_cash=200_000.0)

    with pytest.raises(ContinuousPaperExecutionBlocked, match="initial_cash mismatch"):
        execute_pending_for_session(
            MONDAY,
            _market(),
            config=bad,
            authoritative_sessions=SESSIONS,
        )
    assert not (tmp_path / "canonical.jsonl").exists()
    assert PendingExecutionJournal(good.execution_journal_path).terminal(
        pending.payload_sha256
    ) is None


def test_worker_portfolio_id_mismatch_fails_before_execution(tmp_path) -> None:
    good = _config(tmp_path)
    _record(tmp_path, FRIDAY, 0.50)
    bad = replace(good, portfolio_id="other-paper-book")
    with pytest.raises(ContinuousPaperExecutionBlocked, match="portfolio_id mismatch"):
        execute_pending_for_session(
            MONDAY,
            _market(),
            config=bad,
            authoritative_sessions=SESSIONS,
        )


def test_pending_signal_from_other_account_identity_is_rejected(tmp_path) -> None:
    config = _config(tmp_path)
    _identity(tmp_path)
    pending = PendingPaperSignalStore(tmp_path / "pending").record(
        signal_date=FRIDAY,
        target_weights=_target(FRIDAY, 0.50),
        source_lineage={
            "model": "test-model",
            "paper_account_identity_sha256": "0" * 64,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_ledger_head_hash": "0" * 64,
            "canonical_ledger_records": "0",
            "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,
        },
        created_at=f"{FRIDAY}T07:00:00+00:00",
    )[0]
    PendingExecutionJournal(config.execution_journal_path).append(
        pending_payload_sha256=pending.payload_sha256,
        signal_date=FRIDAY,
        execution_date=FRIDAY,
        status=DAILY_DECISION_STATUS,
        details={
            "decision_kind": "target",
            "paper_account_identity_sha256": "0" * 64,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_records": 0,
            "canonical_head": "0" * 64,
            "assurance": "canonical_account_daily_decision_freeze_v1",
            "commit_protocol": PENDING_COMMIT_PROTOCOL,
            "target_weights_sha256": pending.target_weights_sha256,
        },
    )
    with pytest.raises(ContinuousPaperExecutionBlocked, match="paper_account_identity_sha256"):
        execute_pending_for_session(
            MONDAY,
            _market(),
            config=config,
            authoritative_sessions=SESSIONS,
        )
    assert PendingExecutionJournal(config.execution_journal_path).terminal(
        pending.payload_sha256
    ) is None


def test_uncommitted_staged_pending_is_never_executed(tmp_path) -> None:
    config = _config(tmp_path)
    identity = _identity(tmp_path)
    pending = PendingPaperSignalStore(tmp_path / "pending").record(
        signal_date=FRIDAY,
        target_weights=_target(FRIDAY, 0.50),
        source_lineage={
            "model": "test-model",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_ledger_head_hash": "0" * 64,
            "canonical_ledger_records": "0",
            "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,
        },
        created_at=f"{FRIDAY}T07:00:00+00:00",
    )[0]
    with pytest.raises(ContinuousPaperExecutionBlocked, match="staged but not committed"):
        execute_pending_for_session(
            MONDAY, _market(), config=config, authoritative_sessions=SESSIONS
        )
    assert PendingExecutionJournal(config.execution_journal_path).terminal(
        pending.payload_sha256
    ) is None
    assert not (tmp_path / "canonical.jsonl").exists()


def test_mismatched_daily_commit_cannot_execute_pending(tmp_path) -> None:
    config = _config(tmp_path)
    identity = _identity(tmp_path)
    pending = PendingPaperSignalStore(tmp_path / "pending").record(
        signal_date=FRIDAY,
        target_weights=_target(FRIDAY, 0.50),
        source_lineage={
            "model": "test-model",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_ledger_head_hash": "0" * 64,
            "canonical_ledger_records": "0",
            "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,
        },
        created_at=f"{FRIDAY}T07:00:00+00:00",
    )[0]
    PendingExecutionJournal(config.execution_journal_path).append(
        pending_payload_sha256="f" * 64,
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
        },
    )
    with pytest.raises(ContinuousPaperExecutionBlocked, match="pending_payload_sha256"):
        execute_pending_for_session(
            MONDAY, _market(), config=config, authoritative_sessions=SESSIONS
        )


def test_committed_pending_refuses_missing_bound_summary(tmp_path) -> None:
    config = _config(tmp_path)
    pending = _record(tmp_path, FRIDAY, 0.50)
    journal = PendingExecutionJournal(config.execution_journal_path)
    decision = journal.daily_decision(FRIDAY)
    assert decision is not None
    # Retrofit summary binding for the legacy helper's marker, then remove it.
    # The helper is updated below to bind a real file on all new records.
    summary_path = Path(decision.details["daily_summary_path"])
    summary_path.unlink()
    with pytest.raises(ContinuousPaperExecutionBlocked, match="summary evidence is missing"):
        execute_pending_for_session(
            MONDAY, _market(), config=config, authoritative_sessions=SESSIONS
        )
    assert journal.terminal(pending.payload_sha256) is None


def test_committed_pending_rejects_post_freeze_canonical_drift_before_execution(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    pending = _record(tmp_path, FRIDAY, 0.50)
    CanonicalLedger(config.canonical_ledger_path).append(None, trade_date=FRIDAY)

    with pytest.raises(ContinuousPaperExecutionBlocked, match="changed after daily decision freeze"):
        execute_pending_for_session(
            MONDAY, _market(), config=config, authoritative_sessions=SESSIONS
        )

    assert PendingExecutionJournal(config.execution_journal_path).terminal(
        pending.payload_sha256
    ) is None


def test_hash_valid_unknown_journal_status_freezes_before_duplicate_execution(
    tmp_path,
) -> None:
    config = _config(tmp_path)
    pending = _record(tmp_path, FRIDAY, 0.50)
    journal = PendingExecutionJournal(config.execution_journal_path)
    journal.append(
        pending_payload_sha256=pending.payload_sha256,
        signal_date=FRIDAY,
        execution_date=MONDAY,
        status="execution_observed",
        details={"target_weights_sha256": pending.target_weights_sha256},
    )
    rows = [json.loads(line) for line in Path(config.execution_journal_path).read_text(encoding="utf-8").splitlines()]
    rows[-1]["status"] = "execution_observed_v2"
    rows[-1]["record_sha256"] = execution_journal_module._digest(rows[-1])
    Path(config.execution_journal_path).write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    assert journal.verify() is False
    with pytest.raises(ContinuousPaperExecutionBlocked, match="journal verification failed"):
        execute_pending_for_session(
            MONDAY, _market(), config=config, authoritative_sessions=SESSIONS
        )
    assert not Path(config.canonical_ledger_path).exists()


def test_invalid_authoritative_session_evidence_fails_closed(tmp_path) -> None:
    config = _config(tmp_path)
    pending = _record(tmp_path, FRIDAY, 0.50)

    with pytest.raises(ContinuousPaperExecutionBlocked, match="invalid/non-finite dates"):
        execute_pending_for_session(
            TUESDAY,
            _market(),
            config=config,
            authoritative_sessions=[FRIDAY, "not-a-date", TUESDAY],
        )

    assert PendingExecutionJournal(config.execution_journal_path).terminal(
        pending.payload_sha256
    ) is None
    assert not Path(config.canonical_ledger_path).exists()


def test_torn_idempotency_tail_blocks_before_execution_start(tmp_path) -> None:
    config = _config(tmp_path)
    pending = _record(tmp_path, FRIDAY, 0.50)
    Path(config.idempotency_path).write_text('{"schemaVersion":', encoding="utf-8")

    with pytest.raises(ContinuousPaperExecutionBlocked, match="idempotency evidence is corrupt"):
        execute_pending_for_session(
            MONDAY, _market(), config=config, authoritative_sessions=SESSIONS
        )

    history = PendingExecutionJournal(config.execution_journal_path).history(
        pending.payload_sha256
    )
    assert all(record.status != "execution_started" for record in history)
    assert not Path(config.canonical_ledger_path).exists()
