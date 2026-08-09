from __future__ import annotations

import pandas as pd

from quantagent.paper.continuous_execution import (
    ContinuousPaperExecutionConfig,
    execute_pending_for_session,
)
from quantagent.paper.execution_journal import PendingExecutionJournal
from quantagent.paper.pending_signal import PendingPaperSignalStore
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
        initial_cash=100_000.0,
        max_participation_rate=0.05,
    )


def _record(tmp_path, signal_date: str, weight: float):
    return PendingPaperSignalStore(tmp_path / "pending").record(
        signal_date=signal_date,
        target_weights=_target(signal_date, weight),
        source_lineage={
            "model": "test-model",
            "target_weights_file_sha256": f"sha-{signal_date}-{weight}",
        },
        created_at=f"{signal_date}T07:00:00+00:00",
    )[0]


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
    assert result.shadow_acceptance_calendar_eligible is True

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


def test_unresolved_started_attempt_becomes_indeterminate_and_is_not_retried(tmp_path) -> None:
    config = _config(tmp_path)
    pending = _record(tmp_path, FRIDAY, 0.50)
    journal = PendingExecutionJournal(config.execution_journal_path)
    journal.append(
        pending_payload_sha256=pending.payload_sha256,
        signal_date=FRIDAY,
        execution_date=MONDAY,
        status="execution_started",
        details={"simulated_crash": True},
    )

    result = execute_pending_for_session(
        MONDAY,
        _market(),
        config=config,
        authoritative_sessions=SESSIONS,
    )
    assert len(result) == 1
    assert result[0].status == "execution_indeterminate"
    assert result[0].order_count == 0
    assert result[0].fill_count == 0
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
