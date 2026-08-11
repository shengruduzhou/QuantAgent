from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from quantagent.paper.broker import PaperBroker
from quantagent.paper.continuous_execution import (
    ContinuousPaperExecutionBlocked,
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
    # A caller-supplied list can schedule shadow execution but cannot certify
    # exchange-calendar correctness without source/version/digest provenance.
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
