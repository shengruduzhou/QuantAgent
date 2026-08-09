from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quantagent.domain.idempotency import IdempotencyStore
from quantagent.domain.ledger import CanonicalLedger
from quantagent.paper.pending_execution import (
    AmbiguousPendingExecution,
    ExecutionReceiptCorruption,
    PendingExecutionConfig,
    PendingExecutionReceiptStore,
    PendingExecutionRefused,
    _signal_claim_key,
    execute_due_pending_signals,
    verify_execution_receipt,
)
from quantagent.paper.pending_signal import PendingPaperSignalStore
from quantagent.paper.recovery import recover_from_canonical


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "pending": tmp_path / "pending",
        "receipts": tmp_path / "receipts",
        "canonical": tmp_path / "canonical.jsonl",
        "operational": tmp_path / "operational.jsonl",
        "idem": tmp_path / "idempotency.jsonl",
    }


def _signal(store: PendingPaperSignalStore, date: str, weight: float):
    return store.record(
        signal_date=date,
        target_weights=pd.DataFrame({"trade_date": [date], "600000.SH": [weight]}),
        source_lineage={"model": "test", "dataset": "pit-test"},
        created_at=f"{date}T15:00:00+08:00",
    )[0]


def _market(*dates: str, is_st: bool = False, suspended_on: str | None = None) -> pd.DataFrame:
    closes = {
        "2026-07-31": 10.00,
        "2026-08-03": 10.00,
        "2026-08-04": 10.20,
        "2026-08-05": 10.10,
    }
    rows = []
    for date in dates:
        rows.append(
            {
                "trade_date": pd.Timestamp(date),
                "symbol": "600000.SH",
                "open": closes[date],
                "high": closes[date],
                "low": closes[date],
                "close": closes[date],
                "volume": 1_000_000,
                "amount": closes[date] * 1_000_000,
                "is_st": is_st,
                "is_suspended": date == suspended_on,
            }
        )
    return pd.DataFrame(rows)


def _execute(tmp_path: Path, as_of: str, market: pd.DataFrame):
    p = _paths(tmp_path)
    return execute_due_pending_signals(
        as_of_date=as_of,
        market_panel=market,
        pending_store=PendingPaperSignalStore(p["pending"]),
        receipt_store=PendingExecutionReceiptStore(p["receipts"]),
        canonical_ledger_path=p["canonical"],
        operational_ledger_path=p["operational"],
        idempotency_path=p["idem"],
        config=PendingExecutionConfig(initial_cash=1_000_000.0),
    )


def test_signal_stays_pending_without_a_later_observed_session(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    _signal(PendingPaperSignalStore(p["pending"]), "2026-07-31", 0.50)

    result = _execute(tmp_path, "2026-07-31", _market("2026-07-31"))

    assert result.executed_receipts == ()
    assert result.still_pending == ("2026-07-31",)
    assert len(CanonicalLedger(p["canonical"])) == 0
    assert len(IdempotencyStore(p["idem"])) == 0


def test_friday_signal_executes_once_on_monday_observation(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    signal = _signal(PendingPaperSignalStore(p["pending"]), "2026-07-31", 0.50)
    market = _market("2026-07-31", "2026-08-03")

    first = _execute(tmp_path, "2026-08-03", market)
    second = _execute(tmp_path, "2026-08-03", market)

    assert first.executed_receipts == ("2026-07-31",)
    assert second.skipped_existing_receipts == ("2026-07-31",)
    receipt = PendingExecutionReceiptStore(p["receipts"]).read("2026-07-31")
    assert receipt is not None
    assert receipt.execution_date == "2026-08-03"
    assert receipt.outcome == "execution_observed"
    ledger = CanonicalLedger(p["canonical"])
    verify_execution_receipt(receipt, signal=signal, ledger=ledger)
    records_after_first = receipt.canonical_after_records
    assert len(ledger) == records_after_first
    assert receipt.positions_after.get("600000.SH", 0) > 0


def test_restart_recovery_makes_monday_buy_sellable_on_tuesday(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    store = PendingPaperSignalStore(p["pending"])
    _signal(store, "2026-07-31", 0.50)
    _execute(tmp_path, "2026-08-03", _market("2026-07-31", "2026-08-03"))

    # New process/day: Monday close generates a zero target, Tuesday observes it.
    _signal(store, "2026-08-03", 0.0)
    result = _execute(
        tmp_path,
        "2026-08-04",
        _market("2026-07-31", "2026-08-03", "2026-08-04"),
    )

    assert result.executed_receipts == ("2026-08-03",)
    receipt = PendingExecutionReceiptStore(p["receipts"]).read("2026-08-03")
    assert receipt is not None
    assert receipt.execution_date == "2026-08-04"
    assert receipt.positions_after.get("600000.SH", 0) == 0


def test_recovery_as_of_next_session_derives_t_plus_one_from_canonical_lot_date(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    _signal(PendingPaperSignalStore(p["pending"]), "2026-07-31", 0.50)
    _execute(tmp_path, "2026-08-03", _market("2026-07-31", "2026-08-03"))

    monday = recover_from_canonical(
        str(p["canonical"]),
        portfolio_id="v7-paper-shadow",
        initial_cash=1_000_000.0,
        as_of_trade_date="2026-08-03",
    )
    tuesday = recover_from_canonical(
        str(p["canonical"]),
        portfolio_id="v7-paper-shadow",
        initial_cash=1_000_000.0,
        as_of_trade_date="2026-08-04",
    )

    assert monday.portfolio.position("600000.SH").sellable == 0
    assert tuesday.portfolio.position("600000.SH").sellable > 0


def test_missing_execution_flags_refuses_before_whole_signal_claim(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    _signal(PendingPaperSignalStore(p["pending"]), "2026-07-31", 0.50)
    market = _market("2026-07-31", "2026-08-03").drop(columns=["is_st"])

    with pytest.raises(PendingExecutionRefused, match="is_st"):
        _execute(tmp_path, "2026-08-03", market)

    assert len(IdempotencyStore(p["idem"])) == 0
    assert len(CanonicalLedger(p["canonical"])) == 0


def test_unresolved_durable_claim_never_resubmits(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    store = PendingPaperSignalStore(p["pending"])
    signal = _signal(store, "2026-07-31", 0.50)
    execution_date = "2026-08-03"
    key = _signal_claim_key(signal, execution_date)
    IdempotencyStore(p["idem"]).claim(
        key,
        payload={"signal_date": signal.signal_date, "execution_date": execution_date},
    )

    with pytest.raises(AmbiguousPendingExecution, match="already claimed"):
        _execute(tmp_path, execution_date, _market("2026-07-31", execution_date))

    assert len(CanonicalLedger(p["canonical"])) == 0
    assert not PendingExecutionReceiptStore(p["receipts"]).path_for(signal.signal_date).exists()


def test_suspended_next_session_consumes_signal_as_blocked_once(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    _signal(PendingPaperSignalStore(p["pending"]), "2026-07-31", 0.50)
    market = _market("2026-07-31", "2026-08-03", suspended_on="2026-08-03")

    result = _execute(tmp_path, "2026-08-03", market)
    receipt = PendingExecutionReceiptStore(p["receipts"]).read("2026-07-31")

    assert result.executed_receipts == ("2026-07-31",)
    assert receipt is not None
    assert receipt.outcome == "execution_blocked"
    assert receipt.order_results[0]["state"] == "REJECTED"
    assert "suspended" in str(receipt.order_results[0]["reject_reason"])


def test_st_new_buy_is_blocked_until_pit_and_daily_cap_governance_exist(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    _signal(PendingPaperSignalStore(p["pending"]), "2026-07-31", 0.50)

    _execute(
        tmp_path,
        "2026-08-03",
        _market("2026-07-31", "2026-08-03", is_st=True),
    )
    receipt = PendingExecutionReceiptStore(p["receipts"]).read("2026-07-31")

    assert receipt is not None
    assert receipt.outcome == "execution_blocked"
    assert "ST buy blocked" in str(receipt.order_results[0]["reject_reason"])


def test_tampered_receipt_fails_verification(tmp_path: Path) -> None:
    p = _paths(tmp_path)
    signal = _signal(PendingPaperSignalStore(p["pending"]), "2026-07-31", 0.50)
    market = _market("2026-07-31", "2026-08-03")
    _execute(tmp_path, "2026-08-03", market)
    receipt_path = PendingExecutionReceiptStore(p["receipts"]).path_for(signal.signal_date)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["cash_after"] = float(payload["cash_after"]) + 1.0
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExecutionReceiptCorruption, match="digest mismatch"):
        _execute(tmp_path, "2026-08-03", market)
