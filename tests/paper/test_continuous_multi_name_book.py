"""The continuous paper loop must be able to hold more than one name.

`PaperBroker._validate` used to call `RiskEngine.check_order` without `prices`,
so the risk engine valued the whole book with the order's own price only. Once
the book held symbol A, any order for symbol B raised `UnpriceablePosition(A)`
out of the venue — after `execution_started` was journalled and after the OMS
had written RISK_APPROVED + SUBMITTED. Every top-k target (k > 1) crashed after
its first fill and left a phantom SUBMITTED order that froze the account.
Every earlier fixture traded exactly one symbol.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from quantagent.domain.ledger import CanonicalLedger
from quantagent.paper.account_identity import ensure_paper_account_identity
from quantagent.paper.canonical_receipt import canonical_snapshot
from quantagent.paper.continuous_execution import (
    ContinuousPaperExecutionConfig,
    execute_pending_for_session,
)
from quantagent.paper.execution_journal import DAILY_DECISION_STATUS, PendingExecutionJournal
from quantagent.paper.pending_signal import PENDING_COMMIT_PROTOCOL, PendingPaperSignalStore

FRIDAY, MONDAY, TUESDAY = "2026-08-07", "2026-08-10", "2026-08-11"
SESSIONS = [FRIDAY, MONDAY, TUESDAY]
BANKS = ["600000.SH", "600036.SH", "601398.SH", "601288.SH"]
OTHER = "600519.SH"


def market(symbols=(*BANKS, OTHER)) -> pd.DataFrame:
    rows = []
    for date in SESSIONS:
        for symbol in symbols:
            rows.append({
                "trade_date": date, "symbol": symbol, "open": 10.0, "high": 10.0,
                "low": 10.0, "close": 10.0, "volume": 10_000_000.0,
                "amount": 100_000_000.0, "is_suspended": False, "is_st": False,
                "price_adjustment": "raw", "execution_eligible": True,
            })
    return pd.DataFrame(rows)


def sector_map(tmp_path) -> str:
    path = tmp_path / "sector_map.csv"
    pd.DataFrame(
        {"symbol": [*BANKS, OTHER], "industry": ["bank"] * len(BANKS) + ["liquor"]}
    ).to_csv(path, index=False)
    return str(path)


def config(tmp_path, **overrides) -> ContinuousPaperExecutionConfig:
    values = dict(
        pending_signal_dir=str(tmp_path / "pending"),
        execution_journal_path=str(tmp_path / "execution.jsonl"),
        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),
        operational_ledger_path=str(tmp_path / "operational.jsonl"),
        idempotency_path=str(tmp_path / "idempotency.jsonl"),
        account_identity_path=str(tmp_path / "account_identity.json"),
        initial_cash=1_000_000.0,
        max_participation_rate=0.05,
    )
    values.update(overrides)
    return ContinuousPaperExecutionConfig(**values)


def record_target(tmp_path, signal_date: str, weights: dict[str, float], *,
                  initial_cash: float = 1_000_000.0):
    identity = ensure_paper_account_identity(
        canonical_ledger_path=tmp_path / "canonical.jsonl",
        portfolio_id="v7-paper", initial_cash=initial_cash,
        identity_path=tmp_path / "account_identity.json",
    )
    records, head = canonical_snapshot(tmp_path / "canonical.jsonl")
    frame = pd.DataFrame(
        {"trade_date": [pd.Timestamp(signal_date)], **{k: [v] for k, v in weights.items()}}
    )
    pending = PendingPaperSignalStore(tmp_path / "pending").record(
        signal_date=signal_date, target_weights=frame,
        source_lineage={
            "model": "multi-name-test",
            "target_weights_file_sha256": f"sha-{signal_date}",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_ledger_head_hash": head,
            "canonical_ledger_records": str(records),
            "daily_decision_commit_protocol": PENDING_COMMIT_PROTOCOL,
        },
        created_at=f"{signal_date}T07:00:00+00:00",
    )[0]
    summary = tmp_path / f"summary-{signal_date}.json"
    summary.write_text('{"committed":true}\n', encoding="utf-8")
    PendingExecutionJournal(tmp_path / "execution.jsonl").append(
        pending_payload_sha256=pending.payload_sha256, signal_date=signal_date,
        execution_date=signal_date, status=DAILY_DECISION_STATUS,
        details={
            "decision_kind": "target",
            "paper_account_identity_sha256": identity.payload_sha256,
            "canonical_account_state_sha256": "1" * 64,
            "canonical_records": records, "canonical_head": head,
            "assurance": "canonical_account_daily_decision_freeze_v1",
            "commit_protocol": PENDING_COMMIT_PROTOCOL,
            "target_weights_sha256": pending.target_weights_sha256,
            "daily_summary_path": str(summary.resolve()),
            "daily_summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
            "daily_summary_commit_protocol": "daily_summary_bound_daily_decision_v1",
        },
    )
    return pending


def canonical_states(tmp_path) -> list[str]:
    book = CanonicalLedger(str(tmp_path / "canonical.jsonl")).replay_book()
    return sorted(order.status.value for order in book.orders())


def test_two_name_target_executes_under_production_default_limits(tmp_path) -> None:
    """Two 5% names are far inside every default limit; both must fill."""
    record_target(tmp_path, FRIDAY, {BANKS[0]: 0.05, OTHER: 0.05})

    result = execute_pending_for_session(
        MONDAY, market(), config=config(tmp_path, sector_map_path=sector_map(tmp_path)),
        authoritative_sessions=SESSIONS,
    )

    assert result[0].fill_count == 2
    assert canonical_states(tmp_path) == ["FILLED", "FILLED"]


def test_a_fourth_bank_breaching_the_sector_cap_is_refused_at_the_venue(tmp_path) -> None:
    """Four banks at 9% project a 36% bank sector against the 30% default."""
    record_target(tmp_path, FRIDAY, {symbol: 0.09 for symbol in BANKS})

    result = execute_pending_for_session(
        MONDAY, market(), config=config(tmp_path, sector_map_path=sector_map(tmp_path)),
        authoritative_sessions=SESSIONS,
    )

    assert result[0].fill_count == 3
    book = CanonicalLedger(str(tmp_path / "canonical.jsonl")).replay_book()
    rejected = [o for o in book.orders() if o.status.value == "REJECTED"]
    assert len(rejected) == 1
    assert "industry_weight" in (rejected[0].reason or "")


def test_without_a_sector_map_a_buy_is_refused_as_unmeasured(tmp_path) -> None:
    """No map, default 30% industry limit: unmeasured is not within limit."""
    record_target(tmp_path, FRIDAY, {BANKS[0]: 0.05})

    result = execute_pending_for_session(
        MONDAY, market(), config=config(tmp_path), authoritative_sessions=SESSIONS,
    )

    assert result[0].fill_count == 0
    assert result[0].status == "execution_blocked"
    book = CanonicalLedger(str(tmp_path / "canonical.jsonl")).replay_book()
    assert "industry_unmeasured" in (book.orders()[0].reason or "")
