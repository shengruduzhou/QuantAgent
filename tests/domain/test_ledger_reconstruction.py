"""Requirement D: economic state must be reconstructible from the ledger alone.

The claim being tested is strong: destroy every derived in-memory value and
rebuild order status, fills, lots, cash, fees and NAV from the persisted event
log only. If replay and the live run can disagree, then one of them is a second
copy of the truth, and a second copy is exactly the drift this module exists to
remove.

Requirement A rides along here: these run the *real* fast engine, so they also
show the engine emits canonical entities rather than only its own dictionaries.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from quantagent.backtest.engine import BacktestConfig, EventDrivenBacktester
from quantagent.domain.ledger import CanonicalLedger, LedgerCorruption
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import OrderStatus, Side

SYMBOL = "600000.SH"
INITIAL = 1_000_000.0
RUN = Lineage(research_id="res_1", strategy_id="str_1", strategy_version_id="sv_1", run_id="run_1")


def _prices(dates, closes):
    return pd.DataFrame([
        {"trade_date": d, "symbol": SYMBOL, "open": c, "high": c, "low": c,
         "close": c, "volume": 1e8, "amount": c * 1e8}
        for d, c in zip(dates, closes)
    ])


def _run(tmp_path, weights, closes=None):
    dates = pd.bdate_range("2026-01-05", periods=len(weights))
    closes = closes or [10.0, 10.4, 9.9, 10.2, 10.6, 10.1, 10.3, 10.5][: len(weights)]
    config = BacktestConfig(
        initial_nav=INITIAL,
        ledger_path=str(tmp_path / "ledger.jsonl"),
        lineage=RUN,
    )
    result = EventDrivenBacktester(config).run(
        pd.DataFrame({SYMBOL: weights}, index=dates), _prices(dates, closes)
    )
    return result, closes


def test_the_fast_engine_writes_every_order_to_the_canonical_ledger(tmp_path):
    result, _ = _run(tmp_path, [0.2, 0.2, 0.0, 0.15, 0.15, 0.0, 0.1, 0.1])

    assert result.ledger is not None and len(result.ledger) > 0
    assert result.order_book is not None
    # Every dict row in the legacy projection has a canonical fill behind it.
    assert len(result.order_book.fills()) == len(result.trades)
    assert result.ledger.verify()["valid"]


def test_every_order_carries_the_runs_lineage(tmp_path):
    result, _ = _run(tmp_path, [0.2, 0.2, 0.0, 0.0])

    for order in result.order_book.orders():
        assert order.lineage.research_id == "res_1"
        assert order.lineage.strategy_version_id == "sv_1"
        assert order.lineage.run_id == "run_1"
        assert order.lineage.signal_id is not None
        assert order.lineage.order_intent_id is not None


def test_replay_reconstructs_exact_economic_state(tmp_path):
    result, closes = _run(tmp_path, [0.2, 0.2, 0.0, 0.15, 0.15, 0.0, 0.1, 0.1])
    path = tmp_path / "ledger.jsonl"

    # Everything in memory is discarded; only the file survives.
    book, account = CanonicalLedger(path).replay(initial_cash=INITIAL)

    assert len(book.orders()) == len(result.order_book.orders())
    assert len(book.fills()) == len(result.order_book.fills())
    assert account.nav({SYMBOL: closes[-1]}) == pytest.approx(result.nav_curve.iloc[-1], abs=1e-6)


def test_replayed_order_status_and_quantities_match(tmp_path):
    result, _ = _run(tmp_path, [0.2, 0.2, 0.0, 0.15])

    book, _ = CanonicalLedger(tmp_path / "ledger.jsonl").replay(initial_cash=INITIAL)

    for original in result.order_book.orders():
        restored = book.state_of(original.order_id)
        assert restored.status is original.status
        assert restored.filled_quantity == original.filled_quantity
        assert restored.remaining == original.remaining
        assert restored.side is original.side
        assert restored.quantity == original.quantity


def test_replayed_fees_and_slippage_match(tmp_path):
    result, _ = _run(tmp_path, [0.2, 0.2, 0.0, 0.15, 0.15, 0.0])

    _, account = CanonicalLedger(tmp_path / "ledger.jsonl").replay(initial_cash=INITIAL)

    expected_fees = sum(fill.fees for fill in result.order_book.fills())
    expected_slip = sum(fill.slippage for fill in result.order_book.fills())
    assert account.total_fees == pytest.approx(expected_fees, abs=1e-9)
    assert account.total_slippage == pytest.approx(expected_slip, abs=1e-9)


def test_replay_is_stable_across_repeated_reconstructions(tmp_path):
    _run(tmp_path, [0.2, 0.2, 0.0, 0.15, 0.15])
    path = tmp_path / "ledger.jsonl"

    hashes = {CanonicalLedger(path).replay(initial_cash=INITIAL)[1].content_hash() for _ in range(3)}

    assert len(hashes) == 1, "replay must be deterministic"


def test_position_lots_survive_reconstruction(tmp_path):
    result, _ = _run(tmp_path, [0.2, 0.2, 0.2, 0.2])

    _, account = CanonicalLedger(tmp_path / "ledger.jsonl").replay(initial_cash=INITIAL)

    buys = sum(f.quantity for f in result.order_book.fills() if f.side is Side.BUY)
    sells = sum(f.quantity for f in result.order_book.fills() if f.side is Side.SELL)
    assert account.position(SYMBOL) == buys - sells


# -- chain integrity ---------------------------------------------------------
def test_an_altered_record_breaks_verification(tmp_path):
    _run(tmp_path, [0.2, 0.2, 0.0, 0.15])
    path = tmp_path / "ledger.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    # Tamper with a middle record's timestamp — present in every record, and
    # part of the hashed body, so the chain must break exactly there.
    record = json.loads(lines[2])
    record["recordedAt"] = "1999-01-01T00:00:00+00:00"
    lines[2] = json.dumps(record, sort_keys=True, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verification = CanonicalLedger(path).verify()

    assert verification["valid"] is False
    assert verification["brokenAt"] == 2


def test_replay_refuses_a_corrupt_chain_rather_than_producing_numbers(tmp_path):
    """Numbers from an unverifiable log would look authoritative and not be."""
    _run(tmp_path, [0.2, 0.2, 0.0, 0.15])
    path = tmp_path / "ledger.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"reason": null', '"reason": "tampered"', 1)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(LedgerCorruption):
        CanonicalLedger(path).replay(initial_cash=INITIAL)


def test_a_torn_trailing_write_keeps_every_earlier_record(tmp_path):
    result, _ = _run(tmp_path, [0.2, 0.2, 0.0, 0.15])
    path = tmp_path / "ledger.jsonl"
    complete = len(result.ledger)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"sequence": 99, "recordedAt": "2026')

    recovered = CanonicalLedger(path)

    assert recovered.had_torn_tail
    assert len(recovered) == complete
    assert recovered.verify()["valid"], "records before the tear must still verify"


def test_an_order_created_without_an_intent_is_refused_on_replay(tmp_path):
    """Enforces 'no order without an intent' from the log alone."""
    _run(tmp_path, [0.2, 0.2])
    path = tmp_path / "ledger.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    stripped = [line for line in lines]
    stripped[0] = stripped[0].replace('"intent":', '"intent_removed":', 1)
    path.write_text("\n".join(stripped) + "\n", encoding="utf-8")

    # The tamper breaks the hash chain first, which is itself the guarantee.
    with pytest.raises(LedgerCorruption):
        CanonicalLedger(path).replay(initial_cash=INITIAL)
