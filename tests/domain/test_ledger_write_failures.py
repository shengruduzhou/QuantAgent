"""What happens to the chain when the disk refuses (M1-11, DEF-017).

A durable append has three steps that can each fail independently: the open, the
write, the fsync. Only the first of those failing means "nothing happened". The
other two leave the file in a state the writer cannot determine — an `fsync` that
raises `EIO` has already handed the bytes to the OS — and the in-memory head is
then no longer known to match the file's.

That is the whole defect: appending on top of an unknown head computes
`previousHash` from a stale predecessor, which breaks the chain permanently and
surfaces only at read time. Measured before the fix: one failed fsync left 2
records on disk against 1 in memory, and the next append made the file
unreplayable.

The harness in `scripts/module1_fault_injection.py` runs these against the real
paper service. These are the fast, in-process versions that pin the ledger's own
behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
from unittest import mock

import pytest

from quantagent.domain.ledger import (
    CanonicalLedger,
    LedgerWriteUnavailable,
    mirror_open,
)
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import OrderBook, OrderIntent, Side, Signal

SYMBOL = "600000.SH"
SESSION = "2026-08-04"
RUN = Lineage(research_id="r", strategy_version_id="sv", run_id="run_write_fail")


def _intent(sequence: int) -> OrderIntent:
    signal = Signal.create(
        symbol=SYMBOL, trade_date=f"{SESSION}-{sequence}", score=1.0, lineage=RUN
    )
    return OrderIntent.create(
        symbol=SYMBOL, side=Side.BUY, quantity=100 * sequence, trade_date=SESSION,
        lineage=signal.lineage, limit_price=10.05,
    )


def _append_order(book: OrderBook, ledger: CanonicalLedger, sequence: int):
    return mirror_open(book, ledger, _intent(sequence), trade_date=SESSION)


@pytest.fixture
def chain(tmp_path) -> tuple[OrderBook, CanonicalLedger, Path]:
    path = tmp_path / "chain.jsonl"
    book, ledger = OrderBook(), CanonicalLedger(path)
    _append_order(book, ledger, 1)
    assert len(ledger) == 1
    return book, ledger, path


def _failing_fsync(fd: int) -> None:
    raise OSError(5, "simulated EIO")


def test_a_failed_fsync_surfaces_to_the_caller(chain):
    book, ledger, _ = chain
    with mock.patch("quantagent.domain.ledger.os.fsync", _failing_fsync):
        with pytest.raises(OSError, match="simulated EIO"):
            _append_order(book, ledger, 2)


def test_a_failed_fsync_latches_the_ledger_closed(chain):
    """The core of DEF-017: no second append on top of an unknown tail."""
    book, ledger, path = chain
    with mock.patch("quantagent.domain.ledger.os.fsync", _failing_fsync):
        with pytest.raises(OSError):
            _append_order(book, ledger, 2)

    assert ledger.write_failure is not None
    with pytest.raises(LedgerWriteUnavailable, match="stopped accepting writes"):
        _append_order(book, ledger, 3)


def test_the_latch_holds_even_though_the_bytes_did_land(chain):
    """The failure mode that made this a defect rather than an inconvenience.

    `fsync` raising means the line was already written and flushed, so the file is
    *ahead* of memory. Before the latch, the next append chained from the stale
    in-memory head and the file stopped replaying.
    """
    book, ledger, path = chain
    with mock.patch("quantagent.domain.ledger.os.fsync", _failing_fsync):
        with pytest.raises(OSError):
            _append_order(book, ledger, 2)

    on_disk = len(path.read_text(encoding="utf-8").splitlines())
    assert on_disk == 2, "the write did land; that is what makes the head uncertain"
    assert len(ledger) == 1, "in-memory state correctly did not adopt the record"
    with pytest.raises(LedgerWriteUnavailable):
        _append_order(book, ledger, 3)


def test_a_restart_replays_the_file_that_is_actually_there(chain):
    """Recovery is a restart, not a resynchronisation."""
    book, ledger, path = chain
    with mock.patch("quantagent.domain.ledger.os.fsync", _failing_fsync):
        with pytest.raises(OSError):
            _append_order(book, ledger, 2)

    restarted = CanonicalLedger(path)

    assert restarted.verify()["valid"], "the chain on disk must still verify"
    assert restarted.write_failure is None, "a fresh instance is not latched"
    assert len(restarted) == 2
    restarted_book = restarted.replay_book()
    assert len(restarted_book.orders()) == 2
    # And it can be written to again, because its head is the file's head.
    _append_order(restarted_book, restarted, 3)
    assert restarted.verify()["valid"]


def test_verify_reports_a_write_failure_even_when_the_chain_is_valid(chain):
    """A chain can verify perfectly and still belong to a process that cannot write."""
    book, ledger, _ = chain
    with mock.patch("quantagent.domain.ledger.os.fsync", _failing_fsync):
        with pytest.raises(OSError):
            _append_order(book, ledger, 2)

    verification = ledger.verify()
    assert verification["valid"] is True
    assert verification["writeFailure"] is not None


def test_reads_still_work_after_a_write_failure(chain):
    """Latching writes must not blind the operator diagnosing the failure."""
    book, ledger, _ = chain
    with mock.patch("quantagent.domain.ledger.os.fsync", _failing_fsync):
        with pytest.raises(OSError):
            _append_order(book, ledger, 2)

    assert len(ledger.read()) == 1
    assert len(ledger.events()) == 1
    assert ledger.replay_book().orders()


def test_a_read_only_ledger_file_fails_loudly(chain):
    book, ledger, path = chain
    original = path.stat().st_mode
    path.chmod(stat.S_IRUSR)
    try:
        with pytest.raises(OSError):
            _append_order(book, ledger, 2)
    finally:
        path.chmod(original)

    assert ledger.write_failure is not None
    assert CanonicalLedger(path).verify()["valid"], "the committed record survived"


def test_a_full_disk_fails_loudly_and_latches(chain):
    book, ledger, path = chain
    real_open = Path.open

    def full_disk(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        if self == path:
            handle.write = mock.Mock(side_effect=OSError(28, "No space left on device"))
        return handle

    with mock.patch.object(Path, "open", full_disk):
        with pytest.raises(OSError, match="No space left"):
            _append_order(book, ledger, 2)

    assert ledger.write_failure is not None
    with pytest.raises(LedgerWriteUnavailable):
        _append_order(book, ledger, 3)
    assert CanonicalLedger(path).verify()["valid"]


def test_an_in_memory_ledger_is_unaffected_by_a_disk_fault(tmp_path):
    """No path, no durable write, nothing to fail — and no spurious latch."""
    book, ledger = OrderBook(), CanonicalLedger()
    _append_order(book, ledger, 1)
    with mock.patch("quantagent.domain.ledger.os.fsync", _failing_fsync):
        _append_order(book, ledger, 2)
    assert len(ledger) == 2
    assert ledger.write_failure is None
