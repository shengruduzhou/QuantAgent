"""Round-22 regression for A-04: the governance hash chain must not fork.

``AuditLog.append`` used to be an unsynchronised read-then-write: it read the
chain tail off disk, built an entry from it, and appended without a file lock
and without ``fsync``. Two processes that read the same tail produce two entries
carrying the *same* ``sequence`` and the *same* ``prev_hash``, so the chain
forks and every record on the losing branch becomes unreachable from genesis --
silently, because neither writer sees an error.

The round-21 audit measured this with four processes writing forty entries
each: 160 lines on disk, 76 distinct sequences, and exactly **one** entry
reachable by following ``prev_hash`` from ``GENESIS_HASH``.

This test is that measurement turned into a gate. It is deliberately
round-synchronised (a ``Barrier`` before every single append) so the pre-fix
collision is not left to scheduler luck, and it uses fixed process/entry counts
so the expected numbers are exact rather than statistical.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
from collections import Counter
from pathlib import Path

import pytest

from quantagent.governance.audit import GENESIS_HASH, AuditLog

PROCESS_COUNT = 4
ENTRIES_PER_PROCESS = 40
EXPECTED_TOTAL = PROCESS_COUNT * ENTRIES_PER_PROCESS
SUBJECT = "round22 concurrent governance decisions"

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="uses the POSIX fork start method to observe cross-process appends",
)


def _append_rounds(path: str, tag: str, rounds: int, barrier) -> None:
    """Child process body: append ``rounds`` entries, one per barrier release."""

    log = AuditLog(path)
    for index in range(rounds):
        barrier.wait(timeout=60)
        log.append(
            kind="ENVELOPE",
            actor=tag,
            subject=SUBJECT,
            payload={"writer": tag, "index": index},
        )


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reachable_from_genesis(rows: list[dict]) -> int:
    """Follow ``prev_hash`` from genesis and count how far the walk gets.

    Written out longhand rather than calling ``AuditLog.verify`` so the test
    measures the same quantity against both the pre-fix and post-fix
    implementation, and so a failure reports the orphan count directly.
    """

    by_prev: dict[str, list[dict]] = {}
    for row in rows:
        by_prev.setdefault(row["prev_hash"], []).append(row)

    reached = 0
    cursor = GENESIS_HASH
    seen: set[str] = set()
    while True:
        candidates = by_prev.get(cursor, [])
        if len(candidates) != 1:
            # No candidate ends the walk; more than one is a fork, and a fork has
            # no single successor that a chain verifier could follow.
            return reached
        row = candidates[0]
        if row["entry_hash"] in seen:
            return reached
        seen.add(row["entry_hash"])
        reached += 1
        cursor = row["entry_hash"]


def _diagnose(rows: list[dict]) -> str:
    sequences = Counter(row["sequence"] for row in rows)
    prev_hashes = Counter(row["prev_hash"] for row in rows)
    reachable = _reachable_from_genesis(rows)
    return (
        f"lines={len(rows)} "
        f"distinct_sequences={len(sequences)} "
        f"duplicate_sequences={sum(1 for n in sequences.values() if n > 1)} "
        f"forked_prev_hash={sum(1 for n in prev_hashes.values() if n > 1)} "
        f"reachable_from_genesis={reachable} "
        f"orphaned={len(rows) - reachable}"
    )


def test_concurrent_appends_leave_one_unbroken_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(PROCESS_COUNT)

    workers = [
        ctx.Process(
            target=_append_rounds,
            args=(str(path), f"agent{index}", ENTRIES_PER_PROCESS, barrier),
        )
        for index in range(PROCESS_COUNT)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=180)

    still_running = [worker for worker in workers if worker.is_alive()]
    for worker in still_running:
        worker.terminate()
    assert not still_running, "audit writers deadlocked instead of serialising"
    assert [worker.exitcode for worker in workers] == [0] * PROCESS_COUNT

    rows = _read_rows(path)
    diagnosis = _diagnose(rows)
    sequences = Counter(row["sequence"] for row in rows)
    prev_hashes = Counter(row["prev_hash"] for row in rows)

    assert len(rows) == EXPECTED_TOTAL, diagnosis
    assert max(sequences.values()) == 1, diagnosis
    assert max(prev_hashes.values()) == 1, diagnosis
    assert sorted(sequences) == list(range(EXPECTED_TOTAL)), diagnosis
    assert _reachable_from_genesis(rows) == EXPECTED_TOTAL, diagnosis

    # Every writer's work survived; no branch was silently dropped.
    assert Counter(row["actor"] for row in rows) == Counter(
        {f"agent{index}": ENTRIES_PER_PROCESS for index in range(PROCESS_COUNT)}
    ), diagnosis

    # And the log's own verifier agrees with the longhand walk.
    result = AuditLog(path).verify()
    assert result["valid"] is True, f"{result} / {diagnosis}"
    assert result["checked"] == EXPECTED_TOTAL, f"{result} / {diagnosis}"


def test_interleaved_short_lived_writers_extend_one_chain(tmp_path: Path) -> None:
    """Each writer is a fresh process that exits without unwinding.

    ``os._exit`` skips ``atexit`` hooks and interpreter shutdown, so nothing gets
    to tidy up on the writer's behalf: whatever the next process finds on disk is
    exactly what ``append`` left there. This is the sequential companion to the
    barrier test -- it catches a fix that only holds while one process lives.
    """

    path = tmp_path / "audit.jsonl"
    ctx = multiprocessing.get_context("fork")

    def _append_then_die(index: int) -> None:
        AuditLog(str(path)).append(
            kind="DECISION",
            actor="short-lived-writer",
            subject=SUBJECT,
            payload={"index": index},
        )
        os._exit(9)

    for index in range(12):
        worker = ctx.Process(target=_append_then_die, args=(index,))
        worker.start()
        worker.join(timeout=60)
        assert worker.exitcode == 9, f"writer {index} exited {worker.exitcode}"

    rows = _read_rows(path)
    assert len(rows) == 12, _diagnose(rows)
    assert [row["sequence"] for row in rows] == list(range(12)), _diagnose(rows)
    assert AuditLog(path).verify()["valid"] is True, _diagnose(rows)
