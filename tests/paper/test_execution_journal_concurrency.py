from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from quantagent.paper.execution_journal import (
    ExecutionJournalCorruption,
    PendingExecutionJournal,
)


def test_concurrent_instances_allocate_one_linear_hash_chain(tmp_path) -> None:
    path = tmp_path / "execution.jsonl"
    count = 24

    def append(index: int) -> str:
        journal = PendingExecutionJournal(path)
        record = journal.append(
            pending_payload_sha256=f"payload-{index:02d}",
            signal_date="2026-08-07",
            execution_date="2026-08-10",
            status="execution_started",
            details={"worker": index},
        )
        return record.record_sha256

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(append, range(count)))

    journal = PendingExecutionJournal(path)
    records = journal.records()
    assert len(hashes) == count
    assert len(set(hashes)) == count
    assert journal.verify()
    assert [record.sequence for record in records] == list(range(1, count + 1))
    assert len({record.record_sha256 for record in records}) == count


def test_duplicate_json_keys_are_not_accepted_as_evidence(tmp_path) -> None:
    path = tmp_path / "execution.jsonl"
    path.write_text(
        '{"schema_version":"paper_pending_execution_journal_v1",'
        '"sequence":1,"sequence":2}\n',
        encoding="utf-8",
    )
    with pytest.raises(ExecutionJournalCorruption, match="duplicate JSON key"):
        PendingExecutionJournal(path).records()


def test_unknown_status_is_rejected_before_it_can_enter_hash_chain(tmp_path) -> None:
    journal = PendingExecutionJournal(tmp_path / "execution.jsonl")
    with pytest.raises(ValueError, match="unsupported execution journal status"):
        journal.append(
            pending_payload_sha256="payload",
            signal_date="2026-08-07",
            execution_date="2026-08-10",
            status="looks_good_to_me",
        )
