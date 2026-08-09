"""Crash-safe outcome journal for pending paper signals.

The journal is intentionally conservative.  ``execution_started`` is appended
*before* the OMS/broker is touched.  If the process dies before a terminal
outcome is appended, restart returns ``execution_indeterminate`` and refuses to
auto-retry.  That sacrifices liveness to preserve at-most-once economic safety;
canonical/broker reconciliation may later resolve the incident explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Iterable


JOURNAL_SCHEMA_VERSION = "paper_pending_execution_journal_v1"
TERMINAL_OUTCOMES = frozenset({
    "execution_observed",
    "execution_blocked",
    "missed_execution_session",
    "execution_indeterminate",
})


class ExecutionJournalCorruption(RuntimeError):
    pass


@dataclass(frozen=True)
class PendingExecutionRecord:
    schema_version: str
    sequence: int
    pending_payload_sha256: str
    signal_date: str
    execution_date: str
    status: str
    details: dict[str, object]
    recorded_at: str
    previous_record_sha256: str
    record_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _digest(payload: dict[str, object]) -> str:
    material = dict(payload)
    material.pop("record_sha256", None)
    return sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


class PendingExecutionJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def records(self) -> list[PendingExecutionRecord]:
        if not self.path.exists():
            return []
        rows: list[PendingExecutionRecord] = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                rows.append(PendingExecutionRecord(**payload))
            except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
                raise ExecutionJournalCorruption(
                    f"cannot parse execution journal line {line_no}: {exc}"
                ) from exc
        return rows

    def verify(self) -> bool:
        previous = ""
        expected_sequence = 1
        for record in self.records():
            if record.schema_version != JOURNAL_SCHEMA_VERSION:
                return False
            if record.sequence != expected_sequence:
                return False
            if record.previous_record_sha256 != previous:
                return False
            if _digest(record.to_dict()) != record.record_sha256:
                return False
            previous = record.record_sha256
            expected_sequence += 1
        return True

    def history(self, pending_payload_sha256: str) -> list[PendingExecutionRecord]:
        return [
            record
            for record in self.records()
            if record.pending_payload_sha256 == pending_payload_sha256
        ]

    def terminal(self, pending_payload_sha256: str) -> PendingExecutionRecord | None:
        history = self.history(pending_payload_sha256)
        terminals = [record for record in history if record.status in TERMINAL_OUTCOMES]
        if len(terminals) > 1:
            raise ExecutionJournalCorruption(
                f"pending signal {pending_payload_sha256} has multiple terminal outcomes"
            )
        return terminals[0] if terminals else None

    def has_unresolved_start(self, pending_payload_sha256: str) -> bool:
        history = self.history(pending_payload_sha256)
        return any(record.status == "execution_started" for record in history) and not any(
            record.status in TERMINAL_OUTCOMES for record in history
        )

    def append(
        self,
        *,
        pending_payload_sha256: str,
        signal_date: str,
        execution_date: str,
        status: str,
        details: dict[str, object] | None = None,
        recorded_at: str | None = None,
    ) -> PendingExecutionRecord:
        if not self.verify():
            raise ExecutionJournalCorruption("pending execution journal hash chain is invalid")
        records = self.records()
        terminal = self.terminal(pending_payload_sha256)
        if terminal is not None:
            if status == terminal.status and dict(details or {}) == terminal.details:
                return terminal
            raise ExecutionJournalCorruption(
                f"pending signal {pending_payload_sha256} already has terminal status {terminal.status}"
            )
        if status == "execution_started" and self.has_unresolved_start(pending_payload_sha256):
            raise ExecutionJournalCorruption(
                f"pending signal {pending_payload_sha256} already has unresolved execution_started"
            )
        previous = records[-1].record_sha256 if records else ""
        payload: dict[str, object] = {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "sequence": len(records) + 1,
            "pending_payload_sha256": str(pending_payload_sha256),
            "signal_date": str(signal_date),
            "execution_date": str(execution_date),
            "status": str(status),
            "details": dict(details or {}),
            "recorded_at": recorded_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "previous_record_sha256": previous,
            "record_sha256": "",
        }
        payload["record_sha256"] = _digest(payload)
        record = PendingExecutionRecord(**payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        return record


def replay_terminal_status(records: Iterable[PendingExecutionRecord]) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in records:
        if record.status in TERMINAL_OUTCOMES:
            result[record.pending_payload_sha256] = record.status
    return result


__all__ = [
    "JOURNAL_SCHEMA_VERSION",
    "TERMINAL_OUTCOMES",
    "ExecutionJournalCorruption",
    "PendingExecutionRecord",
    "PendingExecutionJournal",
    "replay_terminal_status",
]
