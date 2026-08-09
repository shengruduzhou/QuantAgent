"""Crash-safe outcome journal for pending paper signals.

The journal is intentionally conservative. ``execution_started`` is appended
*before* the OMS/broker is touched. If the process dies before a terminal
outcome is appended, restart returns ``execution_indeterminate`` and refuses to
auto-retry. That sacrifices liveness to preserve at-most-once economic safety;
canonical/broker reconciliation may later resolve the incident explicitly.

The hash-chain decision and append are protected by a sibling file lock on both
POSIX and Windows. Without that critical section, two consumers can read the
same tail, allocate the same sequence/previous hash, and both append internally
valid records whose combined journal is corrupt.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import RLock
from typing import Iterable

try:  # POSIX research/CI hosts
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None

try:  # QMT/MiniQMT Windows hosts
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - Unix
    _msvcrt = None


JOURNAL_SCHEMA_VERSION = "paper_pending_execution_journal_v1"
TERMINAL_OUTCOMES = frozenset(
    {
        "execution_observed",
        "execution_blocked",
        "missed_execution_session",
        "execution_indeterminate",
    }
)
_ALLOWED_STATUSES = TERMINAL_OUTCOMES | {"execution_started"}


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


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExecutionJournalCorruption(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


class PendingExecutionJournal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._thread_lock = RLock()

    def records(self) -> list[PendingExecutionRecord]:
        if not self.path.exists():
            return []
        rows: list[PendingExecutionRecord] = []
        for line_no, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line, object_pairs_hook=_strict_object)
                rows.append(PendingExecutionRecord(**payload))
            except ExecutionJournalCorruption as exc:
                raise ExecutionJournalCorruption(
                    f"cannot parse execution journal line {line_no}: {exc}"
                ) from exc
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

    @contextmanager
    def _exclusive_file_lock(self):
        """Serialize the read-tail -> sequence/hash -> durable append decision."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+b") as handle:
            if _fcntl is not None:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
                return

            if _msvcrt is not None:  # pragma: no cover - Windows host/CI
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                    os.fsync(handle.fileno())
                handle.seek(0)
                _msvcrt.locking(handle.fileno(), _msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    _msvcrt.locking(handle.fileno(), _msvcrt.LK_UNLCK, 1)
                return

            raise RuntimeError("no supported cross-process file locking primitive")

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
        if status not in _ALLOWED_STATUSES:
            raise ValueError(f"unsupported execution journal status {status!r}")
        if not str(pending_payload_sha256).strip():
            raise ValueError("pending_payload_sha256 must be non-empty")

        with self._thread_lock, self._exclusive_file_lock():
            if not self.verify():
                raise ExecutionJournalCorruption(
                    "pending execution journal hash chain is invalid"
                )
            records = self.records()
            history = [
                record
                for record in records
                if record.pending_payload_sha256 == pending_payload_sha256
            ]
            terminals = [
                record for record in history if record.status in TERMINAL_OUTCOMES
            ]
            if len(terminals) > 1:
                raise ExecutionJournalCorruption(
                    f"pending signal {pending_payload_sha256} has multiple terminal outcomes"
                )
            terminal = terminals[0] if terminals else None
            if terminal is not None:
                if status == terminal.status and dict(details or {}) == terminal.details:
                    return terminal
                raise ExecutionJournalCorruption(
                    f"pending signal {pending_payload_sha256} already has terminal "
                    f"status {terminal.status}"
                )
            if status == "execution_started" and any(
                record.status == "execution_started" for record in history
            ):
                raise ExecutionJournalCorruption(
                    f"pending signal {pending_payload_sha256} already has unresolved "
                    "execution_started"
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
                "recorded_at": recorded_at
                or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "previous_record_sha256": previous,
                "record_sha256": "",
            }
            payload["record_sha256"] = _digest(payload)
            record = PendingExecutionRecord(**payload)
            line = (
                json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            return record


def replay_terminal_status(
    records: Iterable[PendingExecutionRecord],
) -> dict[str, str]:
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
