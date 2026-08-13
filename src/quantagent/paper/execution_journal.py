"""Crash-safe outcome journal for pending paper signals.

The journal is intentionally conservative. ``execution_started`` is appended
*before* the OMS/broker is touched. If the process dies before a terminal
outcome is appended, restart returns ``execution_indeterminate`` and refuses to
auto-retry. That sacrifices liveness to preserve at-most-once economic safety.

An indeterminate incident is not a permanent dead end: an explicit account
reconciliation may append ``execution_reconciled`` after the indeterminate
terminal. The reconciliation never edits or replaces the original outcome and
must reference its record hash, so the audit trail remains append-only.

Legacy terminal rows created before canonical-prefix receipts existed can be
migrated without rewriting history. ``legacy_terminal_bound`` is a distinct,
lower-assurance append-only attestation that binds the immutable terminal record
hash to the current paper-account identity and a verified reconciled canonical
prefix. It must never be reported as an original execution-time receipt.

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
RECONCILIATION_STATUS = "execution_reconciled"
LEGACY_BINDING_STATUS = "legacy_terminal_bound"
_ALLOWED_STATUSES = TERMINAL_OUTCOMES | {
    "execution_started",
    RECONCILIATION_STATUS,
    LEGACY_BINDING_STATUS,
}


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

    def reconciliation(self, pending_payload_sha256: str) -> PendingExecutionRecord | None:
        history = self.history(pending_payload_sha256)
        rows = [record for record in history if record.status == RECONCILIATION_STATUS]
        if len(rows) > 1:
            raise ExecutionJournalCorruption(
                f"pending signal {pending_payload_sha256} has multiple reconciliation records"
            )
        return rows[0] if rows else None

    def legacy_binding(self, pending_payload_sha256: str) -> PendingExecutionRecord | None:
        history = self.history(pending_payload_sha256)
        rows = [record for record in history if record.status == LEGACY_BINDING_STATUS]
        if len(rows) > 1:
            raise ExecutionJournalCorruption(
                f"pending signal {pending_payload_sha256} has multiple legacy binding records"
            )
        return rows[0] if rows else None

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

        normalized_details = dict(details or {})
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
            reconciliations = [
                record for record in history if record.status == RECONCILIATION_STATUS
            ]
            legacy_bindings = [
                record for record in history if record.status == LEGACY_BINDING_STATUS
            ]
            if len(terminals) > 1:
                raise ExecutionJournalCorruption(
                    f"pending signal {pending_payload_sha256} has multiple terminal outcomes"
                )
            if len(reconciliations) > 1:
                raise ExecutionJournalCorruption(
                    f"pending signal {pending_payload_sha256} has multiple reconciliation records"
                )
            if len(legacy_bindings) > 1:
                raise ExecutionJournalCorruption(
                    f"pending signal {pending_payload_sha256} has multiple legacy binding records"
                )
            terminal = terminals[0] if terminals else None
            reconciliation = reconciliations[0] if reconciliations else None
            legacy_binding = legacy_bindings[0] if legacy_bindings else None

            if status == RECONCILIATION_STATUS:
                if terminal is None or terminal.status != "execution_indeterminate":
                    raise ExecutionJournalCorruption(
                        "execution_reconciled requires an execution_indeterminate terminal"
                    )
                if str(normalized_details.get("indeterminate_record_sha256") or "") != terminal.record_sha256:
                    raise ExecutionJournalCorruption(
                        "execution_reconciled must bind the indeterminate terminal record hash"
                    )
                if reconciliation is not None:
                    if reconciliation.details == normalized_details:
                        return reconciliation
                    raise ExecutionJournalCorruption(
                        f"pending signal {pending_payload_sha256} is already reconciled"
                    )
            elif status == LEGACY_BINDING_STATUS:
                if terminal is None:
                    raise ExecutionJournalCorruption(
                        "legacy_terminal_bound requires an existing terminal outcome"
                    )
                if dict(terminal.details or {}).get("canonical_prefix_receipt") is not None:
                    raise ExecutionJournalCorruption(
                        "legacy_terminal_bound is only valid for a terminal without an original receipt"
                    )
                if str(normalized_details.get("terminal_record_sha256") or "") != terminal.record_sha256:
                    raise ExecutionJournalCorruption(
                        "legacy_terminal_bound must bind the immutable terminal record hash"
                    )
                identity_sha = str(normalized_details.get("paper_account_identity_sha256") or "")
                if len(identity_sha) != 64:
                    raise ExecutionJournalCorruption(
                        "legacy_terminal_bound requires paper_account_identity_sha256"
                    )
                try:
                    canonical_records = int(normalized_details["canonical_records"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ExecutionJournalCorruption(
                        "legacy_terminal_bound requires canonical_records"
                    ) from exc
                canonical_head = str(normalized_details.get("canonical_head") or "")
                if canonical_records < 0 or len(canonical_head) != 64:
                    raise ExecutionJournalCorruption(
                        "legacy_terminal_bound canonical prefix metadata is invalid"
                    )
                assurance = str(normalized_details.get("assurance") or "")
                if assurance != "operator_reconciled_legacy_terminal_v1":
                    raise ExecutionJournalCorruption(
                        "legacy_terminal_bound requires explicit migration assurance"
                    )
                if legacy_binding is not None:
                    if legacy_binding.details == normalized_details:
                        return legacy_binding
                    raise ExecutionJournalCorruption(
                        f"pending signal {pending_payload_sha256} is already legacy-bound"
                    )
            elif terminal is not None:
                if status == terminal.status and normalized_details == terminal.details:
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
                "details": normalized_details,
                "recorded_at": recorded_at
                or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "previous_record_sha256": previous,
                "record_sha256": "",
            }
            payload["record_sha256"] = _digest(payload)
            record = PendingExecutionRecord(**payload)
            line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            return record


def replay_terminal_status(
    records: Iterable[PendingExecutionRecord],
) -> dict[str, str]:
    result: dict[str, str] = {}
    reconciled: set[str] = set()
    for record in records:
        if record.status in TERMINAL_OUTCOMES:
            result[record.pending_payload_sha256] = record.status
        elif record.status == RECONCILIATION_STATUS:
            reconciled.add(record.pending_payload_sha256)
    for payload in reconciled:
        if result.get(payload) == "execution_indeterminate":
            result[payload] = RECONCILIATION_STATUS
    return result


__all__ = [
    "JOURNAL_SCHEMA_VERSION",
    "TERMINAL_OUTCOMES",
    "RECONCILIATION_STATUS",
    "LEGACY_BINDING_STATUS",
    "ExecutionJournalCorruption",
    "PendingExecutionRecord",
    "PendingExecutionJournal",
    "replay_terminal_status",
]