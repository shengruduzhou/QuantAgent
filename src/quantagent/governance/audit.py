"""Append-only audit log for inter-agent messages and decisions.

Persisted **outside Git**, under the runtime tree, for two reasons: the log
grows without bound, and a governance record that can be amended by a rebase is
not a governance record.

Tamper-evidence is a hash chain. Each entry carries ``prev_hash`` and
``entry_hash``, so removing or editing any line breaks verification from that
point onward. This does not make the log tamper-*proof* -- anyone who can write
the file can rewrite the whole chain -- and it is documented that way rather
than being described as immutable.

Concurrency (round-22, defect A-04)
-----------------------------------
``append`` is a read-then-write: it derives ``sequence`` and ``prev_hash`` from
the current chain tail. Without mutual exclusion two writers read the same tail
and emit two entries claiming the same position, which forks the chain and
strands every record on the losing branch -- while both writers return
successfully. Measured at four processes x forty entries: 160 lines on disk,
zero reachable from genesis.

The write path therefore holds an exclusive cross-process file lock for the
whole read-tail -> compute-hash -> durable-append decision, re-reading the tail
*inside* the lock so it can never be a stale one. This is the same lock helper
used by the repository's other append-only writers (``paper/execution_journal``,
``paper/pending_signal``, ``paper/account_identity``, ``execution/parent_child``).

Durability failure (DEF-017)
----------------------------
A failed ``fsync`` latches the log closed rather than resynchronising -- see
``AuditWriteUnavailable``.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Mapping

try:  # POSIX research/CI hosts
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None

try:  # QMT/MiniQMT Windows hosts
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - Unix
    _msvcrt = None

from quantagent.governance.envelopes import payload_hash

GENESIS_HASH = "0" * 64

#: Read backwards in blocks of this size when locating the chain tail. The tail
#: read used to scan the whole file, making `n` appends O(n^2) against a log the
#: module docstring itself describes as unbounded.
_TAIL_BLOCK_BYTES = 8192


class AuditChainCorruption(RuntimeError):
    """The chain does not verify: a record was altered, reordered or removed.

    Raised only by the explicit ``require_intact`` entry point and by ``append``
    when the tail it would chain onto is itself unreadable. ``verify`` reports
    rather than raises, so a caller can surface the break without aborting.
    """


class AuditWriteUnavailable(RuntimeError):
    """A durable write failed, so this log will not accept another one.

    After a failed append nobody knows whether the bytes landed: an ``fsync``
    that raises ``EIO`` has already handed the line to the OS, a full disk may
    have written part of it, a read-only mount may have written none. Appending
    again would chain onto whatever that left behind -- possibly a torn line,
    possibly a complete one the caller believes was rejected -- and the damage
    only surfaces at read time, when it is already durable.

    Failing closed rather than resynchronising is deliberate, and is the same
    latch ``quantagent.domain.ledger`` adopted after DEF-017. A fresh process
    re-reads the file honestly and continues from what is actually there; the
    process that saw the failure does not get to guess.
    """


@dataclass
class AuditEntry:
    sequence: int
    recorded_at: str
    kind: str
    actor: str
    subject: str
    payload: dict[str, Any]
    prev_hash: str
    entry_hash: str = ""

    def compute_hash(self) -> str:
        body = {
            "sequence": self.sequence, "recorded_at": self.recorded_at,
            "kind": self.kind, "actor": self.actor, "subject": self.subject,
            "payload": self.payload, "prev_hash": self.prev_hash,
        }
        return payload_hash(body)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLog:
    """Hash-chained JSONL log. Appends only; never rewrites an entry."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = RLock()
        self._write_failure: str | None = None

    # -- locking ----------------------------------------------------------
    @contextmanager
    def _exclusive_file_lock(self) -> Iterator[None]:
        """Serialize the read-tail -> sequence/hash -> durable append decision.

        The lock lives in a sidecar file rather than on the log itself: the log
        is opened in append mode for a few microseconds per write, and locking a
        handle that is repeatedly reopened gives no protection across the gap.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with open(lock_path, "a+b") as handle:
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

            raise RuntimeError(  # pragma: no cover - no such host in CI
                "no supported cross-process file locking primitive; refusing to "
                "append to a hash-chained governance log without mutual exclusion"
            )

    # -- writing ----------------------------------------------------------
    def append(
        self, *, kind: str, actor: str, subject: str, payload: Mapping[str, Any]
    ) -> AuditEntry:
        with self._thread_lock:
            if self._write_failure is not None:
                raise AuditWriteUnavailable(
                    f"governance audit log {self.path} was latched closed by an "
                    f"earlier durable-write failure ({self._write_failure}); the "
                    "chain tail on disk is unknown, so this process will not "
                    "append on top of it"
                )
            with self._exclusive_file_lock():
                # Re-read the tail *inside* the lock. A tail captured before
                # acquiring it is exactly the stale head that forks the chain.
                last = self._tail_entry()
                entry = AuditEntry(
                    sequence=(last.sequence + 1) if last else 0,
                    recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    kind=kind, actor=actor, subject=subject, payload=dict(payload),
                    prev_hash=last.entry_hash if last else GENESIS_HASH,
                )
                entry.entry_hash = entry.compute_hash()
                line = json.dumps(entry.to_dict(), ensure_ascii=False, default=str) + "\n"
                try:
                    with open(self.path, "a", encoding="utf-8") as handle:
                        handle.write(line)
                        handle.flush()
                        os.fsync(handle.fileno())
                except OSError as exc:
                    # Latch closed before re-raising. The caller may well retry,
                    # and a retry that succeeded would chain from a tail nobody
                    # can vouch for.
                    self._write_failure = f"{type(exc).__name__}: {exc}"
                    raise
                return entry

    # -- reading ----------------------------------------------------------
    def _raw_lines(self) -> Iterator[str]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield line

    def entries(self) -> Iterator[AuditEntry]:
        for line in self._raw_lines():
            yield AuditEntry(**json.loads(line))

    def _last_raw_line(self) -> str | None:
        """Return the final non-empty line without reading the whole file.

        Newline (0x0A) never occurs inside a multi-byte UTF-8 sequence and
        ``json.dumps`` escapes newlines inside strings, so splitting the tail
        bytes on ``b"\\n"`` yields whole records regardless of where the block
        boundary fell.
        """

        try:
            with open(self.path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                position = handle.tell()
                buffer = b""
                while position > 0:
                    step = min(_TAIL_BLOCK_BYTES, position)
                    position -= step
                    handle.seek(position)
                    buffer = handle.read(step) + buffer
                    trimmed = buffer.rstrip(b"\r\n \t")
                    if b"\n" in trimmed:
                        return trimmed.rsplit(b"\n", 1)[1].strip().decode("utf-8")
                    if position == 0:
                        return trimmed.decode("utf-8") if trimmed else None
                return None
        except FileNotFoundError:
            return None

    def _tail_entry(self) -> AuditEntry | None:
        """Chain tail, plus the O(1) integrity check ``append`` can afford.

        A full-chain walk on every append would make writing O(n^2) again, so
        only the tail's own self-consistency is checked here: it catches a tail
        that was edited or torn, which is what ``append`` is about to chain onto.
        Breaks further back are the business of ``require_intact``.
        """

        line = self._last_raw_line()
        if line is None:
            return None
        try:
            entry = AuditEntry(**json.loads(line))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise AuditChainCorruption(
                f"the last line of governance audit log {self.path} is not a "
                f"readable entry ({exc}); refusing to chain onto it"
            ) from exc
        if entry.entry_hash != entry.compute_hash():
            raise AuditChainCorruption(
                f"entry {entry.sequence} at the tail of governance audit log "
                f"{self.path} does not match its recorded hash; refusing to "
                "chain onto an edited record"
            )
        return entry

    def last_entry(self) -> AuditEntry | None:
        return self._tail_entry()

    def __len__(self) -> int:
        return sum(1 for _ in self._raw_lines())

    def verify(self) -> dict[str, Any]:
        """Walk the chain and report the first break, if any.

        Reports rather than raises, and reports *counts*: a fork leaves the file
        full of records that no verifier can reach, and "valid: False" alone does
        not say how many governance decisions became unreachable.
        ``entries_total`` counts every line on disk, ``reachable`` counts those
        the walk from ``GENESIS_HASH`` actually reached, and ``orphaned`` is the
        difference. Use ``require_intact`` when a break should stop the caller.
        """

        expected_prev = GENESIS_HASH
        expected_sequence = 0
        reachable = 0
        entries_total = 0
        error: str | None = None

        for line_no, line in enumerate(self._raw_lines(), start=1):
            entries_total += 1
            if error is not None:
                # Keep counting lines so `orphaned` is the true remainder.
                continue
            try:
                entry = AuditEntry(**json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                error = f"line {line_no} is not a readable audit entry: {exc}"
                continue
            if entry.sequence != expected_sequence:
                error = (f"sequence jumped to {entry.sequence}, "
                         f"expected {expected_sequence}")
                continue
            if entry.prev_hash != expected_prev:
                error = (f"entry {entry.sequence} does not chain to its "
                         "predecessor; the log has been edited or truncated")
                continue
            if entry.entry_hash != entry.compute_hash():
                error = (f"entry {entry.sequence} content does not match "
                         "its recorded hash")
                continue
            expected_prev = entry.entry_hash
            expected_sequence += 1
            reachable += 1

        result: dict[str, Any] = {
            "valid": error is None,
            "checked": reachable,
            "reachable": reachable,
            "entries_total": entries_total,
            "orphaned": entries_total - reachable,
        }
        if error is None:
            result["head"] = expected_prev
        else:
            result["error"] = error
        return result

    def require_intact(self, *, expected_head: str | None = None) -> dict[str, Any]:
        """``verify``, but a broken chain raises instead of being returned.

        The A-04 failure mode was not that corruption went undetected -- it was
        that nothing asked. Callers that treat the log as evidence should call
        this, so an unreachable record is an error at the point of use rather
        than a quiet absence.

        ``expected_head`` closes the one break a hash chain cannot see by
        itself. Dropping entries from the *end* leaves a shorter chain that is
        internally perfect, because nothing inside the file records how long the
        file is supposed to be. A caller that recorded the head from an earlier
        read can pass it back here; without such an anchor, tail truncation is
        undetectable and this method will report the truncated log as intact.
        """

        result = self.verify()
        if not result["valid"]:
            raise AuditChainCorruption(
                f"governance audit chain at {self.path} is broken: "
                f"{result['error']}; {result['reachable']} of "
                f"{result['entries_total']} records are reachable from genesis, "
                f"{result['orphaned']} are orphaned"
            )
        if expected_head is not None and result["head"] != expected_head:
            raise AuditChainCorruption(
                f"governance audit chain at {self.path} verifies internally but "
                f"its head is {result['head']}, not the expected "
                f"{expected_head}; {result['entries_total']} records remain, so "
                "entries were removed from the end of the log"
            )
        return result
