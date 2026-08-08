"""Claim-once protection for economic actions.

An order intent may be delivered many times over: an operator double-clicks, an
API call is retried after a timeout that actually succeeded, a worker is killed
mid-submit and restarts, a socket reconnects and replays its buffer, a broker
sends the same fill callback twice, an event log is replayed to rebuild state,
or a process recovers from a checkpoint. Every one of those paths must converge
on *one* economic order.

The guarantee here is claim-once: the first caller to claim a key performs the
action, and every later caller for that key is told what the first one decided.
Two properties make it hold where a naive in-memory ``set`` would not:

* **Durability.** Claims are appended to disk and fsynced before the claim is
  reported as won, so a process killed between claiming and acting still sees
  the claim on restart.
* **Cross-process exclusion.** The same lock-file protocol works with POSIX
  ``flock`` on Unix and ``msvcrt.locking`` on Windows.  QMT/MiniQMT is normally
  deployed on Windows, so importing a Unix-only ``fcntl`` module at package
  import time would make the live execution safety layer unusable on the very
  host where it is required.
* **Content-derived keys.** The key is a digest of what makes the action
  economically distinct. A random id would make every retry look new, and a key
  that is too coarse silently swallows legitimate re-trades.

What this does *not* do: decide whether two actions are economically the same.
That is the caller's job when it builds the key.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Mapping

try:  # POSIX production/research hosts
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    _fcntl = None

try:  # QMT/MiniQMT production hosts
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - exercised on Unix
    _msvcrt = None

from quantagent.domain.lineage import content_id

#: Bumped when the on-disk record layout changes incompatibly.
SCHEMA_VERSION = "quantagent.idempotency.v1"


class DuplicateAction(RuntimeError):
    """A second attempt to claim a key that is already resolved, under strict mode."""

    def __init__(self, key: str, existing: "ClaimRecord") -> None:
        super().__init__(
            f"idempotency key {key!r} was already claimed at {existing.claimed_at} "
            f"with outcome {existing.outcome!r}"
        )
        self.key = key
        self.existing = existing


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    """One claimed key and whatever the winning caller recorded against it."""

    key: str
    claimed_at: str
    outcome: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "key": self.key,
            "claimedAt": self.claimed_at,
            "outcome": self.outcome,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ClaimRecord":
        return cls(
            key=str(payload["key"]),
            claimed_at=str(payload.get("claimedAt") or ""),
            outcome=payload.get("outcome"),
            payload=dict(payload.get("payload") or {}),
        )


@dataclass(frozen=True, slots=True)
class ClaimResult:
    """Outcome of a claim attempt."""

    granted: bool
    record: ClaimRecord

    @property
    def duplicate(self) -> bool:
        return not self.granted


def order_intent_key(
    *,
    run_id: str,
    signal_id: str,
    symbol: str,
    side: str,
    quantity: int,
    trade_date: str,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """The economic identity of one order intent."""
    return content_id(
        "idem",
        run_id=run_id,
        signal_id=signal_id,
        symbol=symbol,
        side=str(side).upper(),
        quantity=int(quantity),
        trade_date=trade_date,
        **dict(extra or {}),
    )


def broker_callback_key(
    *,
    broker_order_id: str,
    execution_id: str,
    event_type: str,
) -> str:
    """The identity of one broker message."""
    return content_id(
        "cb",
        broker_order_id=broker_order_id,
        execution_id=execution_id,
        event_type=str(event_type).upper(),
    )


class IdempotencyStore:
    """Durable claim-once registry.

    Backed by an append-only JSONL file. Appending rather than rewriting means a
    crash mid-write can at worst leave a trailing partial line; it cannot corrupt
    an earlier claim. Pass ``path=None`` only for single-process research uses.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._lock = RLock()
        self._path = Path(path) if path is not None else None
        self._claims: dict[str, ClaimRecord] = {}
        self._read_offset = 0
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> None:
        """Fold any complete records appended since the previous read."""
        if self._path is None or not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as handle:
            handle.seek(self._read_offset)
            for line in handle:
                if not line.endswith("\n"):
                    # Torn trailing write: leave the offset before it so a later
                    # complete append/recovery can be observed rather than
                    # treating a partial record as evidence.
                    break
                stripped = line.strip()
                self._read_offset += len(line.encode("utf-8"))
                if not stripped:
                    continue
                try:
                    record = ClaimRecord.from_dict(json.loads(stripped))
                except (json.JSONDecodeError, KeyError):
                    continue
                self._claims[record.key] = record

    def _append(self, record: ClaimRecord) -> None:
        if self._path is None:
            return
        line = json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            # The claim must be durable before the guarded economic action can
            # happen.  Otherwise SIGKILL/power-loss can resurrect the order.
            os.fsync(handle.fileno())

    # -- claiming -----------------------------------------------------------
    def claim(
        self,
        key: str,
        *,
        outcome: str | None = None,
        payload: Mapping[str, Any] | None = None,
        strict: bool = False,
    ) -> ClaimResult:
        """Attempt to claim ``key`` exactly once across process restarts."""
        with self._lock, self._exclusive_file_lock():
            self._load()
            existing = self._claims.get(key)
            if existing is not None:
                if strict:
                    raise DuplicateAction(key, existing)
                return ClaimResult(granted=False, record=existing)
            record = ClaimRecord(
                key=key,
                claimed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                outcome=outcome,
                payload=dict(payload or {}),
            )
            self._claims[key] = record
            self._append(record)
            return ClaimResult(granted=True, record=record)

    @contextmanager
    def _exclusive_file_lock(self):
        """Cross-process mutual exclusion on Unix *and* Windows.

        A sibling ``.lock`` file is used so the append-only evidence file stays
        append-only. POSIX uses ``flock``. Windows ``msvcrt.locking`` locks a
        byte range, so the lock file is guaranteed to contain one sentinel byte
        and byte zero is held for the duration of the claim decision.
        """
        if self._path is None:
            yield
            return
        lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        with lock_path.open("a+b") as handle:
            if _fcntl is not None:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
                return

            if _msvcrt is not None:  # pragma: no cover - executed on Windows CI/host
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

            raise RuntimeError(
                "no supported cross-process file-lock backend is available; "
                "durable economic idempotency cannot be guaranteed"
            )

    def resolve(
        self,
        key: str,
        *,
        outcome: str,
        payload: Mapping[str, Any] | None = None,
    ) -> ClaimRecord:
        """Record what the winning caller produced as a second append."""
        with self._lock, self._exclusive_file_lock():
            self._load()
            existing = self._claims.get(key)
            if existing is None:
                raise KeyError(f"cannot resolve unclaimed key {key!r}")
            merged = dict(existing.payload)
            merged.update(dict(payload or {}))
            record = ClaimRecord(
                key=key,
                claimed_at=existing.claimed_at,
                outcome=outcome,
                payload=merged,
            )
            self._claims[key] = record
            self._append(record)
            return record

    # -- reading ------------------------------------------------------------
    @property
    def path(self) -> Path | None:
        return self._path

    def get(self, key: str) -> ClaimRecord | None:
        with self._lock:
            return self._claims.get(key)

    def seen(self, key: str) -> bool:
        with self._lock:
            return key in self._claims

    def __len__(self) -> int:
        with self._lock:
            return len(self._claims)

    def __iter__(self) -> Iterator[ClaimRecord]:
        with self._lock:
            return iter(list(self._claims.values()))


__all__ = [
    "ClaimRecord",
    "ClaimResult",
    "DuplicateAction",
    "IdempotencyStore",
    "SCHEMA_VERSION",
    "broker_callback_key",
    "order_intent_key",
]
