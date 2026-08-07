"""Claim-once protection for economic actions.

An order intent may be delivered many times over: an operator double-clicks, an
API call is retried after a timeout that actually succeeded, a worker is killed
mid-submit and restarts, a socket reconnects and replays its buffer, a broker
sends the same fill callback twice, an event log is replayed to rebuild state,
or a process recovers from a checkpoint. Every one of those paths must converge
on *one* economic order.

The guarantee here is claim-once: the first caller to claim a key performs the
action, and every later caller for that key is told what the first one decided.
Two properties make it hold where a naive in-memory `set` would not:

* **Durability.** Claims are appended to disk and fsynced before the claim is
  reported as won, so a process killed between claiming and acting still sees
  the claim on restart. An in-memory guard forgets everything on SIGKILL, which
  is exactly when duplicates are most likely.
* **Content-derived keys.** The key is a digest of what makes the action
  economically distinct. A random id would make every retry look new, and a key
  that is too coarse (say symbol+side, with no date) silently swallows a
  legitimate re-trade on a later day — the INC-E1 defect this project already
  paid for once.

What this does *not* do: decide whether two actions are economically the same.
That is the caller's job when it builds the key, and getting it wrong is
visible in both directions — too coarse drops real orders, too fine admits
duplicates.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Mapping

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
    #: Free-form marker of what the winner produced — typically an order_id.
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

    #: True only for the caller that won the right to perform the action.
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
    """The economic identity of one order intent.

    `trade_date` is deliberately part of the key: without it, buying a symbol
    on Monday would suppress the identical buy on Wednesday. `quantity` is
    included so a revised intent for the same signal is a genuinely new action
    rather than a silent no-op.
    """
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
    """The identity of one broker message.

    Brokers redeliver. The same execution report arriving twice must apply once,
    or a position is credited twice and the book silently diverges from reality.
    """
    return content_id(
        "cb",
        broker_order_id=broker_order_id,
        execution_id=execution_id,
        event_type=str(event_type).upper(),
    )


class IdempotencyStore:
    """Durable claim-once registry.

    Backed by an append-only JSONL file. Appending (rather than rewriting)
    means a crash mid-write can at worst leave a trailing partial line, which
    `_load` skips; it can never corrupt an earlier claim.

    Pass ``path=None`` for an in-memory store. That is appropriate for a single
    backtest process, and inappropriate anywhere a restart must not replay an
    economic action.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._lock = RLock()
        self._path = Path(path) if path is not None else None
        self._claims: dict[str, ClaimRecord] = {}
        #: Bytes of the file already folded into `_claims`. Claims appended by
        #: *other* processes after we started are read from here on each claim.
        self._read_offset = 0
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> None:
        """Fold any records appended since we last looked.

        Re-reading (rather than loading once at construction) is what makes the
        store safe for *concurrent* workers. Loading once was safe only for
        sequential processes: two recovery workers started together each held an
        empty view and both granted the same key, producing two economic orders.
        """
        if self._path is None or not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as handle:
            handle.seek(self._read_offset)
            for line in handle:
                if not line.endswith("\n"):
                    # A torn trailing write from a killed process. Stop before
                    # it and leave the offset so a later complete write is read.
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
            # Without the fsync a SIGKILL can lose the claim while the action it
            # guarded already happened — the exact window duplicates come from.
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
        """Attempt to claim `key`.

        Returns ``granted=True`` exactly once per key across the store's
        lifetime, including across process restarts when the store is durable.
        With ``strict=True`` a duplicate raises instead of being reported, for
        call sites where a repeat indicates a genuine bug rather than an
        expected retry.
        """
        with self._lock, self._exclusive_file_lock():
            # Refresh under the lock: another process may have claimed this key
            # since we last read. Deciding from a stale in-memory view is how
            # two concurrent workers both "win".
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
        """Cross-process mutual exclusion around the claim decision.

        The in-process `RLock` only serialises threads of one interpreter; two
        worker *processes* need the OS to arbitrate. A separate lock file is used
        so the ledger itself is only ever opened for append.
        """
        if self._path is None:
            yield
            return
        lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def resolve(self, key: str, *, outcome: str, payload: Mapping[str, Any] | None = None) -> ClaimRecord:
        """Record what the winning caller produced.

        Written as a second append rather than an in-place edit so the file
        stays append-only and the original claim remains readable.
        """
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
        """Where claims are durable, or None for an in-memory store.

        Public so a caller that shares one claim file between two stores can
        assert they really do share it — a store silently constructed with
        ``path=None`` is an in-memory guard wearing a durable one's name.
        """
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
