"""Append-only, tamper-evident event ledger for the local paper broker.

The ledger is the system of record. Cash, positions, open orders, realised and
unrealised P&L, risk state and the kill switch are all *derived* from it by
replay -- none of them is stored as mutable state. That is what makes restart
recovery meaningful: there is no separate snapshot that could disagree with the
history, because the history is the only copy.

Tamper-evidence is a hash chain: each event carries ``previous_hash`` and
``event_hash``, so editing or removing any event breaks verification from that
point on. This is **tamper-evident, not tamper-proof** -- anyone who can write
the file can rewrite the whole chain -- and it is documented that way rather
than overstated.

Sequence numbers are assigned by the ledger and are strictly contiguous from 0.
A gap is a corruption signal, not a tolerable quirk.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from uuid import uuid4

GENESIS_HASH = "0" * 64

# --- event types ------------------------------------------------------------
SIGNAL_CREATED = "SIGNAL_CREATED"
TARGET_GENERATED = "TARGET_GENERATED"
RISK_APPROVED = "RISK_APPROVED"
RISK_REJECTED = "RISK_REJECTED"
ORDER_CREATED = "ORDER_CREATED"
ORDER_ACCEPTED = "ORDER_ACCEPTED"
ORDER_REJECTED = "ORDER_REJECTED"
ORDER_PARTIALLY_FILLED = "ORDER_PARTIALLY_FILLED"
ORDER_FILLED = "ORDER_FILLED"
ORDER_CANCEL_REQUESTED = "ORDER_CANCEL_REQUESTED"
ORDER_CANCELLED = "ORDER_CANCELLED"
CASH_CHANGED = "CASH_CHANGED"
POSITION_CHANGED = "POSITION_CHANGED"
CORPORATE_ACTION_APPLIED = "CORPORATE_ACTION_APPLIED"
MARK_TO_MARKET = "MARK_TO_MARKET"
KILL_SWITCH_ARMED = "KILL_SWITCH_ARMED"
KILL_SWITCH_TRIGGERED = "KILL_SWITCH_TRIGGERED"
SESSION_CLOSED = "SESSION_CLOSED"
RECONCILIATION_PASSED = "RECONCILIATION_PASSED"
RECONCILIATION_FAILED = "RECONCILIATION_FAILED"

EVENT_TYPES: tuple[str, ...] = (
    SIGNAL_CREATED, TARGET_GENERATED, RISK_APPROVED, RISK_REJECTED,
    ORDER_CREATED, ORDER_ACCEPTED, ORDER_REJECTED, ORDER_PARTIALLY_FILLED,
    ORDER_FILLED, ORDER_CANCEL_REQUESTED, ORDER_CANCELLED, CASH_CHANGED,
    POSITION_CHANGED, CORPORATE_ACTION_APPLIED, MARK_TO_MARKET,
    KILL_SWITCH_ARMED, KILL_SWITCH_TRIGGERED, SESSION_CLOSED,
    RECONCILIATION_PASSED, RECONCILIATION_FAILED,
)


class LedgerError(RuntimeError):
    """Raised when an append would violate the ledger's invariants."""


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


@dataclass
class Event:
    event_id: str
    sequence: int
    event_time: str
    market_time: str | None
    event_type: str
    run_id: str
    strategy_id: str | None
    portfolio_id: str
    symbol: str | None
    payload: dict[str, Any]
    previous_hash: str
    event_hash: str = ""

    def compute_hash(self) -> str:
        body = {
            "event_id": self.event_id, "sequence": self.sequence,
            "event_time": self.event_time, "market_time": self.market_time,
            "event_type": self.event_type, "run_id": self.run_id,
            "strategy_id": self.strategy_id, "portfolio_id": self.portfolio_id,
            "symbol": self.symbol, "payload": self.payload,
            "previous_hash": self.previous_hash,
        }
        return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventLedger:
    """Append-only JSONL ledger with a hash chain and contiguous sequences."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence, self._head = self._resume()

    # -- state ------------------------------------------------------------
    def _resume(self) -> tuple[int, str]:
        sequence, head = 0, GENESIS_HASH
        for event in self.read():
            sequence = event.sequence + 1
            head = event.event_hash
        return sequence, head

    @property
    def next_sequence(self) -> int:
        return self._sequence

    @property
    def head_hash(self) -> str:
        return self._head

    def __len__(self) -> int:
        return sum(1 for _ in self.read())

    # -- writing ----------------------------------------------------------
    def append(
        self,
        event_type: str,
        *,
        run_id: str,
        portfolio_id: str,
        payload: Mapping[str, Any] | None = None,
        symbol: str | None = None,
        strategy_id: str | None = None,
        market_time: str | None = None,
    ) -> Event:
        if event_type not in EVENT_TYPES:
            raise LedgerError(
                f"unknown event type {event_type!r}; known: {list(EVENT_TYPES)}"
            )
        event = Event(
            event_id=str(uuid4()),
            sequence=self._sequence,
            event_time=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            market_time=market_time,
            event_type=event_type,
            run_id=run_id,
            strategy_id=strategy_id,
            portfolio_id=portfolio_id,
            symbol=symbol,
            payload=dict(payload or {}),
            previous_hash=self._head,
        )
        event.event_hash = event.compute_hash()

        # Write and flush before advancing in-memory state, so a crash mid-write
        # cannot leave the ledger believing it recorded something it did not.
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(_canonical(event.to_dict()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        self._sequence += 1
        self._head = event.event_hash
        return event

    # -- reading ----------------------------------------------------------
    def read(self) -> Iterator[Event]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield Event(**json.loads(line))

    def events_of(self, *event_types: str) -> Iterator[Event]:
        wanted = set(event_types)
        for event in self.read():
            if event.event_type in wanted:
                yield event

    # -- integrity --------------------------------------------------------
    def verify(self) -> dict[str, Any]:
        """Walk the chain and report the first break, if any."""
        expected_previous = GENESIS_HASH
        expected_sequence = 0
        checked = 0
        for event in self.read():
            if event.sequence != expected_sequence:
                return {"valid": False, "checked": checked,
                        "error": f"sequence gap: saw {event.sequence}, "
                                 f"expected {expected_sequence}"}
            if event.previous_hash != expected_previous:
                return {"valid": False, "checked": checked,
                        "error": f"event {event.sequence} does not chain to its "
                                 "predecessor; the ledger was edited or truncated"}
            if event.event_hash != event.compute_hash():
                return {"valid": False, "checked": checked,
                        "error": f"event {event.sequence} content does not match "
                                 "its recorded hash"}
            expected_previous = event.event_hash
            expected_sequence += 1
            checked += 1
        return {
            "valid": True, "checked": checked, "head": expected_previous,
            "guarantee": "tamper-evident, not tamper-proof: a writer with file "
                         "access can rebuild the whole chain",
        }
