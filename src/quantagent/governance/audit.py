"""Append-only audit log for inter-agent messages and decisions.

Persisted **outside Git**, under the runtime tree, for two reasons: the log
grows without bound, and a governance record that can be amended by a rebase is
not a governance record.

Tamper-evidence is a hash chain. Each entry carries ``prev_hash`` and
``entry_hash``, so removing or editing any line breaks verification from that
point onward. This does not make the log tamper-*proof* -- anyone who can write
the file can rewrite the whole chain -- and it is documented that way rather
than being described as immutable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from quantagent.governance.envelopes import payload_hash

GENESIS_HASH = "0" * 64


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

    # -- writing ----------------------------------------------------------
    def append(
        self, *, kind: str, actor: str, subject: str, payload: Mapping[str, Any]
    ) -> AuditEntry:
        last = self.last_entry()
        entry = AuditEntry(
            sequence=(last.sequence + 1) if last else 0,
            recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            kind=kind, actor=actor, subject=subject, payload=dict(payload),
            prev_hash=last.entry_hash if last else GENESIS_HASH,
        )
        entry.entry_hash = entry.compute_hash()
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False, default=str) + "\n")
        return entry

    # -- reading ----------------------------------------------------------
    def entries(self) -> Iterator[AuditEntry]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield AuditEntry(**json.loads(line))

    def last_entry(self) -> AuditEntry | None:
        last: AuditEntry | None = None
        for entry in self.entries():
            last = entry
        return last

    def __len__(self) -> int:
        return sum(1 for _ in self.entries())

    def verify(self) -> dict[str, Any]:
        """Walk the chain and report the first break, if any."""
        expected_prev = GENESIS_HASH
        expected_sequence = 0
        checked = 0
        for entry in self.entries():
            if entry.sequence != expected_sequence:
                return {"valid": False, "checked": checked,
                        "error": f"sequence jumped to {entry.sequence}, "
                                 f"expected {expected_sequence}"}
            if entry.prev_hash != expected_prev:
                return {"valid": False, "checked": checked,
                        "error": f"entry {entry.sequence} does not chain to its "
                                 "predecessor; the log has been edited or truncated"}
            if entry.entry_hash != entry.compute_hash():
                return {"valid": False, "checked": checked,
                        "error": f"entry {entry.sequence} content does not match "
                                 "its recorded hash"}
            expected_prev = entry.entry_hash
            expected_sequence += 1
            checked += 1
        return {"valid": True, "checked": checked, "head": expected_prev}
