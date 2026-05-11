from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json


@dataclass(frozen=True)
class AuditReplayResult:
    event_count: int
    event_types: dict[str, int]
    last_event: dict[str, Any] | None


class AuditReplay:
    def replay(self, path: str | Path) -> AuditReplayResult:
        audit_path = Path(path)
        counts: dict[str, int] = {}
        last: dict[str, Any] | None = None
        if not audit_path.exists():
            return AuditReplayResult(0, {}, None)
        with audit_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                event_type = str(event.get("event_type", "unknown"))
                counts[event_type] = counts.get(event_type, 0) + 1
                last = event
        return AuditReplayResult(sum(counts.values()), counts, last)

