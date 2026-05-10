from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json


class AuditLogger:
    """Append-only deterministic JSONL audit logger."""

    def __init__(self, log_dir: str | Path = "logs/execution", file_name: str = "audit.jsonl") -> None:
        self.path = Path(log_dir) / file_name

    def write(self, event_type: str, payload: Any) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(payload) if is_dataclass(payload) else dict(payload)
        row = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "payload": data,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")
        return self.path
