from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json


@dataclass(frozen=True)
class EvidenceRecord:
    source: str
    timestamp: str
    symbol: str | None = None
    sector: str | None = None
    event_type: str = "unknown"
    horizon_days: int = 5
    direction: float = 0.0
    magnitude: float = 0.0
    confidence: float = 0.5
    decay_half_life: float = 5.0
    rationale: str = ""
    raw_reference: str | dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentView:
    view_id: str
    symbols: tuple[str, ...]
    exposure: dict[str, float]
    q: float
    omega: float
    confidence: float
    constraints: dict[str, Any] = field(default_factory=dict)
    expires_at: str | None = None
    evidence: tuple[EvidenceRecord, ...] = ()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_audit_jsonl(records: list[EvidenceRecord | AgentView], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(record), sort_keys=True, ensure_ascii=False, default=str) for record in records]
    output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output
