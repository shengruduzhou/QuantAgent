"""Regulatory penalty and audit-opinion evidence ingestor.

Reads CSRC penalty decisions, exchange inquiry letters (问询函), audit
opinion changes, and non-standard audit reports. Each row maps to an
``EvidenceRecord`` with ``event_type`` in:

* ``regulatory_penalty`` — 证监会 / 交易所处罚
* ``audit_opinion``      — 非标审计意见 / 保留意见
* ``inquiry_letter``     — 问询函
* ``restatement``        — 财务重述

The fraud-risk agent multiplies its score by the count of these events
and uses the row count to set ``regulatory_penalty_score`` and
``audit_opinion_score`` on the downstream :class:`FraudRiskScore`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.data.ingestion.daily_evidence_job import (
    DailyEvidenceJobConfig,
    EvidenceIngestor,
    attach_source_profile,
)
from quantagent.data.ingestion.source_registry import SourceCredibilityRegistry


_DEFAULT_PENALTY_PATTERNS = {
    "regulatory_penalty": ("行政处罚", "立案", "立案调查", "警示函", "纪律处分"),
    "inquiry_letter": ("问询函", "关注函"),
    "audit_opinion": ("保留意见", "非标准", "非标审计", "无法表示意见", "否定意见"),
    "restatement": ("会计差错", "前期差错", "财务报表更正", "重述"),
}


@dataclass
class RegulatoryPenaltyIngestor(EvidenceIngestor):
    name: str = "regulatory_penalty"
    source_type: str = "regulatory"
    local_cache_root: str = field(default_factory=lambda: str(quant_paths().data_root / "v7" / "evidence" / "regulatory"))

    def fetch(
        self,
        config: DailyEvidenceJobConfig,
        registry: SourceCredibilityRegistry,
    ) -> pd.DataFrame:
        root = Path(self.local_cache_root)
        if not root.exists():
            return pd.DataFrame()
        frames = [pd.read_csv(path) for path in sorted(root.glob("*.csv"))]
        if not frames:
            return pd.DataFrame()
        frame = pd.concat(frames, ignore_index=True, sort=False)
        if "published_at" in frame.columns:
            frame = frame[pd.to_datetime(frame["published_at"], errors="coerce") <= pd.Timestamp(config.as_of_date)]
        if frame.empty:
            return frame
        frame = self._tag_events(frame)
        frame = attach_source_profile(frame, registry)
        frame["source_type"] = "regulatory"
        return frame

    def _tag_events(self, frame: pd.DataFrame) -> pd.DataFrame:
        text = (
            frame.get("title", "").fillna("") + " " + frame.get("body", "").fillna("")
        ).str.lower()
        events: list[str] = []
        for body in text:
            tag = "regulatory_penalty"
            for event_type, patterns in _DEFAULT_PENALTY_PATTERNS.items():
                if any(pattern.lower() in body for pattern in patterns):
                    tag = event_type
                    break
            events.append(tag)
        data = frame.copy()
        data["event_type"] = events
        data["confidence"] = data.get("confidence", 0.90)
        return data
