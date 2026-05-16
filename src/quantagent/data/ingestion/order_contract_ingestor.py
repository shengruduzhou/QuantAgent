"""Order / contract evidence ingestor.

Reads pre-cached order-contract evidence (typically extracted from
exchange announcements) and emits an evidence frame keyed by
``(symbol, event_type=order_confirmed)``. The downstream
``order_contract_agent`` and the long-horizon-factor pipeline turn this
into ``order_visibility_score`` and ``capacity_release_score``.

The ingestor is intentionally simple — it does not crawl websites; the
disclosure ingestor handles HTML parsing. Production deployments are
expected to populate the local CSV via TuShare ``express`` /
``announcement`` APIs or an internal NLP pipeline that extracts contract
size / counterparty / delivery schedule from the announcement body.
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


@dataclass
class OrderContractIngestor(EvidenceIngestor):
    name: str = "order_contract"
    source_type: str = "disclosure"
    local_cache_root: str = field(default_factory=lambda: str(quant_paths().data_root / "v7" / "evidence" / "order_contract"))

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
        frame = frame.copy()
        frame["event_type"] = frame.get("event_type", "order_confirmed")
        frame["confidence"] = frame.get("confidence", 0.80)
        frame = attach_source_profile(frame, registry)
        frame["source_type"] = "disclosure"
        return frame
