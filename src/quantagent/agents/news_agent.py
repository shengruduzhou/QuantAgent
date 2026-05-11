from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1

import numpy as np
import pandas as pd

from quantagent.agents.views_schema import EvidenceRecord


POSITIVE_TERMS = ("support", "growth", "beat", "approval", "upgrade", "buyback", "contract")
NEGATIVE_TERMS = ("risk", "penalty", "fraud", "downgrade", "loss", "probe", "default")


@dataclass
class NewsAgent:
    source: str = "news_agent"

    def run(self, news: pd.DataFrame) -> list[EvidenceRecord]:
        if news.empty:
            return []
        records: list[EvidenceRecord] = []
        for _, row in news.iterrows():
            text = f"{row.get('title', '')} {row.get('summary', '')}".lower()
            score = _score_text(text)
            if "polarity" in row and pd.notna(row["polarity"]):
                score = 0.5 * score + 0.5 * float(row["polarity"])
            confidence = float(np.clip(0.4 + abs(score) * 0.4, 0.2, 0.95))
            raw = {"title": row.get("title", ""), "summary": row.get("summary", "")}
            records.append(
                EvidenceRecord(
                    source=self.source,
                    timestamp=str(row.get("timestamp", row.get("trade_date", ""))),
                    symbol=str(row["symbol"]) if row.get("symbol") is not None else None,
                    sector=str(row.get("sector")) if row.get("sector") else None,
                    event_type=str(row.get("event_type", "news")),
                    horizon_days=int(row.get("horizon_days", 5)),
                    direction=float(np.sign(score)),
                    magnitude=float(abs(score)),
                    confidence=confidence,
                    rationale=str(row.get("title", ""))[:240],
                    raw_reference={"hash": sha1(str(raw).encode("utf-8")).hexdigest()},
                )
            )
        return records


def _score_text(text: str) -> float:
    pos = sum(text.count(term) for term in POSITIVE_TERMS)
    neg = sum(text.count(term) for term in NEGATIVE_TERMS)
    return float(np.tanh((pos - neg) / 2.0))

