"""News ingestor.

Pulls news from configured public pages (财新 / 华尔街见闻 / 新浪 / 东方财富
/ 雪球 etc.) into the unified evidence frame. The ingestor focuses on
classification — it never inflates confidence; downstream
:class:`NewsCrossValidator` will decide whether the news is corroborated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quantagent.data.ingestion.daily_evidence_job import (
    DailyEvidenceJobConfig,
    EvidenceIngestor,
    attach_source_profile,
)
from quantagent.data.ingestion.source_registry import SourceCredibilityRegistry
from quantagent.data.providers.base import ProviderUnavailable
from quantagent.data.providers.web_crawler import PublicWebCrawler


@dataclass
class NewsIngestor(EvidenceIngestor):
    name: str = "news"
    source_type: str = "news"
    allow_network: bool = False
    urls: tuple[str, ...] = ()
    local_cache_root: str = "data/v7/evidence/news"
    keyword_to_theme: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "ai": ("ai_compute",),
            "算力": ("ai_compute",),
            "gpu": ("ai_compute",),
            "服务器": ("ai_compute",),
            "数据中心": ("ai_compute",),
            "半导体": ("semiconductor_domestic_substitution",),
            "集成电路": ("semiconductor_domestic_substitution",),
            "储能": ("energy_storage",),
            "新能源车": ("ev_supply_chain",),
            "新能源汽车": ("ev_supply_chain",),
            "商业航天": ("commercial_space",),
            "低空经济": ("low_altitude_economy",),
            "军工": ("defense_modernisation",),
            "创新药": ("innovative_drug",),
            "白酒": ("consumer_recovery",),
        }
    )
    rumor_keywords: tuple[str, ...] = (
        "传闻", "据传", "据悉", "未经证实", "市场传言", "传言", "or so", "may have",
    )

    def fetch(
        self,
        config: DailyEvidenceJobConfig,
        registry: SourceCredibilityRegistry,
    ) -> pd.DataFrame:
        local_frame = self._read_local_cache(config.as_of_date)
        if self.allow_network and self.urls:
            crawled = self._fetch_remote(config.as_of_date)
            if not crawled.empty:
                local_frame = pd.concat([local_frame, crawled], ignore_index=True, sort=False)
        if local_frame.empty:
            return local_frame
        local_frame = self._tag_themes(local_frame)
        local_frame = self._tag_rumor_risk(local_frame)
        local_frame = attach_source_profile(local_frame, registry)
        local_frame["source_type"] = "news"
        local_frame["event_type"] = local_frame.get("event_type", "sentiment_positive")
        local_frame["confidence"] = local_frame.get("confidence", 0.55)
        return local_frame

    def _read_local_cache(self, as_of_date: str) -> pd.DataFrame:
        root = Path(self.local_cache_root)
        if not root.exists():
            return pd.DataFrame()
        frames = [pd.read_csv(path) for path in sorted(root.glob("*.csv"))]
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True, sort=False)
        if "published_at" in merged.columns:
            merged = merged[pd.to_datetime(merged["published_at"], errors="coerce") <= pd.Timestamp(as_of_date)]
        return merged

    def _fetch_remote(self, as_of_date: str) -> pd.DataFrame:
        if not self.urls:
            return pd.DataFrame()
        crawler = PublicWebCrawler(allow_network=self.allow_network)
        try:
            documents = crawler.fetch_documents(
                self.urls,
                as_of_date=as_of_date,
                source_type="news",
                source_reliability=0.62,
            )
        except ProviderUnavailable:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        for doc in documents:
            rows.append(
                {
                    "source_name": doc.source,
                    "url": doc.url,
                    "title": doc.title,
                    "body": doc.body,
                    "published_at": doc.published_at,
                    "available_at": doc.available_at,
                    "raw_hash": doc.content_hash,
                }
            )
        return pd.DataFrame(rows)

    def _tag_themes(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        text = (frame.get("title", "").fillna("") + " " + frame.get("body", "").fillna("")).str.lower()
        themes: list[str] = []
        for body in text:
            tags = set()
            for keyword, theme_list in self.keyword_to_theme.items():
                if keyword.lower() in body:
                    tags.update(theme_list)
            themes.append(",".join(sorted(tags)))
        data = frame.copy()
        data["theme_candidates"] = themes
        return data

    def _tag_rumor_risk(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        text = (frame.get("title", "").fillna("") + " " + frame.get("body", "").fillna("")).str.lower()
        rumor_flags = [
            any(keyword.lower() in body for keyword in self.rumor_keywords)
            for body in text
        ]
        data = frame.copy()
        data["rumor_risk_flag"] = rumor_flags
        # Penalise confidence for rumour-flagged rows
        data["confidence"] = data.get("confidence", 0.55)
        data.loc[data["rumor_risk_flag"], "confidence"] = data.loc[data["rumor_risk_flag"], "confidence"] * 0.55
        return data
