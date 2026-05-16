"""Policy document ingestor.

Pulls red-headed policy documents from a configured list of URLs (or a
local replay folder). Default mode is offline: it only reads a CSV / JSONL
cache under the unified V7 evidence root so unit tests stay deterministic.

When ``allow_network=True`` the ingestor delegates to
:class:`PublicWebCrawler` and tags each fetched document with the matching
source profile (authority, reliability, primary/official flags).

The ingestor has two network modes:

1. ``urls`` – fetch a static list of documents (legacy path).
2. ``active_discovery`` – walk every official-tier ``SourceProfile``
   that exposes ``discovery_urls``/``rss_urls``/``sitemap_urls`` and
   follow the index pages to find new policy articles. This is the
   "active discovery" model demanded by the V7 evidence layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.data.ingestion.daily_evidence_job import (
    DailyEvidenceJobConfig,
    EvidenceIngestor,
    attach_source_profile,
)
from quantagent.data.ingestion.source_registry import SourceCredibilityRegistry
from quantagent.data.providers.base import ProviderUnavailable
from quantagent.data.providers.web_crawler import PublicWebCrawler


@dataclass
class PolicyIngestor(EvidenceIngestor):
    """Pull policy documents from URLs or a local cache."""

    name: str = "policy"
    source_type: str = "policy"
    allow_network: bool = False
    urls: tuple[str, ...] = ()
    active_discovery: bool = False
    max_articles_per_source: int = 25
    local_cache_root: str = field(default_factory=lambda: str(quant_paths().data_root / "v7" / "evidence" / "policy"))
    keyword_to_theme: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "ai": ("ai_compute",),
            "人工智能": ("ai_compute",),
            "算力": ("ai_compute",),
            "数据中心": ("ai_compute",),
            "半导体": ("semiconductor_domestic_substitution",),
            "集成电路": ("semiconductor_domestic_substitution",),
            "国产替代": ("semiconductor_domestic_substitution",),
            "储能": ("energy_storage",),
            "新能源": ("energy_storage",),
            "商业航天": ("commercial_space",),
            "低空": ("low_altitude_economy",),
            "创新药": ("innovative_drug",),
            "军工": ("defense_modernisation",),
        }
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
        if self.allow_network and self.active_discovery:
            discovered = self._discover_remote(config.as_of_date, registry)
            if not discovered.empty:
                local_frame = pd.concat([local_frame, discovered], ignore_index=True, sort=False)
        if local_frame.empty:
            return local_frame
        local_frame = self._tag_themes(local_frame)
        local_frame = attach_source_profile(local_frame, registry)
        local_frame["source_type"] = "policy"
        local_frame["event_type"] = local_frame.get("event_type", "policy_support")
        local_frame["confidence"] = local_frame.get("confidence", 0.85)
        return local_frame

    def _read_local_cache(self, as_of_date: str) -> pd.DataFrame:
        root = Path(self.local_cache_root)
        if not root.exists():
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        for path in sorted(root.glob("*.csv")):
            frames.append(pd.read_csv(path))
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
                source_type="policy",
                source_reliability=0.85,
            )
        except ProviderUnavailable:
            return pd.DataFrame()
        return _documents_to_frame(documents)

    def _discover_remote(self, as_of_date: str, registry: SourceCredibilityRegistry) -> pd.DataFrame:
        index_urls: list[str] = []
        for profile in registry.by_source_type("policy"):
            index_urls.extend(profile.discovery_endpoints)
        if not index_urls:
            return pd.DataFrame()
        crawler = PublicWebCrawler(
            allow_network=self.allow_network,
            max_links_per_index=self.max_articles_per_source,
        )
        try:
            documents = crawler.discover_documents(
                index_urls,
                as_of_date=as_of_date,
                source_type="policy",
                source_reliability=0.85,
            )
        except ProviderUnavailable:
            return pd.DataFrame()
        return _documents_to_frame(documents)

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
        frame = frame.copy()
        frame["theme_candidates"] = themes
        return frame


def _documents_to_frame(documents: Iterable[object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for doc in documents:
        rows.append(
            {
                "source_name": getattr(doc, "source", ""),
                "url": getattr(doc, "url", ""),
                "title": getattr(doc, "title", ""),
                "body": getattr(doc, "body", ""),
                "published_at": getattr(doc, "published_at", ""),
                "available_at": getattr(doc, "available_at", ""),
                "raw_hash": getattr(doc, "content_hash", ""),
            }
        )
    return pd.DataFrame(rows)
