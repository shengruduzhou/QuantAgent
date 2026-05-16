"""News ingestor.

Pulls news from configured public pages (财新 / 华尔街见闻 / 新浪 / 东方财富
/ 雪球 etc.) into the unified evidence frame. The ingestor focuses on
classification — it never inflates confidence; downstream
:class:`NewsCrossValidator` will decide whether the news is corroborated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.credibility.news_cross_validator import cross_validate
from quantagent.data.ingestion.daily_evidence_job import (
    DailyEvidenceJobConfig,
    EvidenceIngestor,
    attach_source_profile,
)
from quantagent.data.ingestion.policy_ingestor import _documents_to_frame
from quantagent.data.ingestion.source_registry import SourceCredibilityRegistry
from quantagent.data.providers.base import ProviderUnavailable
from quantagent.data.providers.web_crawler import PublicWebCrawler


@dataclass
class NewsIngestor(EvidenceIngestor):
    name: str = "news"
    source_type: str = "news"
    allow_network: bool = False
    urls: tuple[str, ...] = ()
    active_discovery: bool = False
    max_articles_per_source: int = 25
    local_cache_root: str = field(default_factory=lambda: str(quant_paths().data_root / "v7" / "evidence" / "news"))
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
    security_theme_dictionary: dict[str, tuple[str, tuple[str, ...]]] = field(
        default_factory=lambda: {
            "600519.SH": ("Kweichow Moutai", ("consumer_recovery",)),
            "000858.SZ": ("Wuliangye", ("consumer_recovery",)),
            "300750.SZ": ("CATL", ("ev_supply_chain", "energy_storage")),
            "688981.SH": ("SMIC", ("semiconductor_domestic_substitution",)),
        }
    )
    industry_theme_ontology: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "semiconductor": ("semiconductor_domestic_substitution",),
            "integrated circuit": ("semiconductor_domestic_substitution",),
            "ai server": ("ai_compute",),
            "data center": ("ai_compute",),
            "energy storage": ("energy_storage",),
            "ev battery": ("ev_supply_chain",),
            "baijiu": ("consumer_recovery",),
        }
    )
    embedding_reranker: object | None = None

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
        local_frame = attach_source_profile(local_frame, registry)
        local_frame = self._tag_themes(local_frame)
        local_frame = self._apply_company_dictionary(local_frame)
        local_frame = self._apply_industry_ontology(local_frame)
        local_frame = self._apply_embedding_reranker(local_frame)
        local_frame = self._tag_rumor_risk(local_frame)
        local_frame = self._attach_cross_source_quality(local_frame)
        local_frame["source_type"] = "news"
        local_frame["event_type"] = local_frame.get("event_type", "sentiment_positive")
        local_frame["confidence"] = local_frame.get("confidence", 0.55)
        local_frame = self._enforce_low_reliability_news_cap(local_frame)
        local_frame = self._ensure_source_traceability(local_frame, config.as_of_date)
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
        return _documents_to_frame(documents)

    def _discover_remote(self, as_of_date: str, registry: SourceCredibilityRegistry) -> pd.DataFrame:
        index_urls: list[str] = []
        for profile in registry.by_source_type("news"):
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
                source_type="news",
                source_reliability=0.62,
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
        data = frame.copy()
        data["theme_candidates"] = themes
        return data

    def _apply_company_dictionary(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        data = frame.copy()
        text = (data.get("title", "").fillna("") + " " + data.get("body", "").fillna("")).str.lower()
        symbols: list[str] = []
        company_names: list[str] = []
        merged_themes: list[str] = []
        for index, body in enumerate(text):
            row = data.iloc[index]
            themes = _split_candidates(row.get("theme_candidates", ""))
            matched_symbols = _split_candidates(row.get("symbol", ""))
            matched_names = _split_candidates(row.get("company_name", ""))
            for symbol, (company_name, theme_names) in self.security_theme_dictionary.items():
                if symbol.lower() in body or company_name.lower() in body:
                    matched_symbols.append(symbol)
                    matched_names.append(company_name)
                    themes.extend(theme_names)
            symbols.append(",".join(sorted(set(matched_symbols))))
            company_names.append(",".join(sorted(set(matched_names))))
            merged_themes.append(",".join(sorted(set(themes))))
        data["symbol"] = symbols
        data["affected_symbols"] = symbols
        data["company_name"] = company_names
        data["theme_candidates"] = merged_themes
        return data

    def _apply_industry_ontology(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        data = frame.copy()
        text = (data.get("title", "").fillna("") + " " + data.get("body", "").fillna("")).str.lower()
        merged: list[str] = []
        for index, body in enumerate(text):
            themes = _split_candidates(data.iloc[index].get("theme_candidates", ""))
            for keyword, theme_names in self.industry_theme_ontology.items():
                if keyword.lower() in body:
                    themes.extend(theme_names)
            merged.append(",".join(sorted(set(themes))))
        data["theme_candidates"] = merged
        return data

    def _apply_embedding_reranker(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or self.embedding_reranker is None:
            return frame
        rerank = getattr(self.embedding_reranker, "rerank", None)
        if rerank is None:
            return frame
        data = frame.copy()
        try:
            reranked = rerank(data)
        except Exception:
            return data
        if isinstance(reranked, pd.DataFrame) and "theme_candidates" in reranked.columns:
            data["theme_candidates"] = reranked["theme_candidates"]
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
        data["confidence"] = data.get("confidence", 0.55)
        data.loc[data["rumor_risk_flag"], "confidence"] = data.loc[data["rumor_risk_flag"], "confidence"] * 0.55
        data["rumor_risk"] = data["rumor_risk_flag"].map(lambda flagged: 0.75 if flagged else 0.10)
        return data

    def _attach_cross_source_quality(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        data = frame.copy()
        data["cross_validation_count"] = data.get("cross_validation_count", 0)
        data["contradiction_count"] = data.get("contradiction_count", 0)
        summaries = cross_validate(data)
        for summary in summaries:
            symbol_mask = _series(data, "symbol", "").astype(str).str.contains(summary.symbol, regex=False)
            theme_mask = _series(data, "theme_candidates", "").astype(str).str.contains(summary.theme, regex=False)
            event_mask = _series(data, "event_type", "").astype(str).eq(summary.event_type)
            mask = symbol_mask & theme_mask & event_mask
            data.loc[mask, "cross_validation_count"] = summary.confirming_sources
            data.loc[mask, "contradiction_count"] = summary.contradiction_count
            data.loc[mask, "rumor_risk"] = summary.rumor_risk
        return data

    def _enforce_low_reliability_news_cap(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        data = frame.copy()
        reliability = pd.to_numeric(_series(data, "source_reliability", 0.40), errors="coerce").fillna(0.40)
        cross_count = pd.to_numeric(_series(data, "cross_validation_count", 0), errors="coerce").fillna(0)
        weak_single_source = (reliability < 0.65) & (cross_count < 2)
        data["core_pool_signal_allowed"] = ~weak_single_source
        data.loc[weak_single_source, "confidence"] = pd.to_numeric(
            data.loc[weak_single_source, "confidence"],
            errors="coerce",
        ).fillna(0.55).clip(upper=0.34)
        data.loc[weak_single_source, "risk_flags"] = data.loc[weak_single_source].apply(
            lambda row: _append_flag(row.get("risk_flags", ""), "single_low_reliability_source"),
            axis=1,
        )
        return data

    def _ensure_source_traceability(self, frame: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        data = frame.copy()
        if "available_at" not in data.columns or data["available_at"].isna().all():
            data["available_at"] = data.get("published_at", as_of_date)
        if "url" not in data.columns:
            data["url"] = ""
        if "raw_hash" not in data.columns or data["raw_hash"].isna().any():
            data["raw_hash"] = data.apply(
                lambda row: sha256(
                    f"{row.get('url', '')}\n{row.get('title', '')}\n{row.get('body', '')}".encode("utf-8")
                ).hexdigest(),
                axis=1,
            )
        return data


def _split_candidates(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _series(frame: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(default)
    return pd.Series([default] * len(frame), index=frame.index)


def _append_flag(existing: object, flag: str) -> str:
    flags = _split_candidates(existing)
    flags.append(flag)
    return ",".join(sorted(set(flags)))
