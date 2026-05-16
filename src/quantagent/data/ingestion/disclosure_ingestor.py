"""Exchange disclosure ingestor.

Pulls listed-company announcements (上交所 / 深交所 / 北交所 / 巨潮资讯)
into the unified evidence frame. Like the policy ingestor it defaults to
reading a local CSV cache and only hits the network when explicitly
allowed.

Each announcement is tagged with:
- ``symbol`` and ``company_name``,
- ``event_type`` (``order_confirmed`` / ``earnings_growth`` / ``regulatory``
  / ``shareholder_change`` / ``goodwill_impairment`` / ``pledge``),
- ``confidence`` derived from the document title + source authority,
- ``chain_node_candidates`` from keyword overlap.
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
from quantagent.data.ingestion.policy_ingestor import _documents_to_frame
from quantagent.data.ingestion.source_registry import SourceCredibilityRegistry
from quantagent.data.providers.base import ProviderUnavailable
from quantagent.data.providers.web_crawler import PublicWebCrawler


_DEFAULT_EVENT_PATTERNS = {
    "order_confirmed": (
        "中标",
        "中标公告",
        "合同",
        "签订",
        "签署",
        "重大合同",
        "订单",
        "签约",
    ),
    "earnings_growth": (
        "业绩预告",
        "业绩快报",
        "净利润",
        "营业收入",
        "year-on-year",
        "yoy",
    ),
    "regulatory_penalty": (
        "立案",
        "处罚",
        "警示函",
        "问询函",
        "纪律处分",
        "监管措施",
    ),
    "shareholder_change": (
        "减持",
        "增持",
        "回购",
        "股东",
        "持股",
    ),
    "goodwill_impairment": (
        "商誉减值",
        "资产减值",
        "计提",
        "减值准备",
    ),
    "pledge": (
        "质押",
        "解除质押",
        "股权质押",
    ),
    "audit_opinion": (
        "保留意见",
        "非标审计",
        "无法表示意见",
        "否定意见",
    ),
}


@dataclass
class DisclosureIngestor(EvidenceIngestor):
    name: str = "disclosure"
    source_type: str = "disclosure"
    allow_network: bool = False
    active_discovery: bool = False
    max_articles_per_source: int = 25
    local_cache_root: str = field(default_factory=lambda: str(quant_paths().data_root / "v7" / "evidence" / "disclosure"))
    chain_node_keyword_map: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "server": ("服务器", "整机"),
            "gpu": ("gpu", "图形处理器", "ai加速"),
            "pcb": ("pcb", "印制电路板"),
            "optical_module": ("光模块",),
            "hbm": ("hbm", "高带宽存储"),
            "foundry": ("晶圆代工",),
            "data_center": ("数据中心", "智算中心"),
            "liquid_cooling": ("液冷",),
            "energy_storage": ("储能",),
        }
    )

    def fetch(
        self,
        config: DailyEvidenceJobConfig,
        registry: SourceCredibilityRegistry,
    ) -> pd.DataFrame:
        frame = self._read_local_cache(config.as_of_date)
        if self.allow_network and self.active_discovery:
            discovered = self._discover_remote(config.as_of_date, registry)
            if not discovered.empty:
                frame = pd.concat([frame, discovered], ignore_index=True, sort=False)
        if frame.empty:
            return frame
        frame = self._tag_events(frame)
        frame = self._tag_chain_nodes(frame)
        frame = attach_source_profile(frame, registry)
        frame["source_type"] = "disclosure"
        # Exchange disclosures are usually post-close — push available_at to next day
        if "published_at" in frame.columns and "available_at" in frame.columns:
            future_only = frame["available_at"].isna() | (frame["available_at"] == frame["published_at"])
            shifted = (
                pd.to_datetime(frame.loc[future_only, "published_at"], errors="coerce")
                + pd.to_timedelta(max(1, config.available_lag_days), unit="D")
            )
            frame.loc[future_only, "available_at"] = shifted.dt.strftime("%Y-%m-%d")
        return frame

    def _discover_remote(self, as_of_date: str, registry: SourceCredibilityRegistry) -> pd.DataFrame:
        index_urls: list[str] = []
        for profile in registry.by_source_type("disclosure"):
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
                source_type="disclosure",
                source_reliability=0.92,
            )
        except ProviderUnavailable:
            return pd.DataFrame()
        return _documents_to_frame(documents)

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

    def _tag_events(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        title_lower = frame.get("title", "").fillna("").str.lower()
        body_lower = frame.get("body", "").fillna("").str.lower()
        combined = (title_lower + " " + body_lower).tolist()
        event_types: list[str] = []
        confidences: list[float] = []
        for text in combined:
            tag = "no_trade"
            confidence = 0.55
            for event_type, patterns in _DEFAULT_EVENT_PATTERNS.items():
                if any(pattern.lower() in text for pattern in patterns):
                    tag = event_type
                    confidence = 0.82 if event_type in {"regulatory_penalty", "audit_opinion"} else 0.78
                    break
            event_types.append(tag)
            confidences.append(confidence)
        data = frame.copy()
        data["event_type"] = event_types
        data["confidence"] = confidences
        return data

    def _tag_chain_nodes(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        text = (frame.get("title", "").fillna("") + " " + frame.get("body", "").fillna("")).str.lower().tolist()
        tagged: list[str] = []
        for body in text:
            nodes: list[str] = []
            for node_id, keywords in self.chain_node_keyword_map.items():
                if any(keyword.lower() in body for keyword in keywords):
                    nodes.append(node_id)
            tagged.append(",".join(sorted(set(nodes))))
        data = frame.copy()
        data["chain_node_candidates"] = tagged
        return data
