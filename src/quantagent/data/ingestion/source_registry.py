"""Source credibility registry for the V7 evidence ingestion layer.

Every URL or named source produced by an ingestor must resolve to a
:class:`SourceProfile` so downstream agents know:

* what authority tier the source belongs to (`OFFICIAL_PRIMARY` versus
  `OFFICIAL_SECONDARY`, etc.) — this drives the news-credibility gate and
  the policy-strength weighting;
* its baseline reliability score (0–1) used by the deterministic news
  cross-validator;
* the schedule on which it should be polled by the daily evidence job;
* whether the source is allowed to set ``available_at`` to the same day
  as the publication (only exchange/official sources can do that — media
  re-publication is delayed by one trading day so we never read tomorrow's
  story today).

The registry ships with a deterministic default list covering the most
common A-share public sources. Users can override or extend it from
`configs/v7.default.yaml` under ``ingestion.source_registry`` without
touching code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class SourceTier(str, Enum):
    OFFICIAL_PRIMARY = "official_primary"            # 国务院、证监会、交易所原文
    OFFICIAL_SECONDARY = "official_secondary"        # 部委、地方政府
    EXCHANGE_DISCLOSURE = "exchange_disclosure"      # 上交所、深交所公司公告
    REGULATORY_PENALTY = "regulatory_penalty"        # 证监会处罚、问询函、审计意见
    REGULATED_MEDIA = "regulated_media"              # 新华社、人民日报、央视、中国证券报
    TIER1_FINANCIAL_MEDIA = "tier1_financial_media"  # 财新、华尔街见闻、第一财经
    TIER2_FINANCIAL_MEDIA = "tier2_financial_media"  # 东方财富、新浪财经、雪球（编辑后）
    INDUSTRY_MEDIA = "industry_media"                # 集微网、半导体行业观察等垂直媒体
    SELF_MEDIA = "self_media"                        # 公众号、自媒体、博客
    SOCIAL_MEDIA = "social_media"                    # 论坛、微博、知乎、小红书
    ANALYST_REPORT = "analyst_report"                # 券商研报
    DATA_VENDOR = "data_vendor"                      # TuShare / AkShare / Wind 等数据接口
    COMPANY_OFFICIAL = "company_official"            # 上市公司官网、互动易


_TIER_BASELINE: dict[SourceTier, float] = {
    SourceTier.OFFICIAL_PRIMARY: 0.95,
    SourceTier.OFFICIAL_SECONDARY: 0.88,
    SourceTier.EXCHANGE_DISCLOSURE: 0.92,
    SourceTier.REGULATORY_PENALTY: 0.92,
    SourceTier.REGULATED_MEDIA: 0.82,
    SourceTier.TIER1_FINANCIAL_MEDIA: 0.75,
    SourceTier.TIER2_FINANCIAL_MEDIA: 0.62,
    SourceTier.INDUSTRY_MEDIA: 0.58,
    SourceTier.SELF_MEDIA: 0.32,
    SourceTier.SOCIAL_MEDIA: 0.22,
    SourceTier.ANALYST_REPORT: 0.65,
    SourceTier.DATA_VENDOR: 0.85,
    SourceTier.COMPANY_OFFICIAL: 0.78,
}


_DEFAULT_PROFILES: tuple["SourceProfile", ...] = ()


@dataclass(frozen=True)
class SourceProfile:
    name: str
    host_or_id: str
    tier: SourceTier
    is_primary: bool
    is_official: bool
    poll_minutes: int = 1440
    allow_same_day_available_at: bool = False
    source_type: str = "news"
    reliability_override: float | None = None
    aliases: tuple[str, ...] = ()

    @property
    def reliability(self) -> float:
        if self.reliability_override is not None:
            return float(self.reliability_override)
        return _TIER_BASELINE.get(self.tier, 0.40)

    def authority_score(self) -> float:
        if self.tier == SourceTier.OFFICIAL_PRIMARY:
            return 0.95
        if self.tier == SourceTier.OFFICIAL_SECONDARY:
            return 0.88
        if self.tier == SourceTier.EXCHANGE_DISCLOSURE:
            return 0.90
        if self.tier == SourceTier.REGULATORY_PENALTY:
            return 0.90
        if self.tier == SourceTier.REGULATED_MEDIA:
            return 0.78
        if self.tier == SourceTier.TIER1_FINANCIAL_MEDIA:
            return 0.72
        if self.tier == SourceTier.TIER2_FINANCIAL_MEDIA:
            return 0.55
        if self.tier == SourceTier.INDUSTRY_MEDIA:
            return 0.55
        if self.tier == SourceTier.ANALYST_REPORT:
            return 0.62
        if self.tier == SourceTier.DATA_VENDOR:
            return 0.80
        if self.tier == SourceTier.COMPANY_OFFICIAL:
            return 0.70
        return 0.30


@dataclass
class SourceCredibilityRegistry:
    profiles: list[SourceProfile] = field(default_factory=lambda: list(_default_profiles()))

    def register(self, profile: SourceProfile) -> None:
        self.profiles.append(profile)

    def lookup(self, name_or_host: str) -> SourceProfile | None:
        if not name_or_host:
            return None
        text = name_or_host.lower().strip()
        for profile in self.profiles:
            if profile.host_or_id.lower() == text or profile.name.lower() == text:
                return profile
            if any(alias.lower() == text for alias in profile.aliases):
                return profile
        for profile in self.profiles:
            if profile.host_or_id and profile.host_or_id.lower() in text:
                return profile
        return None

    def resolve(self, name_or_host: str, default_tier: SourceTier = SourceTier.SELF_MEDIA) -> SourceProfile:
        profile = self.lookup(name_or_host)
        if profile is not None:
            return profile
        return SourceProfile(
            name=name_or_host or "unknown",
            host_or_id=name_or_host or "unknown",
            tier=default_tier,
            is_primary=False,
            is_official=False,
            source_type="news",
        )

    def primary_sources(self) -> list[SourceProfile]:
        return [profile for profile in self.profiles if profile.is_primary]

    def official_sources(self) -> list[SourceProfile]:
        return [profile for profile in self.profiles if profile.is_official]

    def by_source_type(self, source_type: str) -> list[SourceProfile]:
        return [profile for profile in self.profiles if profile.source_type == source_type]


def merge_user_profiles(
    registry: SourceCredibilityRegistry,
    overrides: Iterable[dict] | None,
) -> SourceCredibilityRegistry:
    """Apply user overrides (typically from configs/v7.default.yaml) onto a registry."""

    if not overrides:
        return registry
    for entry in overrides:
        try:
            profile = SourceProfile(
                name=str(entry["name"]),
                host_or_id=str(entry.get("host_or_id", entry["name"])),
                tier=SourceTier(entry.get("tier", SourceTier.SELF_MEDIA.value)),
                is_primary=bool(entry.get("is_primary", False)),
                is_official=bool(entry.get("is_official", False)),
                poll_minutes=int(entry.get("poll_minutes", 1440)),
                allow_same_day_available_at=bool(entry.get("allow_same_day_available_at", False)),
                source_type=str(entry.get("source_type", "news")),
                reliability_override=(
                    float(entry["reliability_override"]) if "reliability_override" in entry else None
                ),
                aliases=tuple(str(item) for item in entry.get("aliases", ())),
            )
        except (KeyError, ValueError):
            continue
        registry.register(profile)
    return registry


def _default_profiles() -> tuple[SourceProfile, ...]:
    return (
        # 国务院 / 部委 / 央行
        SourceProfile("gov.cn", "www.gov.cn", SourceTier.OFFICIAL_PRIMARY, is_primary=True, is_official=True, poll_minutes=60, allow_same_day_available_at=True, source_type="policy", aliases=("国务院",)),
        SourceProfile("ndrc.gov.cn", "www.ndrc.gov.cn", SourceTier.OFFICIAL_SECONDARY, is_primary=True, is_official=True, source_type="policy", aliases=("发改委",)),
        SourceProfile("miit.gov.cn", "www.miit.gov.cn", SourceTier.OFFICIAL_SECONDARY, is_primary=True, is_official=True, source_type="policy", aliases=("工信部",)),
        SourceProfile("most.gov.cn", "www.most.gov.cn", SourceTier.OFFICIAL_SECONDARY, is_primary=True, is_official=True, source_type="policy", aliases=("科技部",)),
        SourceProfile("mof.gov.cn", "www.mof.gov.cn", SourceTier.OFFICIAL_SECONDARY, is_primary=True, is_official=True, source_type="policy", aliases=("财政部",)),
        SourceProfile("pbc.gov.cn", "www.pbc.gov.cn", SourceTier.OFFICIAL_SECONDARY, is_primary=True, is_official=True, source_type="policy", aliases=("人民银行",)),
        # 监管处罚 / 交易所披露
        SourceProfile("csrc.gov.cn", "www.csrc.gov.cn", SourceTier.REGULATORY_PENALTY, is_primary=True, is_official=True, source_type="regulatory", aliases=("证监会",)),
        SourceProfile("sse.com.cn", "www.sse.com.cn", SourceTier.EXCHANGE_DISCLOSURE, is_primary=True, is_official=True, allow_same_day_available_at=False, source_type="disclosure", aliases=("上交所",)),
        SourceProfile("szse.cn", "www.szse.cn", SourceTier.EXCHANGE_DISCLOSURE, is_primary=True, is_official=True, source_type="disclosure", aliases=("深交所",)),
        SourceProfile("bse.cn", "www.bse.cn", SourceTier.EXCHANGE_DISCLOSURE, is_primary=True, is_official=True, source_type="disclosure", aliases=("北交所",)),
        SourceProfile("cninfo.com.cn", "www.cninfo.com.cn", SourceTier.EXCHANGE_DISCLOSURE, is_primary=True, is_official=True, source_type="disclosure", aliases=("巨潮资讯",)),
        # 监管媒体
        SourceProfile("xinhua", "www.xinhuanet.com", SourceTier.REGULATED_MEDIA, is_primary=False, is_official=True, source_type="news", aliases=("新华社",)),
        SourceProfile("people", "www.people.com.cn", SourceTier.REGULATED_MEDIA, is_primary=False, is_official=True, source_type="news", aliases=("人民日报",)),
        SourceProfile("cs.com.cn", "www.cs.com.cn", SourceTier.REGULATED_MEDIA, is_primary=False, is_official=True, source_type="news", aliases=("中国证券报",)),
        # Tier 1 财经媒体
        SourceProfile("caixin", "www.caixin.com", SourceTier.TIER1_FINANCIAL_MEDIA, is_primary=False, is_official=False, source_type="news", aliases=("财新",)),
        SourceProfile("wallstreetcn", "wallstreetcn.com", SourceTier.TIER1_FINANCIAL_MEDIA, is_primary=False, is_official=False, source_type="news", aliases=("华尔街见闻",)),
        SourceProfile("yicai", "www.yicai.com", SourceTier.TIER1_FINANCIAL_MEDIA, is_primary=False, is_official=False, source_type="news", aliases=("第一财经",)),
        # Tier 2 财经媒体
        SourceProfile("eastmoney", "www.eastmoney.com", SourceTier.TIER2_FINANCIAL_MEDIA, is_primary=False, is_official=False, source_type="news", aliases=("东方财富",)),
        SourceProfile("sina_finance", "finance.sina.com.cn", SourceTier.TIER2_FINANCIAL_MEDIA, is_primary=False, is_official=False, source_type="news", aliases=("新浪财经",)),
        SourceProfile("tencent_finance", "stock.qq.com", SourceTier.TIER2_FINANCIAL_MEDIA, is_primary=False, is_official=False, source_type="news", aliases=("腾讯财经",)),
        SourceProfile("xueqiu", "xueqiu.com", SourceTier.TIER2_FINANCIAL_MEDIA, is_primary=False, is_official=False, source_type="news", aliases=("雪球",)),
        # 数据接口
        SourceProfile("tushare", "tushare.pro", SourceTier.DATA_VENDOR, is_primary=False, is_official=False, source_type="financial", aliases=("tushare_pro",)),
        SourceProfile("akshare", "akshare.akfamily.xyz", SourceTier.DATA_VENDOR, is_primary=False, is_official=False, source_type="financial"),
    )
