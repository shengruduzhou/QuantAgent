"""Source credibility lookup + weighting helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# Tier → numeric credibility band
SOURCE_TIER_TABLE: dict[str, float] = {
    "official": 1.00,            # 国务院、人大、证监会、央行、财政部、发改委等
    "state_media": 0.90,         # 新华社、人民日报、CCTV、中央电视台
    "top_financial": 0.85,       # 财新、21 世纪经济报道、经济观察报
    "mainstream_financial": 0.80,  # 证券时报、上海证券报、中国证券报
    "online_financial": 0.70,    # 财联社、华尔街见闻、第一财经
    "industry_data": 0.60,       # 同花顺、东方财富、雪球、Wind
    "social_media": 0.40,        # 公众号、微博、论坛
    "unknown": 0.50,
}


# Full source → tier table (substring matching; first hit wins)
SOURCE_CREDIBILITY_TABLE: dict[str, str] = {
    # official
    "国务院": "official",
    "证监会": "official",
    "中国证监会": "official",
    "央行": "official",
    "中国人民银行": "official",
    "财政部": "official",
    "发改委": "official",
    "国家发改委": "official",
    "审计署": "official",
    "市场监管总局": "official",
    "csrc.gov.cn": "official",
    "pbc.gov.cn": "official",
    "mof.gov.cn": "official",
    "ndrc.gov.cn": "official",
    "gov.cn": "official",

    # state media
    "新华社": "state_media",
    "人民日报": "state_media",
    "央视": "state_media",
    "CCTV": "state_media",
    "中央电视台": "state_media",
    "新华网": "state_media",
    "人民网": "state_media",
    "xinhuanet.com": "state_media",
    "people.com.cn": "state_media",

    # top financial
    "财新": "top_financial",
    "财新网": "top_financial",
    "21世纪经济报道": "top_financial",
    "21财经": "top_financial",
    "经济观察报": "top_financial",
    "经济观察网": "top_financial",
    "caixin.com": "top_financial",

    # mainstream financial
    "证券时报": "mainstream_financial",
    "中国证券报": "mainstream_financial",
    "上海证券报": "mainstream_financial",
    "中国经济网": "mainstream_financial",
    "经济日报": "mainstream_financial",
    "stcn.com": "mainstream_financial",
    "cs.com.cn": "mainstream_financial",
    "cnstock.com": "mainstream_financial",

    # online financial
    "财联社": "online_financial",
    "华尔街见闻": "online_financial",
    "第一财经": "online_financial",
    "界面新闻": "online_financial",
    "腾讯财经": "online_financial",
    "新浪财经": "online_financial",
    "cls.cn": "online_financial",
    "wallstreetcn.com": "online_financial",
    "yicai.com": "online_financial",
    "jiemian.com": "online_financial",
    "finance.sina.com.cn": "online_financial",

    # industry data
    "同花顺": "industry_data",
    "10jqka": "industry_data",
    "东方财富": "industry_data",
    "eastmoney": "industry_data",
    "雪球": "industry_data",
    "xueqiu": "industry_data",
    "Wind": "industry_data",
    "万得": "industry_data",
    "iFind": "industry_data",
    "Choice": "industry_data",

    # social
    "微博": "social_media",
    "公众号": "social_media",
    "知乎": "social_media",
    "论坛": "social_media",
    "贴吧": "social_media",
    "推特": "social_media",
    "twitter": "social_media",
    "telegram": "social_media",
}


def lookup_source_tier(source: str | None) -> str:
    """Return the tier name for a given source string.

    Substring match — the first key that appears anywhere in
    ``source`` wins. Falls back to ``"unknown"`` when no key matches
    or input is None/empty.
    """
    if source is None:
        return "unknown"
    s = str(source).strip()
    if not s:
        return "unknown"
    s_lower = s.lower()
    for keyword, tier in SOURCE_CREDIBILITY_TABLE.items():
        if keyword.lower() in s_lower:
            return tier
    return "unknown"


def lookup_source_credibility(source: str | None) -> float:
    """Convenience wrapper: tier → numeric credibility."""
    tier = lookup_source_tier(source)
    return float(SOURCE_TIER_TABLE.get(tier, SOURCE_TIER_TABLE["unknown"]))


def apply_credibility_column(
    events: pd.DataFrame,
    *,
    source_col: str = "source",
    out_col: str = "source_credibility",
    tier_col: str | None = "source_tier",
) -> pd.DataFrame:
    """Add a credibility column (and optionally a tier column) to a frame.

    Does NOT mutate the input.  Missing source values get the ``unknown``
    tier (0.50).
    """
    if events is None or events.empty:
        return events.copy() if events is not None else pd.DataFrame()
    if source_col not in events.columns:
        out = events.copy()
        out[out_col] = float(SOURCE_TIER_TABLE["unknown"])
        if tier_col is not None:
            out[tier_col] = "unknown"
        return out

    out = events.copy()
    tiers = out[source_col].map(lookup_source_tier)
    out[out_col] = tiers.map(SOURCE_TIER_TABLE).astype(float)
    if tier_col is not None:
        out[tier_col] = tiers
    return out


def apply_credibility_weight_to_strength(
    events: pd.DataFrame,
    *,
    source_col: str = "source",
    strength_col: str = "evidence_strength",
    out_col: str = "credibility_weighted_strength",
    clip_to: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """Multiply an existing strength column by the source's credibility.

    Useful for downweighting a "buy" signal that comes from a low-tier
    online forum vs. the same call coming from 财新 or 中信证券.
    """
    if events is None or events.empty:
        return events.copy() if events is not None else pd.DataFrame()
    if strength_col not in events.columns:
        raise ValueError(f"events frame missing strength column: {strength_col}")
    out = apply_credibility_column(events, source_col=source_col)
    weighted = pd.to_numeric(out[strength_col], errors="coerce").fillna(0.0) * out[
        "source_credibility"
    ]
    lo, hi = clip_to
    out[out_col] = weighted.clip(lower=lo, upper=hi).astype(float)
    return out
