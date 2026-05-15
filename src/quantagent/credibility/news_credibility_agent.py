from __future__ import annotations

from hashlib import sha1

import numpy as np
import pandas as pd

from quantagent.v7.schemas import EvidenceRecord, EventType, NewsCredibilityScore, SourceType
from quantagent.v7.scoring import news_confidence_score


SOURCE_RELIABILITY = {
    "company_announcement": 0.90,
    "exchange_disclosure": 0.92,
    "official_policy": 0.88,
    "mainstream_media": 0.72,
    "industry_media": 0.62,
    "research_report": 0.58,
    "social_media": 0.25,
    "rumor": 0.15,
}


def score_news_credibility(news: pd.DataFrame) -> list[NewsCredibilityScore]:
    """Score news credibility before sentiment can influence alpha."""
    if news.empty:
        return []
    duplicates = _duplicate_counts(news)
    scores: list[NewsCredibilityScore] = []
    for index, row in news.iterrows():
        source_type = str(row.get("source_type", row.get("source", "news"))).lower()
        reliability = float(row.get("source_reliability", SOURCE_RELIABILITY.get(source_type, 0.40)))
        primary = bool(row.get("is_primary_source", source_type in {"company_announcement", "exchange_disclosure", "official_policy"}))
        official = bool(row.get("is_official", source_type in {"company_announcement", "exchange_disclosure", "official_policy"}))
        contradiction_count = int(row.get("contradiction_count", 0))
        rumor_risk = float(row.get("rumor_risk", 0.7 if source_type in {"rumor", "social_media"} else 0.1))
        duplicate_penalty = min(0.20, max(0, duplicates.get(_fingerprint(row), 1) - 1) * 0.03)
        cross_validation = int(row.get("cross_validation_count", 1 if primary else 0))
        confidence = max(
            0.0,
            news_confidence_score(
                reliability,
                primary,
                official,
                cross_validation,
                contradiction_count,
                rumor_risk,
            )
            - duplicate_penalty,
        )
        sentiment = float(row.get("sentiment_score", _lexicon_sentiment(str(row.get("title", "")) + " " + str(row.get("summary", "")))))
        scores.append(
            NewsCredibilityScore(
                news_id=str(row.get("news_id", sha1(_fingerprint(row).encode("utf-8")).hexdigest())),
                source=str(row.get("source", source_type)),
                source_reliability=reliability,
                is_primary_source=primary,
                is_official=official,
                cross_validation_count=cross_validation,
                event_type=_event_type(str(row.get("event_type", "")), sentiment),
                affected_symbols=tuple(str(row.get("symbol")).split(",")) if row.get("symbol") is not None and not pd.isna(row.get("symbol")) else (),
                affected_theme=str(row.get("theme")) if row.get("theme") is not None and not pd.isna(row.get("theme")) else None,
                sentiment_score=sentiment,
                fundamental_impact_score=float(row.get("fundamental_impact_score", confidence * max(sentiment, 0.0))),
                short_term_impact_score=float(row.get("short_term_impact_score", confidence * abs(sentiment))),
                medium_term_impact_score=float(row.get("medium_term_impact_score", confidence * max(sentiment, 0.0) * (1.0 if primary else 0.4))),
                confidence=confidence,
                decay_half_life=float(row.get("decay_half_life", 3.0 if not primary else 20.0)),
                horizon_days=int(row.get("horizon_days", 5 if not primary else 60)),
                contradiction_flags=tuple(str(row.get("contradiction_flags", "")).split(",")) if row.get("contradiction_flags") else (),
                rumor_risk=rumor_risk,
                rationale=f"source_type={source_type}, primary={primary}, official={official}, duplicate_penalty={duplicate_penalty:.2f}",
            )
        )
    return scores


def news_scores_to_evidence(scores: list[NewsCredibilityScore], as_of_date: str) -> list[EvidenceRecord]:
    """Convert credible news scores into short/medium horizon V7 evidence."""
    records: list[EvidenceRecord] = []
    for score in scores:
        if score.confidence < 0.35:
            continue
        affected_symbols = score.affected_symbols or (None,)
        for symbol in affected_symbols:
            evidence_id = f"news:{score.news_id}:{symbol or score.affected_theme or 'market'}"
            records.append(
                EvidenceRecord(
                    evidence_id=evidence_id,
                    source=score.source,
                    source_type=_source_type(score),
                    source_authority_level=score.source_reliability,
                    timestamp=as_of_date,
                    published_at=as_of_date,
                    available_at=as_of_date,
                    symbol=symbol,
                    theme=score.affected_theme,
                    event_type=score.event_type,
                    direction=1.0 if score.sentiment_score >= 0 else -1.0,
                    magnitude=min(1.0, abs(score.sentiment_score) + score.fundamental_impact_score),
                    confidence=score.confidence,
                    evidence_quality=score.confidence,
                    source_reliability=score.source_reliability,
                    cross_validation_count=score.cross_validation_count,
                    decay_half_life=score.decay_half_life,
                    horizon_days=score.horizon_days,
                    rationale=score.rationale,
                    raw_reference={"news_id": score.news_id, "rumor_risk": score.rumor_risk},
                    raw_hash=score.news_id,
                    point_in_time_valid=True,
                    risk_flags=score.contradiction_flags + (("rumor_risk",) if score.rumor_risk >= 0.6 else ()),
                ).with_hash()
            )
    return records


def _duplicate_counts(news: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, row in news.iterrows():
        key = _fingerprint(row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _fingerprint(row: pd.Series) -> str:
    text = f"{row.get('title', '')} {row.get('summary', '')}".lower()
    return " ".join(text.split())[:160]


def _lexicon_sentiment(text: str) -> float:
    lower = text.lower()
    positive = sum(lower.count(term) for term in ("support", "growth", "contract", "order", "approval", "subsidy", "利好", "订单", "补贴"))
    negative = sum(lower.count(term) for term in ("risk", "penalty", "fraud", "probe", "loss", "监管", "处罚", "造假", "亏损"))
    return float(np.tanh((positive - negative) / 2.0))


def _event_type(value: str, sentiment: float) -> EventType:
    try:
        return EventType(value)
    except ValueError:
        return EventType.SENTIMENT_POSITIVE if sentiment >= 0 else EventType.SENTIMENT_NEGATIVE


def _source_type(score: NewsCredibilityScore) -> SourceType:
    if score.is_official:
        return SourceType.COMPANY_ANNOUNCEMENT if score.affected_symbols else SourceType.OFFICIAL_POLICY
    if score.source_reliability <= 0.30:
        return SourceType.SOCIAL_MEDIA
    return SourceType.NEWS
