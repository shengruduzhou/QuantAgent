from __future__ import annotations

from collections import defaultdict

import pandas as pd

from quantagent.themes.policy_parser import ParsedPolicyDocument, policy_to_evidence
from quantagent.themes.theme_lifecycle import estimate_lifecycle, estimate_theme_expiry
from quantagent.v7.schemas import EvidenceRecord, ThemeProfile
from quantagent.v7.scoring import theme_strength_score


def discover_themes(
    parsed_policies: list[ParsedPolicyDocument],
    as_of_date: str,
    market_theme_metrics: pd.DataFrame | None = None,
    extra_evidence: list[EvidenceRecord] | None = None,
) -> tuple[list[ThemeProfile], list[EvidenceRecord]]:
    evidence = [record for parsed in parsed_policies for record in policy_to_evidence(parsed, as_of_date)]
    evidence.extend(extra_evidence or [])
    market = _market_metrics(market_theme_metrics)
    by_theme: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in evidence:
        if record.theme:
            by_theme[record.theme].append(record)

    profiles: list[ThemeProfile] = []
    for theme, records in sorted(by_theme.items()):
        metrics = market.get(theme, {})
        policy_strength = _weighted_average([r.magnitude * r.confidence for r in records if r.source_type.value == "official_policy"])
        market_strength = float(metrics.get("market_strength", 0.35))
        fundamental_strength = float(metrics.get("industry_fundamental_strength", 0.30))
        capital_flow_strength = float(metrics.get("capital_flow_strength", 0.30))
        news_sentiment_strength = float(metrics.get("news_sentiment_strength", 0.30))
        bubble_risk = float(metrics.get("bubble_risk", 0.20))
        crowding_score = float(metrics.get("crowding_score", 0.25))
        invalidation_score = float(metrics.get("invalidation_score", 0.0))
        decay_score = float(metrics.get("trend_decay_score", 0.0))
        strength = theme_strength_score(
            policy_strength=policy_strength,
            market_strength=market_strength,
            industry_fundamental_strength=fundamental_strength,
            capital_flow_strength=capital_flow_strength,
            news_sentiment_strength=news_sentiment_strength,
            opposing_evidence_penalty=invalidation_score,
            bubble_risk=bubble_risk,
        )
        lifecycle = estimate_lifecycle(
            policy_strength,
            market_strength,
            fundamental_strength,
            capital_flow_strength,
            bubble_risk,
            crowding_score,
            invalidation_score,
            decay_score,
        )
        horizon = min(126, max(record.horizon_days for record in records))
        profiles.append(
            ThemeProfile(
                theme_name=theme,
                theme_category=str(metrics.get("theme_category", "policy_industry")),
                theme_strength=strength,
                policy_strength=policy_strength,
                market_strength=market_strength,
                industry_fundamental_strength=fundamental_strength,
                capital_flow_strength=capital_flow_strength,
                news_sentiment_strength=news_sentiment_strength,
                lifecycle_stage=lifecycle,
                expected_horizon_days=horizon,
                theme_confidence=min(1.0, _weighted_average([r.confidence for r in records]) + 0.05 * len(records)),
                bubble_risk=bubble_risk,
                crowding_score=crowding_score,
                expiry_date=estimate_theme_expiry(as_of_date, lifecycle, horizon),
                update_frequency="daily",
                key_evidence=tuple(r.evidence_id for r in records if r.direction >= 0),
                opposing_evidence=tuple(r.evidence_id for r in records if r.direction < 0 or "policy_constraint" in r.risk_flags),
                required_follow_up_data=_follow_up_data(lifecycle, fundamental_strength),
            )
        )
    return profiles, evidence


def _market_metrics(frame: pd.DataFrame | None) -> dict[str, dict[str, float | str]]:
    if frame is None or frame.empty:
        return {}
    return {str(row["theme"]): row.to_dict() for _, row in frame.iterrows()}


def _weighted_average(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _follow_up_data(lifecycle: object, fundamental_strength: float) -> tuple[str, ...]:
    required = []
    if fundamental_strength < 0.55:
        required.extend(["order_disclosure", "revenue_exposure", "capacity_release"])
    if str(lifecycle).endswith("VALUATION_BUBBLE"):
        required.append("valuation_percentile")
    return tuple(required)
