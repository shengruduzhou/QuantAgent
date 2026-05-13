from __future__ import annotations

from datetime import date, timedelta

from quantagent.v7.schemas import ThemeLifecycleStage
from quantagent.v7.scoring import classify_theme_lifecycle


def estimate_lifecycle(
    policy_strength: float,
    market_strength: float,
    fundamental_strength: float,
    capital_flow_strength: float,
    bubble_risk: float,
    crowding_score: float,
    invalidation_score: float = 0.0,
    trend_decay_score: float = 0.0,
) -> ThemeLifecycleStage:
    return classify_theme_lifecycle(
        policy_strength=policy_strength,
        market_strength=market_strength,
        fundamental_strength=fundamental_strength,
        capital_flow_strength=capital_flow_strength,
        bubble_risk=bubble_risk,
        crowding_score=crowding_score,
        invalidation_score=invalidation_score,
        trend_decay_score=trend_decay_score,
    )


def estimate_theme_expiry(as_of_date: str | date, lifecycle_stage: ThemeLifecycleStage, horizon_days: int) -> str:
    current = date.fromisoformat(str(as_of_date)[:10]) if not isinstance(as_of_date, date) else as_of_date
    multiplier = {
        ThemeLifecycleStage.POLICY_SEED: 1.0,
        ThemeLifecycleStage.NARRATIVE_FORMATION: 0.8,
        ThemeLifecycleStage.CAPITAL_INFLOW: 0.6,
        ThemeLifecycleStage.FUNDAMENTAL_VALIDATION: 0.8,
        ThemeLifecycleStage.EARNINGS_REALIZATION: 0.7,
        ThemeLifecycleStage.VALUATION_BUBBLE: 0.3,
        ThemeLifecycleStage.DIVERGENCE: 0.3,
        ThemeLifecycleStage.DECAY: 0.2,
        ThemeLifecycleStage.INVALIDATED: 0.0,
    }[lifecycle_stage]
    days = max(1, int(horizon_days * multiplier))
    return (current + timedelta(days=days)).isoformat()
