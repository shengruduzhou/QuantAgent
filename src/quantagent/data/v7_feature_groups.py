"""V7 feature-group taxonomy and PIT join helper.

The training pipeline reasons about features in groups: short-term flow,
medium-term momentum, long-horizon macro, fundamentals, valuation, risk,
liquidity, regime. ``select_v7_feature_columns`` resolves which columns
from a frame fall into which group, ``join_pit_features`` performs
strict PIT as-of merges across multiple auxiliary feature frames.

This module is intentionally kept apart from the existing V4-era
``FeatureStore`` in ``data/feature_store.py`` so we do not break the
legacy daily-research path; it only deals with feature *selection* and
*as-of merging*, not factor computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


SHORT_TERM_FEATURES: tuple[str, ...] = (
    "return_1d",
    "momentum_5d",
    "intraday_return",
    "volume_zscore_5d",
    "fund_flow_5d",
    "news_sentiment_score",
)
MEDIUM_TERM_FEATURES: tuple[str, ...] = (
    "momentum_20d",
    "volatility_20d",
    "sector_rotation_score",
    "earnings_revision_score",
    "amount_mean_20d",
    "volume_mean_20d",
)
LONG_HORIZON_FEATURES: tuple[str, ...] = (
    "policy_strength",
    "policy_chain_centrality_120d",
    "macro_industry_phase_120d",
    "macro_credit_cycle_120d",
    "macro_monetary_tailwind_120d",
    "structural_domestic_substitution_120d",
    "structural_bottleneck_120d",
    "exposure_score",
)
FUNDAMENTAL_FEATURES: tuple[str, ...] = (
    "revenue_growth",
    "net_income_growth",
    "gross_margin",
    "net_margin",
    "roe",
    "roa",
    "ocf_to_profit",
    "fcff",
    "quality_score",
    "growth_score",
)
VALUATION_FEATURES: tuple[str, ...] = (
    "pe_ttm",
    "pb",
    "ps_ttm",
    "ev_ebitda",
    "peg",
    "dividend_yield",
    "valuation_history_zscore_120d",
    "valuation_industry_zscore_120d",
    "valuation_margin_of_safety_120d",
)
RISK_FEATURES: tuple[str, ...] = (
    "fraud_risk_score",
    "management_risk_score",
    "debt_to_asset",
    "current_ratio",
    "quick_ratio",
    "regulatory_penalty_score",
    "audit_opinion_score",
)
LIQUIDITY_FEATURES: tuple[str, ...] = (
    "turnover_rate",
    "amount_mean_20d",
    "free_float_market_cap",
    "market_cap",
    "capacity_proxy_120d",
)
REGIME_FEATURES: tuple[str, ...] = (
    "market_regime_score",
    "volatility_regime",
    "liquidity_regime",
    "breadth_score",
    "drawdown_risk",
)


V7_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "short_term": SHORT_TERM_FEATURES,
    "medium_term": MEDIUM_TERM_FEATURES,
    "long_horizon": LONG_HORIZON_FEATURES,
    "fundamental": FUNDAMENTAL_FEATURES,
    "valuation": VALUATION_FEATURES,
    "risk": RISK_FEATURES,
    "liquidity": LIQUIDITY_FEATURES,
    "regime": REGIME_FEATURES,
}


@dataclass(frozen=True)
class V7FeatureSelection:
    selected: tuple[str, ...]
    group_to_columns: dict[str, tuple[str, ...]] = field(default_factory=dict)


def select_v7_feature_columns(
    frame: pd.DataFrame,
    groups: Iterable[str] = (),
    extra_columns: Iterable[str] = (),
) -> V7FeatureSelection:
    available = set(frame.columns) if frame is not None else set()
    target_groups = tuple(groups) or tuple(V7_FEATURE_GROUPS.keys())
    selected: list[str] = []
    seen: set[str] = set()
    group_map: dict[str, tuple[str, ...]] = {}
    for group in target_groups:
        if group not in V7_FEATURE_GROUPS:
            raise ValueError(f"unknown V7 feature group: {group}")
        present = tuple(c for c in V7_FEATURE_GROUPS[group] if c in available)
        group_map[group] = present
        for column in present:
            if column not in seen:
                seen.add(column)
                selected.append(column)
    for column in extra_columns:
        if column in available and column not in seen:
            seen.add(column)
            selected.append(column)
    return V7FeatureSelection(selected=tuple(selected), group_to_columns=group_map)


def join_pit_features(
    base: pd.DataFrame,
    extras: Iterable[pd.DataFrame],
    *,
    on: tuple[str, str] = ("symbol", "available_at"),
) -> pd.DataFrame:
    """Strict PIT as-of join across multiple auxiliary feature frames."""
    if base is None or base.empty:
        return base
    output = base.copy()
    output[on[1]] = pd.to_datetime(output[on[1]], errors="coerce")
    for extra in extras:
        if extra is None or extra.empty:
            continue
        missing = [column for column in on if column not in extra.columns]
        if missing:
            raise ValueError(f"extra frame missing PIT keys {missing}")
        right = extra.copy()
        right[on[1]] = pd.to_datetime(right[on[1]], errors="coerce")
        right = right.dropna(subset=list(on)).sort_values([on[0], on[1]])
        merged_parts: list[pd.DataFrame] = []
        for symbol, base_part in output.sort_values([on[0], on[1]]).groupby(on[0], sort=False):
            symbol_extra = right[right[on[0]].astype(str) == str(symbol)]
            if symbol_extra.empty:
                merged_parts.append(base_part)
                continue
            merged = pd.merge_asof(
                base_part.sort_values(on[1]),
                symbol_extra.drop(columns=[on[0]]).sort_values(on[1]),
                on=on[1],
                direction="backward",
            )
            merged_parts.append(merged)
        output = pd.concat(merged_parts, ignore_index=True, sort=False) if merged_parts else output
    return output


__all__ = [
    "V7_FEATURE_GROUPS",
    "V7FeatureSelection",
    "select_v7_feature_columns",
    "join_pit_features",
    "SHORT_TERM_FEATURES",
    "MEDIUM_TERM_FEATURES",
    "LONG_HORIZON_FEATURES",
    "FUNDAMENTAL_FEATURES",
    "VALUATION_FEATURES",
    "RISK_FEATURES",
    "LIQUIDITY_FEATURES",
    "REGIME_FEATURES",
]
