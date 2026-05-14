from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from quantagent.v7.schemas import ChainNode, ChainRelationType, FundamentalScore, ThematicUniverseMember, ThemeLifecycleStage, ThemeProfile, UniverseBucket
from quantagent.v7.scoring import classify_universe_bucket


def build_thematic_universe(
    base_universe: pd.DataFrame,
    company_theme_map: pd.DataFrame,
    theme_profiles: list[ThemeProfile],
    chain_nodes: list[ChainNode],
    fundamentals: Mapping[str, FundamentalScore] | None = None,
    market_state: pd.DataFrame | None = None,
    as_of_date: str = "",
) -> list[ThematicUniverseMember]:
    if base_universe.empty or company_theme_map.empty:
        return []
    fundamentals = fundamentals or {}
    theme_by_name = {profile.theme_name: profile for profile in theme_profiles}
    node_by_id = {node.node_id: node for node in chain_nodes}
    market_by_symbol = _index_by_symbol(market_state)
    rows = base_universe.merge(company_theme_map, on="symbol", how="inner", suffixes=("", "_mapping"))
    members: list[ThematicUniverseMember] = []
    for _, row in rows.iterrows():
        theme = str(row["theme"])
        profile = theme_by_name.get(theme)
        if profile is None:
            continue
        node_id = str(row.get("chain_node", ""))
        node = node_by_id.get(node_id)
        symbol = str(row["symbol"])
        fundamental = fundamentals.get(symbol)
        market = market_by_symbol.get(symbol, {})
        exposure_score = float(row.get("exposure_score", 0.0))
        if node is not None:
            exposure_score = max(exposure_score, 100.0 * max(node.dependency_strength, node.bottleneck_score, node.policy_support_score))
        liquidity_score = float(market.get("liquidity_score", row.get("liquidity_score", 50.0)))
        fraud_score = float(fundamental.fraud_risk_score if fundamental else row.get("fraud_risk_score", 50.0))
        fundamental_score = float(fundamental.fundamental_score if fundamental else row.get("fundamental_score", 50.0))
        valuation_score = float(fundamental.valuation_score if fundamental else row.get("valuation_score", 50.0))
        quality_score = float(fundamental.quality_score if fundamental else row.get("quality_score", 50.0))
        source_confidence = float(row.get("source_confidence", profile.theme_confidence))
        evidence_count = int(row.get("evidence_count", len(profile.key_evidence)))
        data_quality_flags: list[str] = []
        has_fundamental = fundamental is not None or _has_non_null(row, "fundamental_score")
        if not has_fundamental:
            fundamental_score = min(fundamental_score, 40.0)
            valuation_score = min(valuation_score, 40.0)
            quality_score = min(quality_score, 40.0)
            source_confidence = min(source_confidence, 0.60)
            data_quality_flags.append("missing_fundamentals_core_block")
        bucket = classify_universe_bucket(
            exposure_score=exposure_score,
            fundamental_score=fundamental_score,
            fraud_risk_score=fraud_score,
            liquidity_score=liquidity_score,
            source_confidence=source_confidence,
            evidence_count=evidence_count,
            valuation_score=valuation_score,
        )
        if not has_fundamental and bucket == UniverseBucket.CORE_BENEFICIARY:
            bucket = UniverseBucket.STRONG_CORRELATION
        if profile.lifecycle_stage in {ThemeLifecycleStage.DECAY, ThemeLifecycleStage.INVALIDATED}:
            bucket = UniverseBucket.WATCHLIST if bucket != UniverseBucket.EXCLUSION else bucket
            data_quality_flags.append("theme_lifecycle_not_active")
        members.append(
            ThematicUniverseMember(
                symbol=symbol,
                company_name=str(row.get("company_name", symbol)),
                theme=theme,
                sub_theme=str(row.get("sub_theme", node_id)),
                chain_node=node_id,
                exposure_type=_relation_type(row.get("exposure_type", "weak_association")),
                exposure_score=exposure_score,
                revenue_exposure_estimate=_optional_float(row.get("revenue_exposure_estimate")),
                profit_exposure_estimate=_optional_float(row.get("profit_exposure_estimate")),
                evidence_count=evidence_count,
                source_confidence=source_confidence,
                fundamental_score=fundamental_score,
                valuation_score=valuation_score,
                quality_score=quality_score,
                fraud_risk_score=fraud_score,
                liquidity_score=liquidity_score,
                market_attention_score=float(market.get("market_attention_score", row.get("market_attention_score", 50.0))),
                theme_lifecycle_stage=profile.lifecycle_stage,
                entry_date=str(row.get("entry_date", as_of_date)),
                expiry_date=profile.expiry_date,
                last_validated_at=str(row.get("last_validated_at", as_of_date)),
                watchlist_status=bucket,
                removal_reason=_removal_reason(bucket, fraud_score, liquidity_score, source_confidence),
                sector=_optional_str(row.get("sector")),
                industry=_optional_str(row.get("industry")),
                membership_ttl_days=_optional_int(row.get("membership_ttl_days"), profile.expected_horizon_days),
                validation_status="blocked" if bucket == UniverseBucket.EXCLUSION else "active",
                data_quality_flags=tuple(data_quality_flags),
            )
        )
    return sorted(members, key=lambda item: (item.theme, item.watchlist_status.value, -item.exposure_score, item.symbol))


def _index_by_symbol(frame: pd.DataFrame | None) -> dict[str, dict]:
    if frame is None or frame.empty:
        return {}
    return {str(row["symbol"]): row.to_dict() for _, row in frame.iterrows()}


def _relation_type(value: object) -> ChainRelationType:
    text = str(value)
    try:
        return ChainRelationType(text)
    except ValueError:
        return ChainRelationType.WEAK_ASSOCIATION


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_str(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    return text if text else None


def _has_non_null(row: pd.Series, column: str) -> bool:
    return column in row.index and row.get(column) is not None and not pd.isna(row.get(column))


def _optional_int(value: object, default: int) -> int:
    if value is None or pd.isna(value):
        return int(default)
    return int(value)


def _removal_reason(bucket: object, fraud_score: float, liquidity_score: float, source_confidence: float) -> str | None:
    if str(getattr(bucket, "value", bucket)) != "exclusion_pool":
        return None
    if fraud_score > 80.0:
        return "high_fraud_risk"
    if liquidity_score < 25.0:
        return "low_liquidity"
    if source_confidence < 0.25:
        return "low_source_confidence"
    return "weak_or_false_association"
