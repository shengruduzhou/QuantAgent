"""AI orchestration layer that drives the V7 workflow with LLM skills.

The user-facing requirement is explicit: every decision that today reads from a
hardcoded keyword table or a hand-written heuristic (theme discovery from
policy text, industry-chain construction, news credibility, valuation, fraud
forensics, sentiment, economic analysis) must be derivable by an AI. This
module is the single seam where those skill calls are made.

Behaviour
---------
* When ``llm_skills.enabled`` is ``true`` and ``allow_network`` is ``true`` and
  a per-skill flag is set, the orchestrator invokes the corresponding
  ``LLMSkillClient`` skill with the curated prompt from ``agents.skills``.
* When any of those gates is off, the orchestrator returns a deterministic
  fallback that is **vocabulary-free**: it does not pretend to know which
  industry a policy targets. Instead it surfaces token frequency, source
  authority, and structural co-occurrence so the rest of the pipeline can
  still operate without ever needing a hand-maintained industry list.

The orchestrator never emits orders. Every method returns ``Decision``
objects with ``used_llm``/``fallback_reason`` audit fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from collections import Counter
from typing import Any, Iterable

import pandas as pd

from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig, LLMSkillResult
from quantagent.agents.skills import (
    ECONOMICS_AGENT,
    FINANCIAL_FORENSICS,
    NEWS_CREDIBILITY,
    POLICY_ANALYST,
    SENTIMENT_AGENT,
    VALUATION_AGENT,
    SkillPrompt,
)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThemeExtraction:
    """One policy → many themes; each theme carries chain nodes + policy strength."""

    theme: str
    sub_theme: str | None
    chain_nodes: tuple[str, ...]
    supported_sectors: tuple[str, ...]
    policy_strength: float
    binding: str
    horizon_days: int
    direction: float
    risk_flags: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class PolicyAnalysis:
    document_id: str
    source_authority: float
    effective_start_date: str | None
    effective_end_date: str | None
    themes: tuple[ThemeExtraction, ...]
    used_llm: bool
    fallback_reason: str | None


@dataclass(frozen=True)
class NewsCredibilityAIScore:
    news_id: str
    source_reliability: float
    is_primary_source: bool
    is_official: bool
    cross_validation_count: int
    event_type: str
    affected_symbols: tuple[str, ...]
    affected_theme: str | None
    sentiment_score: float
    short_term_impact: float
    medium_term_impact: float
    fundamental_impact: float
    decay_half_life: float
    horizon_days: int
    rumor_risk: float
    confidence: float
    rationale: str
    used_llm: bool


@dataclass(frozen=True)
class SentimentAIResult:
    scope: str
    retail_sentiment: float
    institutional_sentiment: float
    sentiment_divergence: float
    coordinated_pumping_risk: float
    short_squeeze_risk: float
    attention_surge_score: float
    social_volume_zscore: float
    rationale: str
    used_llm: bool


@dataclass(frozen=True)
class ValuationOverlay:
    symbol: str
    fair_value_per_share: float | None
    margin_of_safety_pct: float
    valuation_score: float
    bubble_risk_score: float
    investment_horizon_days: int
    method_weights: dict[str, float]
    key_assumptions: dict[str, float]
    rationale: str
    used_llm: bool


@dataclass(frozen=True)
class ForensicsOverlay:
    symbol: str
    fraud_risk_score: float
    confidence_in_reported_numbers: float
    key_red_flags: tuple[str, ...]
    accruals_quality_score: float
    cashflow_quality_score: float
    receivables_risk_score: float
    inventory_risk_score: float
    related_party_risk_score: float
    audit_opinion_score: float
    earnings_quality_score: float
    recent_restatement: bool
    rationale: str
    used_llm: bool


@dataclass(frozen=True)
class EconomicsOverlay:
    industry: str
    industry_cycle_stage: str
    supply_demand_balance: float
    pricing_power: float
    capacity_utilization: float
    inventory_days_zscore: float
    capex_intensity_trend: float
    credit_impulse_alignment: float
    monetary_tailwind: float
    fx_pressure: float
    commodity_cost_pressure: float
    policy_support_strength: float
    expected_industry_revenue_growth_yoy: float
    expected_horizon_days: int
    economic_thesis: str
    rationale: str
    used_llm: bool


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillToggles:
    policy_analyst: bool = True
    industry_chain_reasoner: bool = True
    news_credibility_agent: bool = True
    sentiment_agent: bool = False
    valuation_agent: bool = False
    financial_forensics_agent: bool = False
    economics_agent: bool = False


class LLMOrchestrator:
    """Coordinates LLM skill calls for the V7 pipeline."""

    def __init__(
        self,
        client: LLMSkillClient | None = None,
        toggles: SkillToggles | dict[str, bool] | None = None,
    ) -> None:
        self.client = client or LLMSkillClient(LLMSkillConfig())
        self.toggles = _coerce_toggles(toggles)

    # ------------------------------------------------------------------
    # Skill: policy_analyst
    # ------------------------------------------------------------------

    def analyze_policies(self, documents: Iterable[Any]) -> list[PolicyAnalysis]:
        analyses: list[PolicyAnalysis] = []
        for document in documents:
            user_text = _policy_user_text(document)
            authority = _source_authority(getattr(document, "source_level", ""))
            if self.toggles.policy_analyst and self.client.config.enabled:
                result = self.client.invoke(
                    POLICY_ANALYST.name,
                    system_prompt=POLICY_ANALYST.system_prompt,
                    user_text=user_text,
                    fallback=POLICY_ANALYST.fallback_shape,
                )
                if not result.used_fallback:
                    themes = _policy_themes_from_llm(result.output)
                    analyses.append(
                        PolicyAnalysis(
                            document_id=str(getattr(document, "document_id", "")),
                            source_authority=float(result.output.get("source_authority", authority) or authority),
                            effective_start_date=_optional_str(result.output.get("effective_start_date")),
                            effective_end_date=_optional_str(result.output.get("effective_end_date")),
                            themes=themes,
                            used_llm=True,
                            fallback_reason=None,
                        )
                    )
                    continue
                fallback_reason = result.fallback_reason
            else:
                fallback_reason = "disabled"
            themes = _policy_themes_fallback(document, authority)
            analyses.append(
                PolicyAnalysis(
                    document_id=str(getattr(document, "document_id", "")),
                    source_authority=authority,
                    effective_start_date=getattr(document, "effective_start_date", None),
                    effective_end_date=getattr(document, "effective_end_date", None),
                    themes=themes,
                    used_llm=False,
                    fallback_reason=fallback_reason,
                )
            )
        return analyses

    # ------------------------------------------------------------------
    # Skill: news_credibility_agent
    # ------------------------------------------------------------------

    def score_news_batch(self, news: pd.DataFrame) -> list[NewsCredibilityAIScore]:
        if news is None or news.empty:
            return []
        scores: list[NewsCredibilityAIScore] = []
        use_llm = self.toggles.news_credibility_agent and self.client.config.enabled
        for _, row in news.iterrows():
            payload = _news_user_text(row)
            ai = None
            if use_llm:
                ai_result = self.client.invoke(
                    NEWS_CREDIBILITY.name,
                    system_prompt=NEWS_CREDIBILITY.system_prompt,
                    user_text=payload,
                    fallback=NEWS_CREDIBILITY.fallback_shape,
                )
                if not ai_result.used_fallback:
                    ai = ai_result.output
            scores.append(_news_score_from_dict(row, ai))
        return scores

    # ------------------------------------------------------------------
    # Skill: sentiment_agent
    # ------------------------------------------------------------------

    def assess_sentiment(self, scope: str, text_blob: str) -> SentimentAIResult:
        if not text_blob:
            return _sentiment_fallback(scope, "empty_input")
        if not (self.toggles.sentiment_agent and self.client.config.enabled):
            return _sentiment_fallback(scope, "disabled")
        ai_result = self.client.invoke(
            SENTIMENT_AGENT.name,
            system_prompt=SENTIMENT_AGENT.system_prompt,
            user_text=text_blob,
            fallback=SENTIMENT_AGENT.fallback_shape,
        )
        if ai_result.used_fallback:
            return _sentiment_fallback(scope, ai_result.fallback_reason or "fallback")
        out = ai_result.output
        return SentimentAIResult(
            scope=scope,
            retail_sentiment=_clamp(out.get("retail_sentiment"), -1.0, 1.0),
            institutional_sentiment=_clamp(out.get("institutional_sentiment"), -1.0, 1.0),
            sentiment_divergence=_clamp(out.get("sentiment_divergence"), -2.0, 2.0),
            coordinated_pumping_risk=_clamp(out.get("coordinated_pumping_risk"), 0.0, 1.0),
            short_squeeze_risk=_clamp(out.get("short_squeeze_risk"), 0.0, 1.0),
            attention_surge_score=_clamp(out.get("attention_surge_score"), 0.0, 1.0),
            social_volume_zscore=_to_float(out.get("social_volume_zscore"), 0.0),
            rationale=str(out.get("rationale", "")),
            used_llm=True,
        )

    # ------------------------------------------------------------------
    # Skill: valuation_agent
    # ------------------------------------------------------------------

    def overlay_valuation(self, symbol: str, financials_blob: str) -> ValuationOverlay | None:
        if not (self.toggles.valuation_agent and self.client.config.enabled) or not financials_blob:
            return None
        ai_result = self.client.invoke(
            VALUATION_AGENT.name,
            system_prompt=VALUATION_AGENT.system_prompt,
            user_text=financials_blob,
            fallback=VALUATION_AGENT.fallback_shape,
        )
        if ai_result.used_fallback:
            return None
        out = ai_result.output
        return ValuationOverlay(
            symbol=symbol,
            fair_value_per_share=_to_optional_float(out.get("fair_value_per_share")),
            margin_of_safety_pct=_to_float(out.get("margin_of_safety_pct"), 0.0),
            valuation_score=_clamp(out.get("valuation_score"), 0.0, 100.0),
            bubble_risk_score=_clamp(out.get("bubble_risk_score"), 0.0, 1.0),
            investment_horizon_days=int(out.get("investment_horizon_days", 120) or 120),
            method_weights={str(key): float(value) for key, value in (out.get("method_weights") or {}).items()},
            key_assumptions={str(key): _to_float(value, 0.0) for key, value in (out.get("key_assumptions") or {}).items()},
            rationale=str(out.get("rationale", "")),
            used_llm=True,
        )

    # ------------------------------------------------------------------
    # Skill: financial_forensics_agent
    # ------------------------------------------------------------------

    def overlay_forensics(self, symbol: str, financials_blob: str) -> ForensicsOverlay | None:
        if not (self.toggles.financial_forensics_agent and self.client.config.enabled) or not financials_blob:
            return None
        ai_result = self.client.invoke(
            FINANCIAL_FORENSICS.name,
            system_prompt=FINANCIAL_FORENSICS.system_prompt,
            user_text=financials_blob,
            fallback=FINANCIAL_FORENSICS.fallback_shape,
        )
        if ai_result.used_fallback:
            return None
        out = ai_result.output
        return ForensicsOverlay(
            symbol=symbol,
            fraud_risk_score=_clamp(out.get("fraud_risk_score"), 0.0, 100.0),
            confidence_in_reported_numbers=_clamp(out.get("confidence_in_reported_numbers"), 0.0, 1.0),
            key_red_flags=tuple(str(flag) for flag in out.get("key_red_flags", ()) if str(flag)),
            accruals_quality_score=_clamp(out.get("accruals_quality_score"), 0.0, 100.0),
            cashflow_quality_score=_clamp(out.get("cashflow_quality_score"), 0.0, 100.0),
            receivables_risk_score=_clamp(out.get("receivables_risk_score"), 0.0, 100.0),
            inventory_risk_score=_clamp(out.get("inventory_risk_score"), 0.0, 100.0),
            related_party_risk_score=_clamp(out.get("related_party_risk_score"), 0.0, 100.0),
            audit_opinion_score=_clamp(out.get("audit_opinion_score"), 0.0, 100.0),
            earnings_quality_score=_clamp(out.get("earnings_quality_score"), 0.0, 100.0),
            recent_restatement=bool(out.get("recent_restatement", False)),
            rationale=str(out.get("rationale", "")),
            used_llm=True,
        )

    # ------------------------------------------------------------------
    # Skill: economics_agent
    # ------------------------------------------------------------------

    def overlay_economics(self, industry: str, economics_blob: str) -> EconomicsOverlay | None:
        if not (self.toggles.economics_agent and self.client.config.enabled) or not economics_blob:
            return None
        ai_result = self.client.invoke(
            ECONOMICS_AGENT.name,
            system_prompt=ECONOMICS_AGENT.system_prompt,
            user_text=economics_blob,
            fallback=ECONOMICS_AGENT.fallback_shape,
        )
        if ai_result.used_fallback:
            return None
        out = ai_result.output
        return EconomicsOverlay(
            industry=industry,
            industry_cycle_stage=str(out.get("industry_cycle_stage", "mid_cycle")),
            supply_demand_balance=_clamp(out.get("supply_demand_balance"), -1.0, 1.0),
            pricing_power=_clamp(out.get("pricing_power"), 0.0, 1.0),
            capacity_utilization=_clamp(out.get("capacity_utilization"), 0.0, 1.0),
            inventory_days_zscore=_to_float(out.get("inventory_days_zscore"), 0.0),
            capex_intensity_trend=_clamp(out.get("capex_intensity_trend"), -1.0, 1.0),
            credit_impulse_alignment=_clamp(out.get("credit_impulse_alignment"), -1.0, 1.0),
            monetary_tailwind=_clamp(out.get("monetary_tailwind"), -1.0, 1.0),
            fx_pressure=_clamp(out.get("fx_pressure"), -1.0, 1.0),
            commodity_cost_pressure=_clamp(out.get("commodity_cost_pressure"), -1.0, 1.0),
            policy_support_strength=_clamp(out.get("policy_support_strength"), 0.0, 1.0),
            expected_industry_revenue_growth_yoy=_to_float(out.get("expected_industry_revenue_growth_yoy"), 0.10),
            expected_horizon_days=int(out.get("expected_horizon_days", 120) or 120),
            economic_thesis=str(out.get("economic_thesis", "")),
            rationale=str(out.get("rationale", "")),
            used_llm=True,
        )


# ---------------------------------------------------------------------------
# Helpers — coercion, fallbacks, vocabulary-free policy parsing
# ---------------------------------------------------------------------------


def _coerce_toggles(toggles: SkillToggles | dict[str, bool] | None) -> SkillToggles:
    if toggles is None:
        return SkillToggles()
    if isinstance(toggles, SkillToggles):
        return toggles
    return SkillToggles(
        policy_analyst=bool(toggles.get("policy_analyst", True)),
        industry_chain_reasoner=bool(toggles.get("industry_chain_reasoner", True)),
        news_credibility_agent=bool(toggles.get("news_credibility_agent", True)),
        sentiment_agent=bool(toggles.get("sentiment_agent", False)),
        valuation_agent=bool(toggles.get("valuation_agent", False)),
        financial_forensics_agent=bool(toggles.get("financial_forensics_agent", False)),
        economics_agent=bool(toggles.get("economics_agent", False)),
    )


def _source_authority(source_level: str) -> float:
    mapping = {
        "central": 0.95,
        "state_council": 0.95,
        "ministry": 0.85,
        "ministry_joint_release": 0.85,
        "provincial": 0.70,
        "municipal": 0.55,
        "industry_association": 0.50,
        "media_interpretation": 0.35,
    }
    return float(mapping.get(str(source_level).lower(), 0.40))


def _policy_user_text(document: Any) -> str:
    title = str(getattr(document, "title", ""))
    body = str(getattr(document, "body", ""))
    source = str(getattr(document, "source", ""))
    source_level = str(getattr(document, "source_level", ""))
    published_at = str(getattr(document, "published_at", ""))
    return (
        f"document_id: {getattr(document, 'document_id', '')}\n"
        f"source: {source}\n"
        f"source_level: {source_level}\n"
        f"published_at: {published_at}\n"
        f"title: {title}\n"
        f"body: {body}"
    )


def _policy_themes_from_llm(output: dict[str, Any]) -> tuple[ThemeExtraction, ...]:
    raw_themes = output.get("themes") if isinstance(output, dict) else None
    if not isinstance(raw_themes, list):
        return ()
    themes: list[ThemeExtraction] = []
    for item in raw_themes:
        if not isinstance(item, dict):
            continue
        theme_name = _slugify(str(item.get("theme_name", item.get("theme", ""))))
        if not theme_name:
            continue
        chain_nodes = tuple(_slugify(str(node)) for node in (item.get("chain_nodes") or ()) if str(node).strip())
        sectors = tuple(_slugify(str(sector)) for sector in (item.get("supported_sectors") or ()) if str(sector).strip())
        themes.append(
            ThemeExtraction(
                theme=theme_name,
                sub_theme=_slugify(str(item.get("sub_theme", ""))) or None,
                chain_nodes=tuple(node for node in chain_nodes if node),
                supported_sectors=tuple(sector for sector in sectors if sector),
                policy_strength=_clamp(item.get("magnitude", item.get("policy_strength")), 0.0, 1.0),
                binding=str(item.get("binding", "encouraged")),
                horizon_days=int(item.get("horizon_days", 120) or 120),
                direction=float(item.get("direction", 1.0) or 1.0),
                risk_flags=tuple(str(flag) for flag in (item.get("risk_flags") or ()) if str(flag)),
                rationale=str(item.get("rationale", "")),
            )
        )
    return tuple(themes)


def _policy_themes_fallback(document: Any, authority: float) -> tuple[ThemeExtraction, ...]:
    """Vocabulary-free fallback: extract noun-like tokens from the document.

    The deterministic path deliberately does **not** use any hand-built
    industry vocabulary. It produces a single "unclassified_policy"-style
    theme keyed by the document's most-cited candidate token group so the
    downstream chain reasoner can still aggregate evidence. Real theme
    naming is the LLM path's job.
    """

    text = f"{getattr(document, 'title', '')}\n{getattr(document, 'body', '')}"
    candidates = _extract_token_groups(text)
    if not candidates:
        slug = _slugify(str(getattr(document, "document_id", "policy")))
        return (
            ThemeExtraction(
                theme=slug or "unclassified_policy",
                sub_theme=None,
                chain_nodes=(),
                supported_sectors=(),
                policy_strength=authority,
                binding="encouraged",
                horizon_days=int(_policy_horizon(getattr(document, "source_level", ""))),
                direction=1.0,
                risk_flags=(),
                rationale="policy_fallback_no_tokens",
            ),
        )
    primary_theme = _slugify(candidates[0][0]) or "unclassified_policy"
    chain_nodes = tuple(_slugify(token) for token, _ in candidates[:8] if _slugify(token))
    return (
        ThemeExtraction(
            theme=primary_theme,
            sub_theme=None,
            chain_nodes=tuple(node for node in chain_nodes if node and node != primary_theme),
            supported_sectors=(),
            policy_strength=authority,
            binding="encouraged",
            horizon_days=int(_policy_horizon(getattr(document, "source_level", ""))),
            direction=1.0,
            risk_flags=(),
            rationale="policy_fallback_token_frequency",
        ),
    )


def _policy_horizon(source_level: str) -> int:
    if source_level in {"central", "state_council"}:
        return 126
    if str(source_level).startswith("ministry"):
        return 90
    if source_level in {"provincial", "municipal"}:
        return 60
    return 20


_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|[一-鿿]{2,6}")
_STOP_TOKENS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "into",
    "data",
    "year",
    "plan",
    "support",
    "policy",
    "project",
    "industry",
    "industrial",
    "industries",
    "company",
    "companies",
    "development",
    "develop",
    "domestic",
    "system",
    "systems",
    "service",
    "services",
    "national",
    "general",
    "government",
    "国家",
    "支持",
    "发展",
    "政策",
    "行业",
    "产业",
    "项目",
    "企业",
    "示范",
    "试点",
    "工作",
    "推进",
    "推动",
    "建设",
    "管理",
    "重点",
    "加快",
    "实施",
}


def _extract_token_groups(text: str) -> list[tuple[str, int]]:
    if not text:
        return []
    tokens = [token.lower() for token in _TOKEN_PATTERN.findall(text)]
    counter: Counter[str] = Counter()
    for token in tokens:
        if token in _STOP_TOKENS:
            continue
        if token.isdigit():
            continue
        counter[token] += 1
    return counter.most_common(16)


def _slugify(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[\s/]+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_一-鿿]", "", cleaned)
    return cleaned


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _news_user_text(row: pd.Series) -> str:
    fields = [
        ("news_id", row.get("news_id", "")),
        ("source", row.get("source", "")),
        ("source_type", row.get("source_type", "")),
        ("symbol", row.get("symbol", "")),
        ("theme", row.get("theme", "")),
        ("title", row.get("title", "")),
        ("summary", row.get("summary", row.get("body", ""))),
        ("published_at", row.get("published_at", "")),
    ]
    return "\n".join(f"{key}: {value}" for key, value in fields if value not in (None, ""))


def _news_score_from_dict(row: pd.Series, ai: dict[str, Any] | None) -> NewsCredibilityAIScore:
    news_id = str(row.get("news_id") or row.get("title") or "news")
    symbols = ai.get("affected_symbols") if isinstance(ai, dict) else None
    if not symbols:
        raw_symbol = row.get("symbol")
        symbols = [str(raw_symbol)] if raw_symbol not in (None, "", float("nan")) else []
    affected_theme = (ai or {}).get("affected_theme") if isinstance(ai, dict) else None
    if not affected_theme:
        theme_raw = row.get("theme")
        affected_theme = str(theme_raw) if theme_raw not in (None, "", float("nan")) else None
    if isinstance(ai, dict):
        return NewsCredibilityAIScore(
            news_id=news_id,
            source_reliability=_clamp(ai.get("source_reliability"), 0.0, 1.0),
            is_primary_source=bool(ai.get("is_primary_source", False)),
            is_official=bool(ai.get("is_official", False)),
            cross_validation_count=int(ai.get("cross_validation_count", 0) or 0),
            event_type=str(ai.get("event_type", "no_trade")),
            affected_symbols=tuple(str(sym) for sym in symbols if str(sym)),
            affected_theme=str(affected_theme) if affected_theme else None,
            sentiment_score=_clamp(ai.get("sentiment_score"), -1.0, 1.0),
            short_term_impact=_clamp(ai.get("short_term_impact_score"), 0.0, 1.0),
            medium_term_impact=_clamp(ai.get("medium_term_impact_score"), 0.0, 1.0),
            fundamental_impact=_clamp(ai.get("fundamental_impact_score"), 0.0, 1.0),
            decay_half_life=_to_float(ai.get("decay_half_life"), 5.0),
            horizon_days=int(ai.get("horizon_days", 5) or 5),
            rumor_risk=_clamp(ai.get("rumor_risk"), 0.0, 1.0),
            confidence=_news_confidence_from_ai(ai),
            rationale=str(ai.get("rationale", "")),
            used_llm=True,
        )
    return _news_fallback(row, symbols, affected_theme)


def _news_confidence_from_ai(ai: dict[str, Any]) -> float:
    reliability = _clamp(ai.get("source_reliability"), 0.0, 1.0)
    primary = 1.0 if ai.get("is_primary_source") else 0.0
    official = 1.0 if ai.get("is_official") else 0.0
    cross = _clamp(float(ai.get("cross_validation_count", 0) or 0) / 4.0, 0.0, 1.0)
    rumor = _clamp(ai.get("rumor_risk"), 0.0, 1.0)
    score = 0.35 * reliability + 0.20 * primary + 0.20 * official + 0.25 * cross
    return float(max(0.0, score - 0.40 * rumor))


def _news_fallback(row: pd.Series, symbols: list[Any], affected_theme: str | None) -> NewsCredibilityAIScore:
    source_type = str(row.get("source_type", row.get("source", "news"))).lower()
    reliability_map = {
        "company_announcement": 0.90,
        "exchange_disclosure": 0.92,
        "official_policy": 0.88,
        "mainstream_media": 0.72,
        "industry_media": 0.62,
        "social_media": 0.25,
        "rumor": 0.15,
    }
    reliability = float(row.get("source_reliability", reliability_map.get(source_type, 0.40)))
    sentiment = _safe_float(row.get("sentiment_score"))
    primary = source_type in {"company_announcement", "exchange_disclosure", "official_policy"}
    cross_validation = int(row.get("cross_validation_count", 1 if primary else 0))
    rumor_risk = float(row.get("rumor_risk", 0.7 if source_type in {"rumor", "social_media"} else 0.1))
    confidence = max(0.0, 0.35 * reliability + 0.20 * (1.0 if primary else 0.0) + 0.20 * (1.0 if primary else 0.0) + 0.25 * (cross_validation / 4.0) - 0.40 * rumor_risk)
    return NewsCredibilityAIScore(
        news_id=str(row.get("news_id") or row.get("title") or "news"),
        source_reliability=reliability,
        is_primary_source=primary,
        is_official=primary,
        cross_validation_count=cross_validation,
        event_type=str(row.get("event_type", "sentiment_positive" if sentiment >= 0 else "sentiment_negative")),
        affected_symbols=tuple(str(sym) for sym in symbols if str(sym)),
        affected_theme=affected_theme,
        sentiment_score=sentiment,
        short_term_impact=float(confidence * abs(sentiment)),
        medium_term_impact=float(confidence * max(sentiment, 0.0)),
        fundamental_impact=float(confidence * max(sentiment, 0.0) * (1.0 if primary else 0.4)),
        decay_half_life=float(row.get("decay_half_life", 20.0 if primary else 5.0)),
        horizon_days=int(row.get("horizon_days", 60 if primary else 5)),
        rumor_risk=rumor_risk,
        confidence=float(confidence),
        rationale=f"news_fallback:source_type={source_type}",
        used_llm=False,
    )


def _sentiment_fallback(scope: str, reason: str) -> SentimentAIResult:
    return SentimentAIResult(
        scope=scope,
        retail_sentiment=0.0,
        institutional_sentiment=0.0,
        sentiment_divergence=0.0,
        coordinated_pumping_risk=0.30,
        short_squeeze_risk=0.20,
        attention_surge_score=0.0,
        social_volume_zscore=0.0,
        rationale=f"sentiment_fallback:{reason}",
        used_llm=False,
    )


# ---------------------------------------------------------------------------
# Numeric coercion helpers
# ---------------------------------------------------------------------------


def _clamp(value: Any, low: float, high: float) -> float:
    numeric = _to_float(value, low)
    if math.isnan(numeric) or math.isinf(numeric):
        return low
    return float(max(low, min(high, numeric)))


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _safe_float(value: Any) -> float:
    return _to_float(value, 0.0)


__all__ = [
    "EconomicsOverlay",
    "ForensicsOverlay",
    "LLMOrchestrator",
    "NewsCredibilityAIScore",
    "PolicyAnalysis",
    "SentimentAIResult",
    "SkillToggles",
    "ThemeExtraction",
    "ValuationOverlay",
]
