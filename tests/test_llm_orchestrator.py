"""Tests for the LLM orchestrator + AI-driven theme discovery.

These tests verify the behaviour you get **without** network access — i.e. the
deterministic vocabulary-free fallback. When the API is enabled the same code
paths simply substitute LLM output, so a passing offline test guarantees the
contract while the real wiring is validated by integration runs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.agents.llm_orchestrator import (
    LLMOrchestrator,
    PolicyAnalysis,
    SkillToggles,
)
from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
from quantagent.agents.llm_skill_client import LLMSkillResult
from quantagent.themes.policy_crawler import local_policy_documents
from quantagent.themes.policy_parser import parse_policy_document, policy_to_evidence


def _disabled_orchestrator() -> LLMOrchestrator:
    return LLMOrchestrator(
        LLMSkillClient(LLMSkillConfig(enabled=False, allow_network=False)),
        SkillToggles(),
    )


def test_orchestrator_policy_fallback_is_vocabulary_free():
    docs = local_policy_documents(
        [
            {
                "document_id": "p1",
                "title": "Synthetic supply chain support plan",
                "body": "Promote synthetic widgets, hyper-loop tunnels and quantum mirrors. Target 2027.",
                "source": "ministry",
                "source_level": "ministry",
                "published_at": "2026-05-14",
            }
        ]
    )
    orchestrator = _disabled_orchestrator()
    analyses = orchestrator.analyze_policies(docs)
    assert len(analyses) == 1
    assert isinstance(analyses[0], PolicyAnalysis)
    assert analyses[0].used_llm is False
    assert analyses[0].themes, "fallback must produce at least one theme"
    primary = analyses[0].themes[0]
    assert primary.theme, "theme name should be a token derived from the document"
    assert primary.policy_strength > 0.0


def test_policy_parser_consumes_orchestrator_output():
    docs = local_policy_documents(
        [
            {
                "document_id": "p2",
                "title": "Industrial chip roadmap",
                "body": "Accelerate domestic substitution for advanced packaging and EDA tooling.",
                "source": "state_council",
                "source_level": "central",
                "published_at": "2026-05-14",
            }
        ]
    )
    orchestrator = _disabled_orchestrator()
    analyses = orchestrator.analyze_policies(docs)
    parsed = parse_policy_document(docs[0], analysis=analyses[0])
    assert parsed.themes, "parser must surface at least one theme"
    assert parsed.authority_score >= 0.85, "central-level documents get high authority"
    evidence = policy_to_evidence(parsed, "2026-05-14")
    assert evidence, "parser must produce evidence records"
    assert evidence[0].theme == parsed.themes[0]


def test_orchestrator_news_fallback_emits_credibility_score():
    news = pd.DataFrame(
        [
            {
                "news_id": "n1",
                "source": "exchange_disclosure",
                "source_type": "exchange_disclosure",
                "symbol": "600001.SH",
                "title": "Order contract signed",
                "summary": "Customer placed a multi-year framework agreement.",
                "published_at": "2026-05-12",
                "sentiment_score": 0.65,
            },
            {
                "news_id": "n2",
                "source": "wechat_group",
                "source_type": "rumor",
                "symbol": "002371.SZ",
                "title": "Rumor of imminent investigation",
                "summary": "Anonymous source claims accounting probe is coming.",
                "published_at": "2026-05-13",
                "sentiment_score": -0.6,
            },
        ]
    )
    scores = _disabled_orchestrator().score_news_batch(news)
    assert len(scores) == 2
    by_id = {score.news_id: score for score in scores}
    assert by_id["n1"].confidence > by_id["n2"].confidence
    assert by_id["n2"].rumor_risk > 0.5


def test_orchestrator_overlays_are_none_when_disabled():
    orchestrator = _disabled_orchestrator()
    assert orchestrator.overlay_valuation("600001.SH", "revenue: 100") is None
    assert orchestrator.overlay_forensics("600001.SH", "receivables: 30") is None
    assert orchestrator.overlay_economics("server", "supply_demand: 0.1") is None
    sentiment = orchestrator.assess_sentiment("ai_compute", "some chatter")
    assert sentiment.used_llm is False


class _FakeValuationClient:
    config = LLMSkillConfig(provider="disabled", enabled=True, allow_network=False)

    def invoke(self, skill_name, *, system_prompt, user_text, fallback=None):
        return LLMSkillResult(
            skill_name=skill_name,
            output={
                "fair_value_per_share": 20.0,
                "margin_of_safety_pct": 0.25,
                "valuation_score": 82.0,
                "bubble_risk_score": 0.2,
                "investment_horizon_days": 120,
                "method_weights": {"relative": 0.5, "peg": 0.5},
                "key_assumptions": {"growth_rate": 0.30},
                "forward_pe": 24.0,
                "peg": 0.8,
                "pe_digestion_years": 1.6,
                "peg_rating": "undervalued",
                "rationale": "PEG is supported by forecast growth.",
            },
            raw_text="{}",
            used_fallback=False,
        )


def test_valuation_overlay_consumes_peg_fields_from_llm():
    orchestrator = LLMOrchestrator(
        _FakeValuationClient(),
        SkillToggles(valuation_agent=True),
    )

    overlay = orchestrator.overlay_valuation("600001.SH", "eps_forward: 1.0")

    assert overlay is not None
    assert overlay.forward_pe == 24.0
    assert overlay.peg == 0.8
    assert overlay.pe_digestion_years == 1.6
    assert overlay.peg_rating == "undervalued"


def test_policy_parser_old_signature_still_works():
    """parse_policy_document(doc) without explicit analysis must still parse."""

    docs = local_policy_documents(
        [
            {
                "document_id": "p3",
                "title": "Renewable thermal storage initiative",
                "body": "Subsidy and pilot programmes for high-temperature thermal storage.",
                "source": "ministry",
                "source_level": "ministry",
                "published_at": "2026-05-14",
            }
        ]
    )
    parsed = parse_policy_document(docs[0])
    assert parsed.themes, "even with no explicit orchestrator, parser yields themes"
    assert parsed.extraction_source.startswith("fallback") or parsed.extraction_source == "llm"
