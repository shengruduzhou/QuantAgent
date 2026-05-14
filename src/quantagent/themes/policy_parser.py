"""Policy document parser.

This module no longer carries hand-maintained ``THEME_KEYWORDS`` or
``CHAIN_KEYWORDS`` lists. Themes and chain nodes are derived in two ways:

1.  ``LLMOrchestrator.analyze_policies`` (when AI is enabled): the
    ``policy_analyst`` skill reads the raw title/body and returns themes,
    chain nodes, magnitude, binding strength, and horizon directly.
2.  Vocabulary-free deterministic fallback (when AI is disabled): we extract
    the highest-frequency content tokens from the document and emit one
    placeholder ``ThemeExtraction`` keyed by the dominant token group so the
    downstream chain reasoner can still aggregate evidence.

Both paths emit the same ``ParsedPolicyDocument`` structure so callers can be
ignorant of which mode produced the result. The ``EvidenceRecord`` objects
generated for V7 carry the chain nodes the parser decided on — never a hand-
maintained list.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import re

from quantagent.agents.llm_orchestrator import (
    LLMOrchestrator,
    PolicyAnalysis,
    ThemeExtraction,
)
from quantagent.themes.policy_crawler import PolicyDocument
from quantagent.v7.schemas import EvidenceRecord, EventType, SourceType
from quantagent.v7.scoring import policy_authority_score


@dataclass(frozen=True)
class ParsedPolicyDocument:
    document: PolicyDocument
    authority_score: float
    themes: tuple[str, ...]
    chain_nodes: tuple[str, ...]
    target_years: tuple[int, ...]
    subsidy_signal: bool
    pilot_signal: bool
    constraint_terms: tuple[str, ...]
    confidence: float
    theme_details: tuple[ThemeExtraction, ...] = ()
    extraction_source: str = "fallback"


def parse_policy_document(
    document: PolicyDocument,
    *,
    analysis: PolicyAnalysis | None = None,
    orchestrator: LLMOrchestrator | None = None,
) -> ParsedPolicyDocument:
    """Parse a policy document; uses AI analysis when provided.

    Backwards-compatible: callers that don't pass an analysis or orchestrator
    still get a deterministic best-effort parse based on token frequency
    extraction — no hardcoded industry vocabularies.
    """

    if analysis is None and orchestrator is not None:
        analyses = orchestrator.analyze_policies([document])
        analysis = analyses[0] if analyses else None
    if analysis is None:
        analysis = LLMOrchestrator().analyze_policies([document])[0]
    text = f"{document.title}\n{document.body}"
    years = tuple(sorted({int(match) for match in re.findall(r"\b20[2-4][0-9]\b", text)}))
    subsidy_signal = bool(_any_marker(text, _SUBSIDY_MARKERS))
    pilot_signal = bool(_any_marker(text, _PILOT_MARKERS))
    constraint_terms = tuple(term for term in _CONSTRAINT_MARKERS if term in text.lower())
    authority = analysis.source_authority or policy_authority_score(document.source_level)
    theme_names = tuple(item.theme for item in analysis.themes if item.theme)
    chain_nodes = tuple(node for item in analysis.themes for node in item.chain_nodes if node)
    confidence = min(
        1.0,
        0.35
        + 0.35 * authority
        + 0.05 * len(theme_names)
        + (0.05 if chain_nodes else 0.0)
        + (0.10 if analysis.used_llm else 0.0),
    )
    return ParsedPolicyDocument(
        document=document,
        authority_score=authority,
        themes=tuple(dict.fromkeys(theme_names)),
        chain_nodes=tuple(dict.fromkeys(chain_nodes)),
        target_years=years,
        subsidy_signal=subsidy_signal,
        pilot_signal=pilot_signal,
        constraint_terms=constraint_terms,
        confidence=confidence,
        theme_details=analysis.themes,
        extraction_source="llm" if analysis.used_llm else f"fallback:{analysis.fallback_reason or 'unknown'}",
    )


def policy_to_evidence(parsed: ParsedPolicyDocument, as_of_date: str) -> list[EvidenceRecord]:
    """Convert parser output into V7 evidence records."""

    records: list[EvidenceRecord] = []
    event_type = EventType.SUBSIDY if parsed.subsidy_signal else EventType.POLICY_SUPPORT
    detail_by_theme = {detail.theme: detail for detail in parsed.theme_details}
    themes_iter = parsed.themes or ("unclassified_policy",)
    for theme in themes_iter:
        detail = detail_by_theme.get(theme)
        chain_nodes = detail.chain_nodes if detail else parsed.chain_nodes
        primary_node = chain_nodes[0] if chain_nodes else None
        horizon = int(detail.horizon_days) if detail else _policy_horizon(parsed.document.source_level)
        magnitude = detail.policy_strength if detail else parsed.authority_score
        direction = float(detail.direction) if detail else 1.0
        risk_flags = detail.risk_flags if detail else tuple("policy_constraint" for _ in parsed.constraint_terms[:1])
        evidence_id = f"{parsed.document.document_id}:{theme}"
        record = EvidenceRecord(
            evidence_id=evidence_id,
            source=parsed.document.source,
            source_type=SourceType.OFFICIAL_POLICY,
            source_authority_level=parsed.authority_score,
            timestamp=as_of_date,
            published_at=parsed.document.published_at,
            effective_start_date=parsed.document.effective_start_date,
            effective_end_date=parsed.document.effective_end_date,
            theme=theme,
            sub_theme=detail.sub_theme if detail else None,
            chain_node=primary_node,
            event_type=event_type,
            direction=direction,
            magnitude=magnitude,
            confidence=parsed.confidence,
            evidence_quality=parsed.authority_score,
            source_reliability=parsed.authority_score,
            cross_validation_count=0,
            decay_half_life=_policy_half_life(parsed.document.source_level, horizon),
            horizon_days=horizon,
            rationale=parsed.document.title[:240],
            raw_reference={
                "document_id": parsed.document.document_id,
                "target_years": parsed.target_years,
                "chain_nodes": chain_nodes,
                "supported_sectors": detail.supported_sectors if detail else (),
                "binding": detail.binding if detail else "encouraged",
                "extraction_source": parsed.extraction_source,
                "reference_hash": sha1((parsed.document.title + parsed.document.body).encode("utf-8")).hexdigest(),
                **parsed.document.raw_reference,
            },
            point_in_time_valid=bool(parsed.document.published_at <= as_of_date),
            risk_flags=risk_flags,
        ).with_hash()
        records.append(record)
    return records


_SUBSIDY_MARKERS = ("subsidy", "tax", "财政", "补贴", "税收", "采购", "扶持", "奖励")
_PILOT_MARKERS = ("pilot", "demonstration", "试点", "示范", "首批")
_CONSTRAINT_MARKERS = ("risk", "compliance", "监管", "约束", "安全", "整改", "限期")


def _any_marker(text: str, markers: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(marker.lower() in lower for marker in markers)


def _policy_horizon(source_level: str) -> int:
    if source_level in {"central", "state_council"}:
        return 126
    if str(source_level).startswith("ministry"):
        return 90
    if source_level in {"provincial", "municipal"}:
        return 60
    return 20


def _policy_half_life(source_level: str, horizon_days: int | None = None) -> float:
    base = horizon_days if horizon_days else _policy_horizon(source_level)
    return max(5.0, float(base) / 2.0)
