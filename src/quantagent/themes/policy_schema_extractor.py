from __future__ import annotations

from typing import Any

from quantagent.agents.schema_extraction_client import OpenAICompatibleSchemaExtractor
from quantagent.data.providers.base import ProviderUnavailable
from quantagent.themes.policy_crawler import PolicyDocument
from quantagent.v7.schemas import EvidenceRecord, EventType, SourceType
from quantagent.v7.scoring import policy_authority_score


POLICY_EXTRACTION_PROMPT = """
You extract structured A-share policy and industrial-chain evidence.
Return JSON only. Do not produce investment advice or orders.
Schema:
{
  "themes": [
    {
      "theme": "ai_compute",
      "sub_theme": "domestic_gpu",
      "chain_nodes": ["gpu", "server"],
      "policy_strength": 0.0,
      "confidence": 0.0,
      "horizon_days": 126,
      "direction": 1.0,
      "risk_flags": ["policy_constraint"]
    }
  ]
}
"""


def extract_policy_schema_evidence(
    documents: list[PolicyDocument],
    as_of_date: str,
    config: dict[str, Any] | None = None,
) -> tuple[list[EvidenceRecord], tuple[str, ...]]:
    """Use an optional remote model API for schema extraction; rules remain the fallback."""
    extractor = OpenAICompatibleSchemaExtractor(config)
    records: list[EvidenceRecord] = []
    warnings: list[str] = []
    for document in documents:
        try:
            extracted = extractor.extract_json(
                system_prompt=POLICY_EXTRACTION_PROMPT,
                user_text=f"title: {document.title}\nbody: {document.body}",
            )
        except ProviderUnavailable as exc:
            warnings.append(str(exc))
            break
        except Exception as exc:  # pragma: no cover - defensive API boundary
            warnings.append(f"policy_schema_extraction_failed:{type(exc).__name__}")
            continue
        for index, item in enumerate(extracted.get("themes", ())):
            records.append(_evidence_from_item(document, item, index, as_of_date))
    return records, tuple(dict.fromkeys(warnings))


def _evidence_from_item(document: PolicyDocument, item: dict[str, Any], index: int, as_of_date: str) -> EvidenceRecord:
    authority = policy_authority_score(document.source_level)
    theme = str(item.get("theme", "unclassified_policy"))
    chain_nodes = tuple(str(node) for node in item.get("chain_nodes", ()) if str(node))
    horizon = int(item.get("horizon_days", _policy_horizon(document.source_level)))
    confidence = float(max(0.0, min(1.0, item.get("confidence", 0.55))))
    strength = float(max(0.0, min(1.0, item.get("policy_strength", authority))))
    return EvidenceRecord(
        evidence_id=f"{document.document_id}:schema:{theme}:{index}",
        source=document.source,
        source_type=SourceType.OFFICIAL_POLICY,
        source_authority_level=authority,
        timestamp=as_of_date,
        published_at=document.published_at,
        effective_start_date=document.effective_start_date,
        effective_end_date=document.effective_end_date,
        theme=theme,
        sub_theme=str(item.get("sub_theme", "")) or None,
        chain_node=chain_nodes[0] if chain_nodes else None,
        event_type=EventType.POLICY_SUPPORT,
        direction=float(item.get("direction", 1.0)),
        magnitude=strength,
        confidence=confidence,
        evidence_quality=authority,
        source_reliability=authority,
        cross_validation_count=0,
        decay_half_life=max(5.0, horizon / 2.0),
        horizon_days=horizon,
        rationale=f"remote_schema_extraction:{document.title[:200]}",
        raw_reference={
            "document_id": document.document_id,
            "chain_nodes": chain_nodes,
            "extraction_source": "remote_schema_api",
            **document.raw_reference,
        },
        point_in_time_valid=bool(document.published_at <= as_of_date),
        risk_flags=tuple(str(flag) for flag in item.get("risk_flags", ()) if str(flag)),
    ).with_hash()


def _policy_horizon(source_level: str) -> int:
    if source_level in {"central", "state_council"}:
        return 126
    if source_level == "ministry":
        return 90
    if source_level in {"provincial", "municipal"}:
        return 60
    return 20
