"""Dynamic, evidence-driven industry-chain reasoner.

Replaces the static AI_COMPUTE_TEMPLATE used by industry_chain_graph.py. The
reasoner reads policy documents, news, exchange disclosures and financial
statements, infers candidate chain nodes and edges from co-mention patterns,
and optionally invokes an LLM skill to refine the graph. No hardcoded chain
templates are used. The result is a per-theme graph with explicit relation
types (DIRECT vs STRONG vs WEAK vs FALSE) so the stock-pool selector can
classify "真相关 / 强相关 / 弱关联 / 伪相关".
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Any

from quantagent.agents.llm_skill_client import LLMSkillClient, LLMSkillConfig
from quantagent.agents.skills import INDUSTRY_CHAIN_REASONER
from quantagent.v7.schemas import (
    ChainEdge,
    ChainNode,
    ChainRelationType,
    EvidenceRecord,
    ThemeProfile,
)


@dataclass(frozen=True)
class IndustryChainReasonerConfig:
    min_evidence_count_for_node: int = 1
    min_evidence_count_for_strong_edge: int = 2
    weak_association_max_evidence: int = 1
    direct_exposure_keywords: tuple[str, ...] = (
        "main product",
        "core product",
        "主营",
        "核心产品",
        "primary revenue",
    )
    bottleneck_keywords: tuple[str, ...] = (
        "bottleneck",
        "shortage",
        "短缺",
        "卡脖子",
        "irreplaceable",
        "monopoly",
        "sole supplier",
    )
    domestic_substitution_keywords: tuple[str, ...] = (
        "domestic substitution",
        "国产替代",
        "import substitution",
        "self-sufficiency",
        "自主可控",
    )
    infrastructure_keywords: tuple[str, ...] = (
        "infrastructure",
        "power",
        "cooling",
        "data center",
        "network",
        "电力",
        "冷却",
        "数据中心",
    )
    cost_beneficiary_keywords: tuple[str, ...] = (
        "input cost",
        "raw material",
        "commodity price",
        "原材料",
        "成本下降",
    )
    use_llm_refinement: bool = False
    min_node_confidence_for_publication: float = 0.30
    strict_no_template_fallback: bool = True


@dataclass(frozen=True)
class IndustryChainReasonerResult:
    theme: str
    nodes: list[ChainNode]
    edges: list[ChainEdge]
    chain_confidence: float
    used_llm: bool
    rationale: str


def reason_industry_chain(
    profile: ThemeProfile,
    evidence: list[EvidenceRecord],
    config: IndustryChainReasonerConfig | None = None,
    llm_client: LLMSkillClient | None = None,
) -> IndustryChainReasonerResult:
    """Build a chain graph for one theme purely from evidence + optional LLM."""
    config = config or IndustryChainReasonerConfig()
    theme_evidence = [record for record in evidence if record.theme in {None, profile.theme_name}]
    raw_node_mentions = _extract_node_mentions(theme_evidence)
    nodes = _build_nodes(raw_node_mentions, theme_evidence, config)
    edges = _build_edges(raw_node_mentions, theme_evidence, nodes, config)
    rationale = (
        f"theme={profile.theme_name}; evidence_count={len(theme_evidence)}; "
        f"nodes={len(nodes)}; edges={len(edges)}; "
        f"policy_strength={profile.policy_strength:.2f}; "
        f"fundamental_strength={profile.industry_fundamental_strength:.2f}"
    )
    used_llm = False
    if config.use_llm_refinement and llm_client is not None and theme_evidence:
        nodes, edges, used_llm = _llm_refine(profile, theme_evidence, nodes, edges, llm_client)
    chain_confidence = _aggregate_chain_confidence(nodes, edges, profile)
    return IndustryChainReasonerResult(
        theme=profile.theme_name,
        nodes=nodes,
        edges=edges,
        chain_confidence=chain_confidence,
        used_llm=used_llm,
        rationale=rationale,
    )


def reason_industry_chain_for_themes(
    profiles: list[ThemeProfile],
    evidence: list[EvidenceRecord],
    config: IndustryChainReasonerConfig | None = None,
    llm_client: LLMSkillClient | None = None,
) -> dict[str, IndustryChainReasonerResult]:
    return {
        profile.theme_name: reason_industry_chain(profile, evidence, config, llm_client)
        for profile in profiles
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _extract_node_mentions(records: list[EvidenceRecord]) -> dict[str, list[EvidenceRecord]]:
    mentions: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in records:
        for node_id in _node_ids_from_record(record):
            mentions[node_id].append(record)
    return mentions


def _node_ids_from_record(record: EvidenceRecord) -> list[str]:
    candidates: list[str] = []
    if record.chain_node:
        candidates.append(_canonicalize_node_id(record.chain_node))
    if record.sub_theme:
        candidates.append(_canonicalize_node_id(record.sub_theme))
    raw_nodes = record.raw_reference.get("chain_nodes") if record.raw_reference else None
    if isinstance(raw_nodes, str):
        candidates.extend(_canonicalize_node_id(item) for item in re.split(r"[,;|]", raw_nodes) if item.strip())
    elif isinstance(raw_nodes, (list, tuple)):
        candidates.extend(_canonicalize_node_id(str(item)) for item in raw_nodes if str(item).strip())
    keywords = record.raw_reference.get("keywords") if record.raw_reference else None
    if isinstance(keywords, (list, tuple)):
        candidates.extend(_canonicalize_node_id(str(item)) for item in keywords if str(item).strip())
    return [item for item in candidates if item]


def _canonicalize_node_id(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[\s/]+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_一-鿿]", "", cleaned)
    return cleaned


def _build_nodes(
    node_mentions: dict[str, list[EvidenceRecord]],
    theme_evidence: list[EvidenceRecord],
    config: IndustryChainReasonerConfig,
) -> list[ChainNode]:
    nodes: list[ChainNode] = []
    total_evidence = max(1, len(theme_evidence))
    for node_id, records in node_mentions.items():
        if len(records) < config.min_evidence_count_for_node:
            continue
        evidence_ids = tuple(sorted({record.evidence_id for record in records if record.evidence_id}))
        policy_records = [record for record in records if record.source_type.value == "official_policy"]
        text_blob = " ".join(_record_text(record) for record in records).lower()
        bottleneck = _keyword_match_score(text_blob, config.bottleneck_keywords)
        substitution = _keyword_match_score(text_blob, config.domestic_substitution_keywords)
        infra = _keyword_match_score(text_blob, config.infrastructure_keywords)
        direct = _keyword_match_score(text_blob, config.direct_exposure_keywords)
        cost = _keyword_match_score(text_blob, config.cost_beneficiary_keywords)
        share = len(records) / total_evidence
        node = ChainNode(
            node_id=node_id,
            node_name=node_id.replace("_", " "),
            dependency_strength=min(1.0, 0.30 + 0.50 * share + 0.20 * direct),
            bottleneck_score=min(1.0, 0.10 + 0.70 * bottleneck + 0.20 * substitution),
            domestic_substitution_score=min(1.0, 0.10 + 0.80 * substitution),
            supply_shortage_score=min(1.0, 0.10 + 0.70 * bottleneck),
            price_elasticity=min(1.0, 0.20 + 0.50 * cost),
            profit_elasticity=min(1.0, 0.20 + 0.60 * direct + 0.30 * bottleneck),
            demand_visibility=min(1.0, 0.30 + 0.40 * share + 0.20 * direct),
            policy_support_score=min(1.0, 0.20 + 0.60 * (len(policy_records) / max(1, len(records)))),
            technology_barrier=min(1.0, 0.20 + 0.60 * bottleneck + 0.20 * substitution),
            competition_intensity=max(0.0, 0.80 - 0.50 * bottleneck - 0.30 * substitution),
            evidence_ids=evidence_ids,
        )
        if node.dependency_strength + node.policy_support_score >= 2 * 0.30:
            nodes.append(node)
    nodes.sort(key=lambda n: (-n.dependency_strength - n.policy_support_score, n.node_id))
    return nodes


def _build_edges(
    node_mentions: dict[str, list[EvidenceRecord]],
    theme_evidence: list[EvidenceRecord],
    nodes: list[ChainNode],
    config: IndustryChainReasonerConfig,
) -> list[ChainEdge]:
    node_ids = {node.node_id for node in nodes}
    node_by_id = {node.node_id: node for node in nodes}
    co_occurrence: dict[tuple[str, str], list[EvidenceRecord]] = defaultdict(list)
    for record in theme_evidence:
        record_nodes = sorted({nid for nid in _node_ids_from_record(record) if nid in node_ids})
        for i, source in enumerate(record_nodes):
            for target in record_nodes[i + 1 :]:
                co_occurrence[(source, target)].append(record)
    edges: list[ChainEdge] = []
    for (source, target), records in co_occurrence.items():
        relation = _classify_relation(node_by_id[source], node_by_id[target], records, config)
        strength = min(1.0, 0.30 + 0.20 * len(records))
        edges.append(
            ChainEdge(
                source_node_id=source,
                target_node_id=target,
                relation_type=relation,
                relation_strength=strength,
                evidence_ids=tuple(sorted({record.evidence_id for record in records if record.evidence_id})),
            )
        )
    edges.sort(key=lambda e: (e.source_node_id, e.target_node_id))
    return edges


def _classify_relation(
    source: ChainNode,
    target: ChainNode,
    records: list[EvidenceRecord],
    config: IndustryChainReasonerConfig,
) -> ChainRelationType:
    if len(records) <= config.weak_association_max_evidence:
        return ChainRelationType.WEAK_ASSOCIATION
    text = " ".join(_record_text(record) for record in records).lower()
    if any(keyword in text for keyword in config.bottleneck_keywords):
        return ChainRelationType.CRITICAL_BOTTLENECK
    if any(keyword in text for keyword in config.domestic_substitution_keywords):
        return ChainRelationType.DOMESTIC_SUBSTITUTION
    if any(keyword in text for keyword in config.direct_exposure_keywords):
        return ChainRelationType.DIRECT_EXPOSURE
    if any(keyword in text for keyword in config.infrastructure_keywords):
        return ChainRelationType.INFRASTRUCTURE_DEPENDENCY
    if any(keyword in text for keyword in config.cost_beneficiary_keywords):
        return ChainRelationType.COST_BENEFICIARY
    if source.dependency_strength >= target.dependency_strength:
        return ChainRelationType.UPSTREAM_SUPPLIER
    return ChainRelationType.DOWNSTREAM_APPLICATION


def _record_text(record: EvidenceRecord) -> str:
    parts = [record.rationale or ""]
    raw = record.raw_reference or {}
    for key in ("title", "body", "snippet", "summary", "keywords"):
        value = raw.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (list, tuple)):
            parts.extend(str(item) for item in value)
    return " ".join(parts)


def _keyword_match_score(text: str, keywords: tuple[str, ...]) -> float:
    if not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword.lower() in text)
    return min(1.0, hits / max(1, math.sqrt(len(keywords))))


def _aggregate_chain_confidence(nodes: list[ChainNode], edges: list[ChainEdge], profile: ThemeProfile) -> float:
    if not nodes:
        return 0.0
    node_strength = sum(node.dependency_strength + node.policy_support_score for node in nodes) / (2 * len(nodes))
    edge_strength = (sum(edge.relation_strength for edge in edges) / len(edges)) if edges else 0.30
    return min(1.0, 0.55 * node_strength + 0.25 * edge_strength + 0.20 * profile.theme_confidence)


def _llm_refine(
    profile: ThemeProfile,
    evidence: list[EvidenceRecord],
    nodes: list[ChainNode],
    edges: list[ChainEdge],
    llm_client: LLMSkillClient,
) -> tuple[list[ChainNode], list[ChainEdge], bool]:
    user_payload = _llm_user_text(profile, evidence)
    fallback = INDUSTRY_CHAIN_REASONER.fallback_shape | {"theme": profile.theme_name}
    result = llm_client.invoke(
        INDUSTRY_CHAIN_REASONER.name,
        system_prompt=INDUSTRY_CHAIN_REASONER.system_prompt,
        user_text=user_payload,
        fallback=fallback,
    )
    if result.used_fallback or not result.output:
        return nodes, edges, False
    llm_nodes = _coerce_llm_nodes(result.output.get("nodes", []))
    llm_edges = _coerce_llm_edges(result.output.get("edges", []))
    return llm_nodes or nodes, llm_edges or edges, True


def _llm_user_text(profile: ThemeProfile, evidence: list[EvidenceRecord]) -> str:
    lines = [f"theme: {profile.theme_name}", f"lifecycle: {profile.lifecycle_stage.value}"]
    for record in evidence[:60]:
        lines.append(
            "evidence: "
            f"id={record.evidence_id} | source={record.source} | "
            f"type={record.source_type.value} | symbol={record.symbol or ''} | "
            f"chain_node={record.chain_node or ''} | rationale={record.rationale}"
        )
    return "\n".join(lines)


def _coerce_llm_nodes(raw: Any) -> list[ChainNode]:
    if not isinstance(raw, list):
        return []
    nodes: list[ChainNode] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        node_id = _canonicalize_node_id(str(item.get("node_id", "")))
        if not node_id:
            continue
        evidence_ids = tuple(str(value) for value in item.get("evidence_ids", []) if str(value))
        nodes.append(
            ChainNode(
                node_id=node_id,
                node_name=str(item.get("node_name", node_id.replace("_", " "))),
                dependency_strength=_clamp_float(item.get("dependency_strength")),
                bottleneck_score=_clamp_float(item.get("bottleneck_score")),
                domestic_substitution_score=_clamp_float(item.get("domestic_substitution_score")),
                supply_shortage_score=_clamp_float(item.get("supply_shortage_score")),
                demand_visibility=_clamp_float(item.get("demand_visibility")),
                policy_support_score=_clamp_float(item.get("policy_support_score")),
                technology_barrier=_clamp_float(item.get("technology_barrier")),
                competition_intensity=_clamp_float(item.get("competition_intensity")),
                evidence_ids=evidence_ids,
            )
        )
    return nodes


def _coerce_llm_edges(raw: Any) -> list[ChainEdge]:
    if not isinstance(raw, list):
        return []
    edges: list[ChainEdge] = []
    valid_relations = {item.value for item in ChainRelationType}
    for item in raw:
        if not isinstance(item, dict):
            continue
        source = _canonicalize_node_id(str(item.get("source_node_id", "")))
        target = _canonicalize_node_id(str(item.get("target_node_id", "")))
        relation_raw = str(item.get("relation_type", "weak_association"))
        if relation_raw not in valid_relations:
            relation_raw = "weak_association"
        evidence_ids = tuple(str(value) for value in item.get("evidence_ids", []) if str(value))
        if not source or not target:
            continue
        edges.append(
            ChainEdge(
                source_node_id=source,
                target_node_id=target,
                relation_type=ChainRelationType(relation_raw),
                relation_strength=_clamp_float(item.get("relation_strength")),
                evidence_ids=evidence_ids,
            )
        )
    return edges


def _clamp_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric) or math.isinf(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def node_mention_frequency(evidence: list[EvidenceRecord]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in evidence:
        for node_id in _node_ids_from_record(record):
            counter[node_id] += 1
    return dict(counter)
