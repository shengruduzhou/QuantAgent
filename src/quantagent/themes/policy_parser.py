from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import re

from quantagent.themes.policy_crawler import PolicyDocument
from quantagent.v7.schemas import EvidenceRecord, EventType, SourceType
from quantagent.v7.scoring import policy_authority_score


THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ai_compute": ("ai", "artificial intelligence", "compute", "算力", "智能算力", "gpu", "服务器", "数据中心"),
    "semiconductor_domestic_substitution": ("semiconductor", "chip", "integrated circuit", "集成电路", "半导体", "芯片", "国产替代"),
    "robotics_embodied_ai": ("robot", "robotics", "具身智能", "机器人"),
    "commercial_space": ("commercial space", "satellite", "商业航天", "卫星"),
    "six_g": ("6g", "通信", "下一代通信"),
    "power_grid_storage": ("power", "grid", "储能", "电力", "电网", "新能源"),
}

CHAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "gpu": ("gpu", "ai accelerator", "加速芯片"),
    "server": ("server", "服务器"),
    "pcb": ("pcb", "printed circuit", "ccl", "覆铜板"),
    "optical_module": ("optical module", "cpo", "光模块", "光芯片"),
    "advanced_packaging": ("advanced packaging", "先进封装"),
    "foundry": ("foundry", "wafer", "晶圆", "代工"),
    "liquid_cooling": ("liquid cooling", "液冷"),
    "power_equipment": ("power equipment", "ups", "电源", "电力设备"),
    "storage": ("hbm", "dram", "nand", "存储"),
}


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


def parse_policy_document(document: PolicyDocument) -> ParsedPolicyDocument:
    text = f"{document.title} {document.body}".lower()
    themes = tuple(theme for theme, keywords in THEME_KEYWORDS.items() if _contains_any(text, keywords))
    chain_nodes = tuple(node for node, keywords in CHAIN_KEYWORDS.items() if _contains_any(text, keywords))
    years = tuple(sorted({int(match) for match in re.findall(r"\b20[2-4][0-9]\b", text)}))
    subsidy_signal = _contains_any(text, ("subsidy", "tax", "财政", "补贴", "税收", "采购"))
    pilot_signal = _contains_any(text, ("pilot", "demo", "试点", "示范", "项目"))
    constraint_terms = tuple(term for term in ("risk", "compliance", "监管", "约束", "安全") if term in text)
    authority = policy_authority_score(document.source_level)
    confidence = min(1.0, 0.35 + 0.35 * authority + 0.05 * len(themes) + (0.05 if chain_nodes else 0.0))
    return ParsedPolicyDocument(
        document=document,
        authority_score=authority,
        themes=themes,
        chain_nodes=chain_nodes,
        target_years=years,
        subsidy_signal=subsidy_signal,
        pilot_signal=pilot_signal,
        constraint_terms=constraint_terms,
        confidence=confidence,
    )


def policy_to_evidence(parsed: ParsedPolicyDocument, as_of_date: str) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    event_type = EventType.SUBSIDY if parsed.subsidy_signal else EventType.POLICY_SUPPORT
    for theme in parsed.themes or ("unclassified_policy",):
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
            event_type=event_type,
            direction=1.0,
            magnitude=parsed.authority_score,
            confidence=parsed.confidence,
            evidence_quality=parsed.authority_score,
            source_reliability=parsed.authority_score,
            cross_validation_count=0,
            decay_half_life=_policy_half_life(parsed.document.source_level),
            horizon_days=_policy_horizon(parsed.document.source_level),
            rationale=parsed.document.title[:240],
            raw_reference={
                "document_id": parsed.document.document_id,
                "target_years": parsed.target_years,
                "chain_nodes": parsed.chain_nodes,
                "reference_hash": sha1((parsed.document.title + parsed.document.body).encode("utf-8")).hexdigest(),
                **parsed.document.raw_reference,
            },
            point_in_time_valid=bool(parsed.document.published_at <= as_of_date),
            risk_flags=tuple("policy_constraint" for _ in parsed.constraint_terms[:1]),
        ).with_hash()
        records.append(record)
    return records


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


def _policy_horizon(source_level: str) -> int:
    if source_level in {"central", "state_council"}:
        return 126
    if source_level == "ministry":
        return 90
    if source_level in {"provincial", "municipal"}:
        return 60
    return 20


def _policy_half_life(source_level: str) -> float:
    return max(5.0, _policy_horizon(source_level) / 2.0)
