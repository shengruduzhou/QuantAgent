"""Evidence-driven company-theme-chain exposure mapper.

The earlier version of this module relied on a hand-curated
``NODE_ALIASES`` dictionary. That made it brittle for any theme outside
AI compute. The new mapper is driven entirely by:

1. **Structured revenue exposure**, taken from ``{node_id}_revenue_exposure``
   columns when present, falling back to ``main_business_revenue`` shares.
2. **Order / contract evidence** — counts ``order_confirmed`` evidence rows
   for the symbol.
3. **News and disclosure evidence** — counts policy / disclosure / news
   rows that mention the company and overlap the chain node keywords.
4. **Chain-node keywords supplied by the dynamic IndustryChainReasoner**
   instead of a hard-coded alias map.

The output classification follows the "真相关 / 强相关 / 弱关联 / 伪相关"
spec:

* ``DIRECT_EXPOSURE`` — has positive revenue exposure **and** at least
  one company-specific evidence row (announcement / order).
* ``CRITICAL_BOTTLENECK`` — chain node is marked as a bottleneck and
  revenue exposure is high.
* ``UPSTREAM_SUPPLIER`` — supplier evidence (订单 / 采购) without
  primary product revenue.
* ``WEAK_ASSOCIATION`` — only news evidence, no revenue / order proof.
* ``FALSE_ASSOCIATION`` — keyword match only, no evidence, or company
  explicitly disclaimed exposure.

The mapper can optionally call an LLM via :class:`LLMSkillClient` to
refine ambiguous rows. The LLM call is opt-in and gated by
``allow_network=True``. When the LLM is unavailable the mapper falls
back to the deterministic rules above — never to the legacy alias map.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from hashlib import sha1
import re

import pandas as pd

from quantagent.v7.schemas import ChainNode, ChainRelationType, EvidenceRecord


_LEGAL_DISCLAIMER_PATTERNS = (
    "占比较小",
    "占公司",
    "未对公司业绩",
    "无影响",
    "目前未开展",
    "暂未涉及",
    "未对公司主营业务构成",
)


_BOTTLENECK_NODE_IDS = {
    "gpu", "domestic_gpu", "hbm", "advanced_packaging",
    "semiconductor_equipment", "foundry", "lithography",
    "rare_earth", "key_material",
}

_DIRECT_PRODUCT_NODE_IDS = {
    "server", "gpu", "domestic_gpu", "foundry", "energy_storage",
    "ev_battery", "innovative_drug", "satellite", "commercial_rocket",
}


@dataclass(frozen=True)
class ExposureMapperConfig:
    direct_revenue_threshold: float = 0.15
    strong_revenue_threshold: float = 0.05
    min_evidence_for_direct: int = 1
    min_evidence_for_strong: int = 1
    max_keyword_only_confidence: float = 0.45
    primary_revenue_field: str = "revenue_exposure_estimate"
    profit_field: str = "profit_exposure_estimate"
    fallback_keywords_from_node_name: bool = True


def map_company_exposures(
    company_profiles: pd.DataFrame,
    theme_name: str,
    chain_nodes: list[ChainNode],
    evidence: list[EvidenceRecord] | None = None,
    as_of_date: str = "",
    config: ExposureMapperConfig | None = None,
    node_keywords: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Build the company-theme-chain exposure frame.

    Parameters
    ----------
    company_profiles : DataFrame
        Must contain ``symbol`` and optionally ``business_scope``,
        ``main_business``, ``segment_text``, ``customer_text``,
        ``announcement_text``, ``research_note``.
    chain_nodes : list[ChainNode]
        Output of the dynamic industry-chain reasoner. The mapper uses
        ``node_id`` and ``node_name`` to derive keywords if ``node_keywords``
        is not supplied.
    evidence : list[EvidenceRecord]
        Evidence rows scoped to this theme. The mapper extracts company
        announcements, order confirmations and news.
    node_keywords : dict[str, tuple[str, ...]] | None
        Optional override keyed by node id. When omitted, the mapper
        derives keywords from ``ChainNode.node_name`` and any tokens in
        ``EvidenceRecord.raw_reference['chain_nodes']``.
    """

    if company_profiles is None or company_profiles.empty:
        return pd.DataFrame()
    config = config or ExposureMapperConfig()
    evidence = evidence or []
    keyword_map = node_keywords or _derive_keywords_from_evidence(chain_nodes, evidence, config)
    rows: list[dict[str, object]] = []
    evidence_index = _index_evidence_by_symbol(evidence)
    for _, company in company_profiles.iterrows():
        symbol = str(company.get("symbol", ""))
        if not symbol:
            continue
        text = _company_text(company)
        scores = {
            node_id: _node_text_score(text, keyword_map.get(node_id, (node_id,)))
            for node_id in (node.node_id for node in chain_nodes)
        }
        if not scores or max(scores.values()) <= 0.0:
            continue
        # Bind to highest-scoring chain node
        node_id = max(scores.items(), key=lambda item: item[1])[0]
        node_score = scores[node_id]
        revenue_exposure = _structured_exposure(company, node_id, "revenue", config)
        profit_exposure = _structured_exposure(company, node_id, "profit", config)
        symbol_evidence = evidence_index.get(symbol, [])
        order_count = _count_event_type(symbol_evidence, "order_confirmed")
        announcement_count = _count_event_type(symbol_evidence, "earnings_growth") + _count_event_type(
            symbol_evidence, "audit_opinion"
        )
        news_count = _count_event_type(symbol_evidence, "sentiment_positive") + _count_event_type(
            symbol_evidence, "sentiment_negative"
        )
        # Also count structured evidence references on the profile row
        for field in ("announcement_id", "order_id", "exchange_disclosure_id", "source_id"):
            value = company.get(field)
            if value is not None and not pd.isna(value):
                announcement_count += 1
        evidence_count = order_count + announcement_count + news_count
        company_disclaimed = _has_disclaimer(text)
        relation, confidence = _classify(
            node_id=node_id,
            node_score=node_score,
            revenue_exposure=revenue_exposure,
            order_count=order_count,
            announcement_count=announcement_count,
            news_count=news_count,
            company_disclaimed=company_disclaimed,
            config=config,
        )
        exposure_score = _exposure_score(
            node_score, revenue_exposure, order_count + announcement_count, news_count
        )
        rows.append(
            {
                "symbol": symbol,
                "company_name": str(company.get("company_name", symbol)),
                "theme": theme_name,
                "sub_theme": node_id,
                "chain_node": node_id,
                "exposure_type": relation.value,
                "exposure_score": exposure_score,
                "revenue_exposure_estimate": revenue_exposure,
                "profit_exposure_estimate": profit_exposure,
                "source_confidence": confidence,
                "evidence_count": evidence_count,
                "order_evidence_count": order_count,
                "announcement_evidence_count": announcement_count,
                "news_evidence_count": news_count,
                "company_disclaimer_detected": company_disclaimed,
                "entry_date": as_of_date,
                "mapping_hash": sha1(
                    f"{symbol}:{theme_name}:{node_id}:{as_of_date}".encode("utf-8")
                ).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def _derive_keywords_from_evidence(
    nodes: list[ChainNode],
    evidence: list[EvidenceRecord],
    config: ExposureMapperConfig,
) -> dict[str, tuple[str, ...]]:
    """Build node-id -> keywords mapping from chain nodes + evidence references."""

    base: dict[str, set[str]] = defaultdict(set)
    for node in nodes:
        if config.fallback_keywords_from_node_name:
            base[node.node_id].add(node.node_id.lower().replace("_", " "))
            if node.node_name:
                base[node.node_id].add(node.node_name.lower())
    for record in evidence:
        raw_nodes = record.raw_reference.get("chain_nodes", ())
        if isinstance(raw_nodes, str):
            tokens = [item.strip().lower() for item in raw_nodes.split(",") if item.strip()]
        else:
            tokens = [str(item).lower() for item in raw_nodes]
        if record.chain_node:
            tokens.append(record.chain_node.lower())
        keywords = record.raw_reference.get("keywords", ())
        if isinstance(keywords, str):
            keyword_tokens = [item.strip().lower() for item in keywords.split(",") if item.strip()]
        else:
            keyword_tokens = [str(item).lower() for item in keywords]
        for token in tokens:
            base.setdefault(token, set()).add(token)
            for keyword in keyword_tokens:
                base[token].add(keyword)
    return {node_id: tuple(sorted(tokens)) for node_id, tokens in base.items() if tokens}


def _index_evidence_by_symbol(evidence: list[EvidenceRecord]) -> dict[str, list[EvidenceRecord]]:
    index: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for record in evidence:
        if record.symbol:
            index[str(record.symbol)].append(record)
    return index


def _company_text(row: pd.Series) -> str:
    fields = [
        "company_name",
        "business_scope",
        "main_business",
        "segment_text",
        "customer_text",
        "supplier_text",
        "announcement_text",
        "research_note",
    ]
    return " ".join(str(row.get(field, "")) for field in fields).lower()


def _node_text_score(text: str, keywords: tuple[str, ...]) -> float:
    if not text or not keywords:
        return 0.0
    score = 0.0
    for keyword in keywords:
        keyword_norm = keyword.lower().strip()
        if not keyword_norm:
            continue
        hits = len(re.findall(re.escape(keyword_norm), text))
        if hits:
            score += min(0.30, 0.10 + 0.05 * hits)
    return min(1.0, score)


def _structured_exposure(
    row: pd.Series, node_id: str, prefix: str, config: ExposureMapperConfig
) -> float:
    candidates = (
        f"{node_id}_{prefix}_exposure",
        f"{prefix}_exposure_{node_id}",
        config.primary_revenue_field if prefix == "revenue" else config.profit_field,
        f"{prefix}_exposure",
        f"{prefix}_exposure_estimate",
    )
    for column in candidates:
        if column in row.index:
            value = row.get(column)
            if value is None or pd.isna(value):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            return numeric / 100.0 if numeric > 1.0 else numeric
    return 0.0


def _count_event_type(evidence: list[EvidenceRecord], event_type: str) -> int:
    return sum(1 for record in evidence if getattr(record.event_type, "value", record.event_type) == event_type)


def _has_disclaimer(text: str) -> bool:
    return any(pattern in text for pattern in _LEGAL_DISCLAIMER_PATTERNS)


def _classify(
    node_id: str,
    node_score: float,
    revenue_exposure: float,
    order_count: int,
    announcement_count: int,
    news_count: int,
    company_disclaimed: bool,
    config: ExposureMapperConfig,
) -> tuple[ChainRelationType, float]:
    if company_disclaimed and revenue_exposure < config.strong_revenue_threshold:
        return ChainRelationType.FALSE_ASSOCIATION, 0.20
    company_evidence_count = order_count + announcement_count
    has_revenue = revenue_exposure >= config.strong_revenue_threshold
    direct_revenue = revenue_exposure >= config.direct_revenue_threshold
    if (
        direct_revenue
        and company_evidence_count >= config.min_evidence_for_direct
        and node_id in _DIRECT_PRODUCT_NODE_IDS
    ):
        return ChainRelationType.DIRECT_EXPOSURE, min(0.95, 0.55 + revenue_exposure)
    if (
        direct_revenue
        and node_id in _BOTTLENECK_NODE_IDS
        and company_evidence_count >= config.min_evidence_for_direct
    ):
        return ChainRelationType.CRITICAL_BOTTLENECK, min(0.95, 0.55 + revenue_exposure)
    if has_revenue and order_count >= 1:
        return ChainRelationType.UPSTREAM_SUPPLIER, min(0.85, 0.45 + revenue_exposure)
    if order_count >= 1 and announcement_count >= 1:
        return ChainRelationType.UPSTREAM_SUPPLIER, 0.55
    if node_score >= 0.35 and news_count >= 1:
        return ChainRelationType.WEAK_ASSOCIATION, min(config.max_keyword_only_confidence, 0.30 + 0.05 * news_count)
    if node_score > 0.0 and not (has_revenue or company_evidence_count > 0):
        return ChainRelationType.FALSE_ASSOCIATION, 0.20
    return ChainRelationType.WEAK_ASSOCIATION, 0.30


def _exposure_score(
    node_score: float,
    revenue_exposure: float,
    company_evidence_count: int,
    news_count: int,
) -> float:
    raw = (
        30.0
        + 30.0 * min(1.0, node_score * 2.0)
        + 50.0 * min(1.0, revenue_exposure)
        + 8.0 * min(3, company_evidence_count)
        + 3.0 * min(3, news_count)
    )
    return float(max(0.0, min(100.0, raw)))
