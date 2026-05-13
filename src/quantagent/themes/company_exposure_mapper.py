from __future__ import annotations

from collections import Counter
from hashlib import sha1
import re

import pandas as pd

from quantagent.v7.schemas import ChainNode, ChainRelationType, EvidenceRecord


NODE_ALIASES: dict[str, tuple[str, ...]] = {
    "gpu": ("gpu", "ai accelerator", "加速芯片", "图形处理器"),
    "domestic_gpu": ("国产gpu", "国产加速芯片", "domestic gpu"),
    "server": ("server", "服务器", "整机", "算力服务器"),
    "pcb": ("pcb", "printed circuit", "印制电路板"),
    "ccl": ("ccl", "覆铜板"),
    "optical_module": ("optical module", "光模块", "800g", "1.6t"),
    "cpo": ("cpo", "co-packaged optics", "共封装光学"),
    "hbm": ("hbm", "dram", "存储", "memory"),
    "advanced_packaging": ("advanced packaging", "先进封装", "封测"),
    "foundry": ("foundry", "晶圆代工", "wafer"),
    "semiconductor_equipment": ("semiconductor equipment", "半导体设备", "刻蚀", "薄膜"),
    "data_center": ("data center", "数据中心", "智算中心"),
    "liquid_cooling": ("liquid cooling", "液冷"),
    "power_equipment": ("power equipment", "ups", "电源", "电力设备"),
    "energy_storage": ("energy storage", "储能"),
    "cloud_application": ("cloud", "大模型应用", "云计算"),
}


def map_company_exposures(
    company_profiles: pd.DataFrame,
    theme_name: str,
    chain_nodes: list[ChainNode],
    evidence: list[EvidenceRecord] | None = None,
    as_of_date: str = "",
) -> pd.DataFrame:
    """Infer company-theme-chain exposure from profile text and structured revenue fields."""
    if company_profiles.empty:
        return pd.DataFrame()
    evidence = evidence or []
    node_ids = {node.node_id for node in chain_nodes}
    rows: list[dict[str, object]] = []
    for _, company in company_profiles.iterrows():
        text = _company_text(company)
        scores = {node_id: _node_score(node_id, text) for node_id in node_ids}
        if not scores:
            continue
        node_id, raw_score = max(scores.items(), key=lambda item: item[1])
        revenue_exposure = _estimate_exposure(company, node_id, "revenue")
        profit_exposure = _estimate_exposure(company, node_id, "profit")
        evidence_count = _evidence_count(company, evidence, theme_name, node_id)
        source_confidence = min(0.95, 0.20 + raw_score * 0.35 + evidence_count * 0.08 + revenue_exposure * 0.35)
        exposure_score = min(100.0, 25.0 + raw_score * 35.0 + revenue_exposure * 35.0 + evidence_count * 5.0)
        relation = _relation_type(node_id, exposure_score, source_confidence, revenue_exposure)
        rows.append(
            {
                "symbol": str(company["symbol"]),
                "company_name": str(company.get("company_name", company["symbol"])),
                "theme": theme_name,
                "sub_theme": node_id,
                "chain_node": node_id,
                "exposure_type": relation.value,
                "exposure_score": exposure_score,
                "revenue_exposure_estimate": revenue_exposure,
                "profit_exposure_estimate": profit_exposure,
                "source_confidence": source_confidence,
                "evidence_count": evidence_count,
                "entry_date": as_of_date,
                "mapping_hash": sha1(f"{company.get('symbol')}:{theme_name}:{node_id}:{text[:256]}".encode("utf-8")).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


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


def _node_score(node_id: str, text: str) -> float:
    aliases = NODE_ALIASES.get(node_id, (node_id,))
    count = sum(len(re.findall(re.escape(alias.lower()), text)) for alias in aliases)
    if count <= 0:
        return 0.0
    weighted = count + sum(1 for alias in aliases if alias.lower() in text)
    return min(1.0, weighted / 4.0)


def _estimate_exposure(row: pd.Series, node_id: str, prefix: str) -> float:
    direct = row.get(f"{node_id}_{prefix}_exposure")
    if direct is not None and not pd.isna(direct):
        value = float(direct)
        return value / 100.0 if value > 1.0 else value
    generic = row.get(f"{prefix}_exposure_estimate")
    if generic is not None and not pd.isna(generic):
        value = float(generic)
        return value / 100.0 if value > 1.0 else value
    return 0.0


def _evidence_count(row: pd.Series, evidence: list[EvidenceRecord], theme: str, node_id: str) -> int:
    symbol = str(row["symbol"])
    count = sum(1 for item in evidence if item.symbol == symbol and item.theme == theme and (item.chain_node in {None, node_id}))
    source_tokens = Counter()
    for field in ("announcement_id", "source_id", "exchange_disclosure_id"):
        value = row.get(field)
        if value is not None and not pd.isna(value):
            source_tokens[str(value)] += 1
    return count + len(source_tokens)


def _relation_type(node_id: str, exposure_score: float, confidence: float, revenue_exposure: float) -> ChainRelationType:
    if exposure_score >= 80 and confidence >= 0.65 and revenue_exposure >= 0.20:
        if node_id in {"gpu", "domestic_gpu", "server", "foundry"}:
            return ChainRelationType.DIRECT_EXPOSURE
        if node_id in {"hbm", "advanced_packaging", "semiconductor_equipment"}:
            return ChainRelationType.CRITICAL_BOTTLENECK
        return ChainRelationType.UPSTREAM_SUPPLIER
    if exposure_score >= 55 and confidence >= 0.45:
        return ChainRelationType.STRONG_ASSOCIATION if hasattr(ChainRelationType, "STRONG_ASSOCIATION") else ChainRelationType.UPSTREAM_SUPPLIER
    if exposure_score >= 35:
        return ChainRelationType.WEAK_ASSOCIATION
    return ChainRelationType.FALSE_ASSOCIATION
