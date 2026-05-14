from __future__ import annotations

from quantagent.v7.schemas import ChainEdge, ChainNode, ChainRelationType, EvidenceRecord, ThemeProfile


AI_COMPUTE_TEMPLATE: tuple[ChainNode, ...] = (
    ChainNode("gpu", "GPU / AI accelerator", bottleneck_score=0.95, domestic_substitution_score=0.85, supply_shortage_score=0.75, profit_elasticity=0.80, demand_visibility=0.75, policy_support_score=0.85, technology_barrier=0.95, competition_intensity=0.60),
    ChainNode("domestic_gpu", "Domestic GPU", upstream_nodes=("semiconductor_equipment",), bottleneck_score=0.90, domestic_substitution_score=0.95, supply_shortage_score=0.80, profit_elasticity=0.85, demand_visibility=0.70, policy_support_score=0.90, technology_barrier=0.95, competition_intensity=0.55),
    ChainNode("server", "AI server", upstream_nodes=("gpu", "pcb", "storage"), downstream_nodes=("data_center",), dependency_strength=0.85, demand_visibility=0.80, policy_support_score=0.75, competition_intensity=0.70),
    ChainNode("pcb", "PCB / CCL", upstream_nodes=("ccl",), downstream_nodes=("server",), dependency_strength=0.70, price_elasticity=0.55, profit_elasticity=0.65, demand_visibility=0.65, competition_intensity=0.75),
    ChainNode("ccl", "Copper clad laminate", downstream_nodes=("pcb",), dependency_strength=0.60, price_elasticity=0.60, profit_elasticity=0.55, demand_visibility=0.55, competition_intensity=0.70),
    ChainNode("optical_module", "Optical module", upstream_nodes=("optical_chip",), downstream_nodes=("data_center",), dependency_strength=0.82, bottleneck_score=0.75, domestic_substitution_score=0.65, supply_shortage_score=0.70, profit_elasticity=0.75, demand_visibility=0.75),
    ChainNode("cpo", "CPO", upstream_nodes=("optical_chip",), downstream_nodes=("data_center",), dependency_strength=0.72, bottleneck_score=0.80, technology_barrier=0.85, demand_visibility=0.60),
    ChainNode("hbm", "HBM / advanced memory", downstream_nodes=("gpu",), dependency_strength=0.80, bottleneck_score=0.88, supply_shortage_score=0.85, price_elasticity=0.70, technology_barrier=0.90),
    ChainNode("advanced_packaging", "Advanced packaging", downstream_nodes=("gpu", "hbm"), dependency_strength=0.76, bottleneck_score=0.72, domestic_substitution_score=0.80, technology_barrier=0.82, demand_visibility=0.68),
    ChainNode("foundry", "Wafer foundry", upstream_nodes=("semiconductor_equipment",), downstream_nodes=("gpu", "optical_chip"), dependency_strength=0.88, bottleneck_score=0.82, domestic_substitution_score=0.86, technology_barrier=0.95),
    ChainNode("semiconductor_equipment", "Semiconductor equipment", downstream_nodes=("foundry", "advanced_packaging"), bottleneck_score=0.90, domestic_substitution_score=0.90, policy_support_score=0.85, technology_barrier=0.92),
    ChainNode("data_center", "Data center", upstream_nodes=("server", "optical_module", "liquid_cooling", "power_equipment"), dependency_strength=0.80, demand_visibility=0.85, policy_support_score=0.75),
    ChainNode("liquid_cooling", "Liquid cooling", downstream_nodes=("data_center",), dependency_strength=0.62, supply_shortage_score=0.45, profit_elasticity=0.55, demand_visibility=0.60),
    ChainNode("power_equipment", "Power equipment / UPS", downstream_nodes=("data_center",), dependency_strength=0.68, demand_visibility=0.66, policy_support_score=0.70),
    ChainNode("energy_storage", "Energy storage", downstream_nodes=("data_center", "power_equipment"), dependency_strength=0.58, demand_visibility=0.60, policy_support_score=0.72),
    ChainNode("cloud_application", "Cloud and model applications", upstream_nodes=("data_center",), dependency_strength=0.52, demand_visibility=0.64, competition_intensity=0.80),
)


GENERIC_THEME_TEMPLATE: tuple[ChainNode, ...] = (
    ChainNode("core_product", "Core product", dependency_strength=0.70, demand_visibility=0.55, policy_support_score=0.60),
    ChainNode("upstream_material", "Upstream material", dependency_strength=0.50, price_elasticity=0.45, competition_intensity=0.65),
    ChainNode("equipment", "Equipment", dependency_strength=0.60, technology_barrier=0.60, policy_support_score=0.55),
    ChainNode("downstream_application", "Downstream application", dependency_strength=0.45, demand_visibility=0.50, competition_intensity=0.70),
)


def build_industry_chain_graph(profile: ThemeProfile, evidence: list[EvidenceRecord] | None = None) -> tuple[list[ChainNode], list[ChainEdge]]:
    evidence = evidence or []
    nodes = list(AI_COMPUTE_TEMPLATE if profile.theme_name == "ai_compute" else GENERIC_THEME_TEMPLATE)
    nodes = _merge_evidence_nodes(nodes, profile.theme_name, evidence)
    edges: list[ChainEdge] = []
    node_ids = {node.node_id for node in nodes}
    for node in nodes:
        for upstream in node.upstream_nodes:
            if upstream in node_ids:
                edges.append(ChainEdge(upstream, node.node_id, _relation_for(upstream, node.node_id), max(node.dependency_strength, 0.50)))
        for downstream in node.downstream_nodes:
            if downstream in node_ids:
                edges.append(ChainEdge(node.node_id, downstream, _relation_for(node.node_id, downstream), max(node.dependency_strength, 0.50)))
    return nodes, edges


def _merge_evidence_nodes(nodes: list[ChainNode], theme: str, evidence: list[EvidenceRecord]) -> list[ChainNode]:
    by_id = {node.node_id: node for node in nodes}
    evidence_by_node: dict[str, list[str]] = {}
    for record in evidence:
        if record.theme not in {None, theme}:
            continue
        chain_nodes = []
        if record.chain_node:
            chain_nodes.append(record.chain_node)
        raw_nodes = record.raw_reference.get("chain_nodes", ())
        if isinstance(raw_nodes, str):
            chain_nodes.extend(item.strip() for item in raw_nodes.split(",") if item.strip())
        elif raw_nodes:
            chain_nodes.extend(str(item) for item in raw_nodes)
        for node_id in chain_nodes:
            evidence_by_node.setdefault(node_id, []).append(record.evidence_id)
    for node_id, evidence_ids in evidence_by_node.items():
        if node_id in by_id:
            existing = by_id[node_id]
            by_id[node_id] = ChainNode(
                node_id=existing.node_id,
                node_name=existing.node_name,
                upstream_nodes=existing.upstream_nodes,
                downstream_nodes=existing.downstream_nodes,
                dependency_strength=existing.dependency_strength,
                bottleneck_score=existing.bottleneck_score,
                domestic_substitution_score=existing.domestic_substitution_score,
                supply_shortage_score=existing.supply_shortage_score,
                price_elasticity=existing.price_elasticity,
                profit_elasticity=existing.profit_elasticity,
                demand_visibility=existing.demand_visibility,
                policy_support_score=max(existing.policy_support_score, 0.60),
                technology_barrier=existing.technology_barrier,
                competition_intensity=existing.competition_intensity,
                listed_company_count=existing.listed_company_count,
                evidence_ids=tuple(sorted(set(existing.evidence_ids + tuple(evidence_ids)))),
            )
        else:
            by_id[node_id] = ChainNode(
                node_id=node_id,
                node_name=node_id.replace("_", " "),
                dependency_strength=0.45,
                demand_visibility=0.45,
                policy_support_score=0.55,
                evidence_ids=tuple(sorted(set(evidence_ids))),
            )
    return list(by_id.values())


def _relation_for(source: str, target: str) -> ChainRelationType:
    if source in {"gpu", "domestic_gpu", "server", "data_center"}:
        return ChainRelationType.DIRECT_EXPOSURE
    if source in {"semiconductor_equipment", "hbm", "advanced_packaging"}:
        return ChainRelationType.CRITICAL_BOTTLENECK
    if target in {"server", "data_center"}:
        return ChainRelationType.INFRASTRUCTURE_DEPENDENCY
    return ChainRelationType.UPSTREAM_SUPPLIER
