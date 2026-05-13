from __future__ import annotations

from quantagent.v7.schemas import ChainEdge, ChainNode, ChainRelationType, ThemeProfile


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


def build_industry_chain_graph(profile: ThemeProfile) -> tuple[list[ChainNode], list[ChainEdge]]:
    nodes = list(AI_COMPUTE_TEMPLATE if profile.theme_name == "ai_compute" else GENERIC_THEME_TEMPLATE)
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


def _relation_for(source: str, target: str) -> ChainRelationType:
    if source in {"gpu", "domestic_gpu", "server", "data_center"}:
        return ChainRelationType.DIRECT_EXPOSURE
    if source in {"semiconductor_equipment", "hbm", "advanced_packaging"}:
        return ChainRelationType.CRITICAL_BOTTLENECK
    if target in {"server", "data_center"}:
        return ChainRelationType.INFRASTRUCTURE_DEPENDENCY
    return ChainRelationType.UPSTREAM_SUPPLIER
