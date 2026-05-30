"""Stage 4.3 — bond-market flow data layer.

Daily macro features from the China bond market — yield curve points,
term spreads, credit spreads, interbank repo rate, bond-fund flows.
Joined to the equity panel via merge_asof backward so a 2020 OOS row
never sees a 2026 yield print.
"""

from quantagent.data.bond.builder import (
    BOND_FLOW_REQUIRED_COLUMNS,
    BondFlowBuilder,
    BondFlowConfig,
    BondFlowResult,
    apply_bond_flow_features,
    bond_flows_for_features,
    build_bond_flows,
)
from quantagent.data.evidence.canonical import bond_flows_to_evidence

__all__ = [
    "BOND_FLOW_REQUIRED_COLUMNS",
    "BondFlowBuilder",
    "BondFlowConfig",
    "BondFlowResult",
    "apply_bond_flow_features",
    "bond_flows_for_features",
    "bond_flows_to_evidence",
    "build_bond_flows",
]
