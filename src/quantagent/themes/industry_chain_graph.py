"""Industry-chain graph entrypoint.

The previous version of this module shipped a hand-curated
``AI_COMPUTE_TEMPLATE`` chain template. Per the V7 evidence-driven design
the template is gone — :func:`build_industry_chain_graph` is now a thin
adapter that delegates to :func:`industry_chain_reasoner.reason_industry_chain`.
When evidence is empty the function returns empty ``nodes`` / ``edges`` so
the downstream stock-pool gate refuses to populate a theme with no real
proof.
"""

from __future__ import annotations

from quantagent.themes.industry_chain_reasoner import (
    IndustryChainReasonerConfig,
    reason_industry_chain,
)
from quantagent.v7.schemas import ChainEdge, ChainNode, EvidenceRecord, ThemeProfile


def build_industry_chain_graph(
    profile: ThemeProfile,
    evidence: list[EvidenceRecord] | None = None,
    config: IndustryChainReasonerConfig | None = None,
) -> tuple[list[ChainNode], list[ChainEdge]]:
    """Return the (nodes, edges) tuple for the given theme.

    Evidence-driven: when ``evidence`` is empty the reasoner returns empty
    lists. The legacy template-based fallback has been removed.
    """

    result = reason_industry_chain(profile, evidence or [], config)
    return result.nodes, result.edges
