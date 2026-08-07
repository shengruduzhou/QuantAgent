"""Forward-looking research reports and statistical selection governance."""

from quantagent.research.forward_report import (
    ForwardResearchContract,
    ForwardResearchValidation,
    PredictionWindow,
    build_forward_research_contract,
    render_forward_research_header,
    validate_forward_research_payload,
)
from quantagent.research.selection_governance import (
    NestedSelectionConfig,
    OuterFoldSelection,
    SelectionGovernanceReport,
    TrialRecord,
    TrialRegistry,
    nested_purged_select,
)

__all__ = [
    "ForwardResearchContract",
    "ForwardResearchValidation",
    "NestedSelectionConfig",
    "OuterFoldSelection",
    "PredictionWindow",
    "SelectionGovernanceReport",
    "TrialRecord",
    "TrialRegistry",
    "build_forward_research_contract",
    "nested_purged_select",
    "render_forward_research_header",
    "validate_forward_research_payload",
]
