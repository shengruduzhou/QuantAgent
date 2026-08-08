"""Forward-looking research reports and statistical selection governance."""

from quantagent.research.forward_report import (
    ForwardResearchContract,
    ForwardResearchValidation,
    PredictionWindow,
    build_forward_research_contract,
    render_forward_research_header,
    validate_forward_research_payload,
)
from quantagent.research.governed_model_comparison import (
    GovernedModelComparison,
    run_governed_model_comparison,
)
from quantagent.research.nonlinear_promotion import (
    NonlinearPromotionConfig,
    NonlinearPromotionReport,
    evaluate_nonlinear_promotion,
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
    "GovernedModelComparison",
    "NestedSelectionConfig",
    "NonlinearPromotionConfig",
    "NonlinearPromotionReport",
    "OuterFoldSelection",
    "PredictionWindow",
    "SelectionGovernanceReport",
    "TrialRecord",
    "TrialRegistry",
    "build_forward_research_contract",
    "evaluate_nonlinear_promotion",
    "nested_purged_select",
    "render_forward_research_header",
    "run_governed_model_comparison",
    "validate_forward_research_payload",
]
