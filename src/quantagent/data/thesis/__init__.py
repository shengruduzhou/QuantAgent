"""Capital-flow thesis layer.

Aggregates canonical evidence into testable theses keyed by
``direction`` (sector / theme / province). A thesis is the unit a
LLM agent or policy analyst may emit; the validation loop in
:mod:`.validation` re-scores its status from the realised post-event
panel returns.

Theses are never trading signals. Only the optimizer + RiskGate
chain may produce target_weights.
"""

from quantagent.data.thesis.builder import (
    CAPITAL_FLOW_THESIS_COLUMNS,
    CapitalFlowThesis,
    CapitalFlowThesisBuilder,
    CapitalFlowThesisConfig,
    THESIS_VALIDATION_STATES,
    build_capital_flow_theses,
    theses_to_frame,
)
from quantagent.data.thesis.validation import (
    ThesisValidationConfig,
    ThesisValidationResult,
    validate_thesis,
    validate_theses,
)


__all__ = [
    "CAPITAL_FLOW_THESIS_COLUMNS",
    "CapitalFlowThesis",
    "CapitalFlowThesisBuilder",
    "CapitalFlowThesisConfig",
    "THESIS_VALIDATION_STATES",
    "ThesisValidationConfig",
    "ThesisValidationResult",
    "build_capital_flow_theses",
    "theses_to_frame",
    "validate_thesis",
    "validate_theses",
]
