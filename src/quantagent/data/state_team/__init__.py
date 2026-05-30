"""Stage 4.4 — state-team (国家队) inference data layer.

Output is ALWAYS labelled ``evidence_label = "inferred"`` — never
"confirmed". This module reads only public data (ETF flows, top-10
shareholder filings, large-trade reports) and emits a probability-
weighted signal of likely state-team buying. It does NOT claim
official confirmation, and downstream consumers must surface the
"inferred" label in any UI or report that uses this data.
"""

from quantagent.data.state_team.builder import (
    INFERENCE_REQUIRED_COLUMNS,
    EVIDENCE_TYPES,
    StateTeamInferenceBuilder,
    StateTeamInferenceConfig,
    StateTeamInferenceResult,
    build_state_team_inference,
    state_team_inference_for_features,
    infer_etf_concentrated_inflow,
    infer_post_crash_index_buying,
    infer_top10_holder_appearance,
    apply_state_team_features,
)
from quantagent.data.evidence.canonical import state_team_events_to_evidence

__all__ = [
    "INFERENCE_REQUIRED_COLUMNS",
    "EVIDENCE_TYPES",
    "StateTeamInferenceBuilder",
    "StateTeamInferenceConfig",
    "StateTeamInferenceResult",
    "build_state_team_inference",
    "state_team_inference_for_features",
    "state_team_events_to_evidence",
    "infer_etf_concentrated_inflow",
    "infer_post_crash_index_buying",
    "infer_top10_holder_appearance",
    "apply_state_team_features",
]
