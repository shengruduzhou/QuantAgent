"""State-team (国家队) public-evidence inference layer.

Every output remains labelled ``evidence_label='inferred'``.  No public-data
pattern is treated as confirmation of an institution's current trading.
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
    "apply_state_team_features",
    "build_state_team_inference",
    "infer_etf_concentrated_inflow",
    "infer_post_crash_index_buying",
    "infer_top10_holder_appearance",
    "state_team_events_to_evidence",
    "state_team_inference_for_features",
]
