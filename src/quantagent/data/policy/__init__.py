"""Stage 4 — policy event data layer.

Ingests policy announcements from official sources (CSRC, PBoC, MoF,
NDRC, State Council), tags them with themes + affected sectors, and
emits a PIT-safe silver dataset.

The module is a data product: consumers wanting to use policy events
as alpha overlays must route through
:func:`policy_events_for_features` which enforces the manifest gate.
"""

from quantagent.data.policy.builder import (
    POLICY_EVENT_REQUIRED_COLUMNS,
    PolicyEventBuilder,
    PolicyEventConfig,
    PolicyEventResult,
    build_policy_events,
    policy_events_for_features,
)
from quantagent.data.policy.theme_tagger import (
    POLICY_THEMES,
    SECTOR_KEYWORDS,
    tag_policy_event,
)
from quantagent.data.policy.time_lag import (
    TimeLagConfig,
    TimeLagResult,
    apply_policy_lag_features,
    estimate_policy_lag,
)
from quantagent.data.evidence.canonical import policy_events_to_evidence

__all__ = [
    "POLICY_EVENT_REQUIRED_COLUMNS",
    "POLICY_THEMES",
    "SECTOR_KEYWORDS",
    "PolicyEventBuilder",
    "PolicyEventConfig",
    "PolicyEventResult",
    "TimeLagConfig",
    "TimeLagResult",
    "apply_policy_lag_features",
    "build_policy_events",
    "estimate_policy_lag",
    "policy_events_for_features",
    "policy_events_to_evidence",
    "tag_policy_event",
]
