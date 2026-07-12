"""Stage 4 — policy event data layer.

The public API normalises policy timestamp columns to ``datetime64[ns]`` before
calling the builder. This prevents pandas object-dtype row reductions from
comparing ``NaT``/float sentinels with ``Timestamp`` values when malformed rows
are present.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from quantagent.data.policy.builder import (
    POLICY_EVENT_REQUIRED_COLUMNS,
    PolicyEventBuilder as _PolicyEventBuilder,
    PolicyEventConfig,
    PolicyEventResult,
    build_policy_events as _build_policy_events,
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


def _normalise_policy_clocks(raw: pd.DataFrame | None) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return raw
    frame = raw.copy()
    published_name = "published_at" if "published_at" in frame.columns else "announced_at"
    if published_name in frame.columns:
        published = pd.to_datetime(frame[published_name], errors="coerce")
        if "public_available_at" in frame.columns:
            public_available = pd.to_datetime(
                frame["public_available_at"], errors="coerce"
            )
            frame["public_available_at"] = public_available.where(
                public_available.notna(), published
            )
        else:
            frame["public_available_at"] = published

    if "ingested_at" in frame.columns:
        frame["ingested_at"] = pd.to_datetime(frame["ingested_at"], errors="coerce")
    elif "fetched_at" in frame.columns:
        frame["ingested_at"] = pd.to_datetime(frame["fetched_at"], errors="coerce")
    return frame


def build_policy_events(
    raw: pd.DataFrame,
    *,
    config: PolicyEventConfig | None = None,
) -> PolicyEventResult:
    return _build_policy_events(_normalise_policy_clocks(raw), config=config)


class PolicyEventBuilder(_PolicyEventBuilder):
    """Builder using the public timestamp-normalisation boundary."""

    def build(self, raw: pd.DataFrame) -> PolicyEventResult:
        return build_policy_events(raw, config=self.config)


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
