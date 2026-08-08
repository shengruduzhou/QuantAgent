"""Fundamental ranker data layer.

This package converts PIT fundamental metrics into a normalised cross-sectional
ranking.  The public API is fail-closed on sector history: a sector map without
a valid ``available_at`` timestamp is never allowed to masquerade as a
historical industry classification and falls back to board-proxy ranking.
"""

from quantagent.data.fundamental.safe_ranker import (
    FUNDAMENTAL_RANKER_REQUIRED_COLUMNS,
    FundamentalRankerBuilder,
    FundamentalRankerConfig,
    FundamentalRankerResult,
    build_fundamental_ranker,
    fundamental_ranker_for_overlay,
    pit_safe_sector_map,
)

__all__ = [
    "FUNDAMENTAL_RANKER_REQUIRED_COLUMNS",
    "FundamentalRankerBuilder",
    "FundamentalRankerConfig",
    "FundamentalRankerResult",
    "build_fundamental_ranker",
    "fundamental_ranker_for_overlay",
    "pit_safe_sector_map",
]
