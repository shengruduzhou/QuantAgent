"""Fundamental ranker data layer.

This package converts raw PIT fundamental metrics (valuation, quality,
growth) into a normalised cross-sectional ranking suitable for use as
an alpha overlay or sleeve filter. Like ``quantagent.data.sector`` it
is a data product, not a portfolio signal — consumers must route
through ``fundamental_ranker_for_overlay`` to respect the manifest
gate.
"""

from quantagent.data.fundamental.ranker import (
    FUNDAMENTAL_RANKER_REQUIRED_COLUMNS,
    FundamentalRankerBuilder,
    FundamentalRankerConfig,
    FundamentalRankerResult,
    build_fundamental_ranker,
    fundamental_ranker_for_overlay,
)

__all__ = [
    "FUNDAMENTAL_RANKER_REQUIRED_COLUMNS",
    "FundamentalRankerBuilder",
    "FundamentalRankerConfig",
    "FundamentalRankerResult",
    "build_fundamental_ranker",
    "fundamental_ranker_for_overlay",
]
