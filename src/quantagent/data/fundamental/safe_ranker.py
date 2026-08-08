"""Fail-closed public facade for the fundamental PIT ranker.

Historical fundamental ranking must never use a current industry snapshot as if
it were a point-in-time sector classification.  The legacy implementation can
accept a sector frame without ``available_at`` for diagnostics; that is useful
for current-state exploration but is not safe for historical ranking or a
weight overlay.

The public data-layer API therefore sanitises sector input before delegating to
the underlying scorer: a sector map without a valid ``available_at`` column is
dropped and the ranker falls back to its board proxy.  This makes missing PIT
sector history conservative rather than silently survivorship-biased.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from quantagent.data.fundamental import ranker as _base

FUNDAMENTAL_RANKER_REQUIRED_COLUMNS = _base.FUNDAMENTAL_RANKER_REQUIRED_COLUMNS
FundamentalRankerConfig = _base.FundamentalRankerConfig
FundamentalRankerResult = _base.FundamentalRankerResult
fundamental_ranker_for_overlay = _base.fundamental_ranker_for_overlay


def pit_safe_sector_map(sector_map: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return only a sector map that can support an as-of join.

    A current snapshot may contain ``coverage_status=current_snapshot`` but no
    publication/availability timestamp.  Such a frame is deliberately treated
    as unavailable for historical ranking.  Rows with invalid timestamps are
    removed; if none remain, the caller receives ``None`` and uses board proxy
    buckets instead.
    """
    if sector_map is None or sector_map.empty:
        return None
    if "available_at" not in sector_map.columns:
        return None
    safe = sector_map.copy()
    safe["available_at"] = pd.to_datetime(safe["available_at"], errors="coerce", utc=True).dt.tz_convert(None)
    safe = safe.dropna(subset=["available_at"])
    return safe if not safe.empty else None


def build_fundamental_ranker(
    metrics: pd.DataFrame,
    *,
    as_of_dates: Iterable[str | pd.Timestamp],
    sector_map: pd.DataFrame | None = None,
    config: FundamentalRankerConfig | None = None,
    generated_at: str | None = None,
) -> FundamentalRankerResult:
    return _base.build_fundamental_ranker(
        metrics,
        as_of_dates=as_of_dates,
        sector_map=pit_safe_sector_map(sector_map),
        config=config,
        generated_at=generated_at,
    )


class FundamentalRankerBuilder(_base.FundamentalRankerBuilder):
    """Builder whose production/public path always enforces PIT-safe sectors."""

    def build(
        self,
        metrics: pd.DataFrame,
        *,
        as_of_dates: Iterable[str | pd.Timestamp],
        sector_map: pd.DataFrame | None = None,
        generated_at: str | None = None,
    ) -> FundamentalRankerResult:
        result = build_fundamental_ranker(
            metrics,
            as_of_dates=as_of_dates,
            sector_map=sector_map,
            config=self.config,
            generated_at=generated_at,
        )
        gate = self._gate(result)
        coverage = dict(result.coverage)
        coverage["gate"] = gate
        return FundamentalRankerResult(
            frame=result.frame,
            coverage=coverage,
            validation=result.validation,
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
