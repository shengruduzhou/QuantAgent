"""Sector mapping + ST flag data layer (Stage 2.2).

PIT-safe note
-------------
AkShare publishes the **current** industry classification only. There is
no public historical "industry-as-of-T" feed cheap enough to integrate
here. Therefore every sector mapping row produced by this module is
explicitly labelled as a current snapshot via the ``coverage_status``
column:

* ``current_snapshot`` — the row reflects today's classification only.
  Safe to use for live trading and stratified diagnostics ON CURRENT
  DATA, but **must not** be used to bucket OOS historical alpha (a
  stock reclassified between then and now would be wrongly bucketed
  and would silently leak future information into the analysis).
* ``pit_historical`` — reserved for future sources that carry an
  actual point-in-time classification timeline.

Same convention for the ST flag table: ``st_source`` records the
provider, and ``available_at`` records when the flag became
observable.
"""

from quantagent.data.sector.sector_mapping import (
    BOARD_PROXY_SOURCE,
    SOURCE_PRIORITY,
    SectorMapBuilder,
    SectorMapConfig,
    SectorMapResult,
    board_proxy_rows,
    coverage_report,
    normalize_sector_source,
    sector_coverage_gate,
    source_conflict_report,
    source_priority_rank,
    source_priority_report,
    validate_sector_map,
)
from quantagent.data.sector.st_history import (
    StFlagBuilder,
    StFlagConfig,
    StFlagResult,
    coverage_report_st,
    st_coverage_gate,
    validate_st_table,
)
from quantagent.data.sector.sector_pool import (
    DEFAULT_TIER_WEIGHTS,
    SECTOR_POOL_REQUIRED_COLUMNS,
    SectorPoolBuilder,
    SectorPoolConfig,
    SectorPoolResult,
    VALID_POOL_TIERS,
    build_sector_pool,
    sector_pool_for_weight_overlay,
)

__all__ = [
    "SectorMapBuilder",
    "SectorMapConfig",
    "SectorMapResult",
    "BOARD_PROXY_SOURCE",
    "SOURCE_PRIORITY",
    "board_proxy_rows",
    "coverage_report",
    "normalize_sector_source",
    "sector_coverage_gate",
    "source_conflict_report",
    "source_priority_rank",
    "source_priority_report",
    "validate_sector_map",
    "StFlagBuilder",
    "StFlagConfig",
    "StFlagResult",
    "coverage_report_st",
    "st_coverage_gate",
    "validate_st_table",
    "DEFAULT_TIER_WEIGHTS",
    "SECTOR_POOL_REQUIRED_COLUMNS",
    "SectorPoolBuilder",
    "SectorPoolConfig",
    "SectorPoolResult",
    "VALID_POOL_TIERS",
    "build_sector_pool",
    "sector_pool_for_weight_overlay",
]
