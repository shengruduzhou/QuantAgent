"""Stock-pool hard gate: filters universe members before they reach alpha.

The original V7 pipeline used the :class:`StockPoolSelectionReport` only as
an *observability* artefact — the same universe (including weak-association
or false-association names) was still fed into the alpha model. That made
it possible for a poorly-scored member to leak into ``target_weights``.

This module turns the selection report into a pre-trade hard gate:

* Only members whose ``watchlist_status`` is in
  :data:`DEFAULT_ALLOWED_BUCKETS` (``core_beneficiary_pool`` and
  ``strong_correlation_pool``) survive.
* Optional ``allow_satellite_if_confidence_above`` lets ``OPTIONAL_SATELLITE``
  members pass when their ``source_confidence`` exceeds a threshold — this
  keeps high-conviction satellite names available without weakening the
  core/strong default.
* ``false_association`` and ``exclusion_pool`` symbols are always blocked.
* When ``require_factor_coverage=True`` the gate also drops any theme that
  has zero applicable production-stage factors for the horizon bucket; that
  prevents the alpha model from receiving members without any validated
  factor support.

The output is a tuple of (filtered_members, gate_log) where ``gate_log``
records each dropped symbol with the reason — needed for the daily audit.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantagent.v7.schemas import (
    ChainRelationType,
    StockPoolSelectionReport,
    ThematicUniverseMember,
    UniverseBucket,
)


DEFAULT_ALLOWED_BUCKETS: tuple[UniverseBucket, ...] = (
    UniverseBucket.CORE_BENEFICIARY,
    UniverseBucket.STRONG_CORRELATION,
)

DEFAULT_BLOCKED_RELATIONS: tuple[ChainRelationType, ...] = (
    ChainRelationType.FALSE_ASSOCIATION,
)


@dataclass(frozen=True)
class StockPoolGateConfig:
    allowed_buckets: tuple[UniverseBucket, ...] = DEFAULT_ALLOWED_BUCKETS
    allow_satellite_if_confidence_above: float = 0.75
    blocked_relations: tuple[ChainRelationType, ...] = DEFAULT_BLOCKED_RELATIONS
    require_factor_coverage: bool = True
    block_false_association: bool = True


def apply_stock_pool_gate(
    universe_members: list[ThematicUniverseMember],
    selection_reports: list[StockPoolSelectionReport],
    config: StockPoolGateConfig | None = None,
) -> tuple[list[ThematicUniverseMember], dict[str, str]]:
    """Filter universe members so only pool-approved names reach the alpha model.

    Returns the filtered member list and a per-symbol drop log:
    ``{symbol: drop_reason}``.
    """

    config = config or StockPoolGateConfig()
    factor_coverage_by_theme = {
        report.theme_name: bool(report.applicable_factor_names)
        for report in selection_reports
    }
    drop_log: dict[str, str] = {}
    kept: list[ThematicUniverseMember] = []
    allowed_buckets = set(config.allowed_buckets)
    blocked_relations = set(config.blocked_relations)
    for member in universe_members:
        if config.block_false_association and member.exposure_type in blocked_relations:
            drop_log[member.symbol] = f"blocked_relation:{member.exposure_type.value}"
            continue
        if member.watchlist_status == UniverseBucket.EXCLUSION:
            drop_log[member.symbol] = "excluded_bucket"
            continue
        if member.watchlist_status in allowed_buckets:
            survives = True
        elif (
            member.watchlist_status == UniverseBucket.OPTIONAL_SATELLITE
            and member.source_confidence >= config.allow_satellite_if_confidence_above
        ):
            survives = True
        else:
            survives = False
        if not survives:
            drop_log[member.symbol] = f"bucket:{member.watchlist_status.value}"
            continue
        if config.require_factor_coverage and not factor_coverage_by_theme.get(member.theme, True):
            drop_log[member.symbol] = "no_factor_coverage_for_theme"
            continue
        kept.append(member)
    return kept, drop_log


def gate_summary(drop_log: dict[str, str]) -> dict[str, int]:
    """Aggregate drop reasons into counts for the daily audit log."""

    counts: dict[str, int] = {}
    for reason in drop_log.values():
        bucket = reason.split(":", 1)[0]
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts
