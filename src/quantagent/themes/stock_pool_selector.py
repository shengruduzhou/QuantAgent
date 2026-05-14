from __future__ import annotations

from collections import defaultdict

from quantagent.v7.schemas import (
    ChainRelationType,
    FactorApplicability,
    InvestmentHorizonBucket,
    StockPoolSelectionReport,
    ThematicUniverseMember,
    ThemeLifecycleStage,
    ThemeProfile,
    UniverseBucket,
)


DIRECT_RELATIONS = {
    ChainRelationType.DIRECT_EXPOSURE,
    ChainRelationType.CRITICAL_BOTTLENECK,
    ChainRelationType.DOMESTIC_SUBSTITUTION,
    ChainRelationType.CUSTOMER_SUPPLIER_LINK,
    ChainRelationType.POLICY_BENEFICIARY,
}

STRONG_RELATIONS = {
    ChainRelationType.UPSTREAM_SUPPLIER,
    ChainRelationType.DOWNSTREAM_APPLICATION,
    ChainRelationType.INFRASTRUCTURE_DEPENDENCY,
    ChainRelationType.COST_BENEFICIARY,
    ChainRelationType.CAPACITY_EXPANSION,
    ChainRelationType.TECHNOLOGY_ENABLER,
}

FALSE_RELATIONS = {
    ChainRelationType.WEAK_ASSOCIATION,
    ChainRelationType.FALSE_ASSOCIATION,
}


def build_stock_pool_selection(
    universe_members: list[ThematicUniverseMember],
    theme_profiles: list[ThemeProfile],
    factor_applicability: list[FactorApplicability] | tuple[FactorApplicability, ...] = (),
    as_of_date: str = "",
) -> list[StockPoolSelectionReport]:
    """Build explicit theme stock pools and bind them to horizon-specific factor coverage."""
    if not universe_members:
        return []
    profile_by_theme = {profile.theme_name: profile for profile in theme_profiles}
    by_theme: dict[str, list[ThematicUniverseMember]] = defaultdict(list)
    for member in universe_members:
        by_theme[member.theme].append(member)

    reports: list[StockPoolSelectionReport] = []
    for theme, members in sorted(by_theme.items()):
        profile = profile_by_theme.get(theme)
        expected_horizon = profile.expected_horizon_days if profile else _member_horizon(members)
        horizon_bucket = classify_horizon_bucket(expected_horizon)
        lifecycle = profile.lifecycle_stage if profile else _dominant_lifecycle(members)
        reports.append(
            StockPoolSelectionReport(
                theme_name=theme,
                horizon_bucket=horizon_bucket,
                expected_horizon_days=expected_horizon,
                lifecycle_stage=lifecycle,
                core_symbols=_symbols(members, UniverseBucket.CORE_BENEFICIARY),
                strong_symbols=_symbols(members, UniverseBucket.STRONG_CORRELATION),
                satellite_symbols=_symbols(members, UniverseBucket.OPTIONAL_SATELLITE),
                watchlist_symbols=_symbols(members, UniverseBucket.WATCHLIST),
                exclusion_symbols=_symbols(members, UniverseBucket.EXCLUSION),
                direct_relation_symbols=_relation_symbols(members, DIRECT_RELATIONS),
                strong_relation_symbols=_relation_symbols(members, STRONG_RELATIONS),
                false_association_symbols=_relation_symbols(members, FALSE_RELATIONS),
                applicable_factor_names=_applicable_factor_names(theme, horizon_bucket, factor_applicability),
                revalidation_interval_days=_revalidation_interval(horizon_bucket, lifecycle),
                selection_rationale=_selection_rationale(theme, members, expected_horizon, lifecycle, as_of_date),
            )
        )
    return reports


def classify_horizon_bucket(expected_horizon_days: int) -> InvestmentHorizonBucket:
    if expected_horizon_days <= 20:
        return InvestmentHorizonBucket.SHORT_TERM
    if expected_horizon_days <= 60:
        return InvestmentHorizonBucket.MEDIUM_TERM
    return InvestmentHorizonBucket.LONG_TERM


def _symbols(members: list[ThematicUniverseMember], bucket: UniverseBucket) -> tuple[str, ...]:
    return tuple(sorted(member.symbol for member in members if member.watchlist_status == bucket))


def _relation_symbols(members: list[ThematicUniverseMember], relation_types: set[ChainRelationType]) -> tuple[str, ...]:
    return tuple(sorted(member.symbol for member in members if member.exposure_type in relation_types))


def _applicable_factor_names(
    theme: str,
    horizon_bucket: InvestmentHorizonBucket,
    factor_applicability: list[FactorApplicability] | tuple[FactorApplicability, ...],
) -> tuple[str, ...]:
    names = []
    for item in factor_applicability:
        if item.factor_lifecycle_stage not in {"production", "validation"}:
            continue
        if item.applicable_theme and theme not in item.applicable_theme:
            continue
        if classify_horizon_bucket(item.horizon_days) != horizon_bucket:
            continue
        names.append(item.factor_name)
    return tuple(sorted(set(names)))


def _revalidation_interval(horizon_bucket: InvestmentHorizonBucket, lifecycle: ThemeLifecycleStage) -> int:
    if lifecycle in {ThemeLifecycleStage.VALUATION_BUBBLE, ThemeLifecycleStage.DIVERGENCE, ThemeLifecycleStage.DECAY}:
        return 1
    if horizon_bucket == InvestmentHorizonBucket.SHORT_TERM:
        return 1
    if horizon_bucket == InvestmentHorizonBucket.MEDIUM_TERM:
        return 5
    return 20


def _selection_rationale(
    theme: str,
    members: list[ThematicUniverseMember],
    expected_horizon: int,
    lifecycle: ThemeLifecycleStage,
    as_of_date: str,
) -> str:
    core = len(_symbols(members, UniverseBucket.CORE_BENEFICIARY))
    strong = len(_symbols(members, UniverseBucket.STRONG_CORRELATION))
    excluded = len(_symbols(members, UniverseBucket.EXCLUSION))
    direct = len(_relation_symbols(members, DIRECT_RELATIONS))
    return (
        f"as_of={as_of_date}; theme={theme}; lifecycle={lifecycle.value}; "
        f"horizon={expected_horizon}d; core={core}; strong={strong}; "
        f"direct_relation={direct}; excluded={excluded}"
    )


def _member_horizon(members: list[ThematicUniverseMember]) -> int:
    if not members:
        return 20
    return int(max(member.membership_ttl_days for member in members))


def _dominant_lifecycle(members: list[ThematicUniverseMember]) -> ThemeLifecycleStage:
    if not members:
        return ThemeLifecycleStage.POLICY_SEED
    counts: dict[ThemeLifecycleStage, int] = defaultdict(int)
    for member in members:
        counts[member.theme_lifecycle_stage] += 1
    return max(counts, key=counts.get)
