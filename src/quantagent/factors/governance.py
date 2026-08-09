from __future__ import annotations

import pandas as pd

from quantagent.factors.lifecycle import FactorLifecycleReport


def factor_group_metrics(
    reports: list[FactorLifecycleReport],
    group_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate one-window lifecycle diagnostics by factor group.

    ``FactorLifecycleReport`` may recommend ``validated`` but does not own the
    stateful ACTIVE decision.  Real active membership must come from
    ``FactorLifecycleLedger``; reporting an ``active_ratio`` from these reports
    would therefore be a false capital-allocation claim.
    """

    columns = [
        "group",
        "mean_rank_ic",
        "mean_rank_icir",
        "validated_ratio",
        "mean_turnover",
        "mean_capacity_proxy",
        "mean_crowding_proxy",
        "active_state_source",
    ]
    if not reports:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame([report.__dict__ for report in reports])
    frame["group"] = frame["factor_name"].map(group_map or {}).fillna("ungrouped")
    frame["is_validated"] = frame["recommended_status"].eq("validated").astype(float)
    result = (
        frame.groupby("group", sort=True)
        .agg(
            mean_rank_ic=("rolling_rank_ic", "mean"),
            mean_rank_icir=("rank_icir", "mean"),
            validated_ratio=("is_validated", "mean"),
            mean_turnover=("turnover", "mean"),
            mean_capacity_proxy=("capacity_proxy", "mean"),
            mean_crowding_proxy=("crowding_proxy", "mean"),
        )
        .reset_index()
    )
    result["active_state_source"] = "factor_lifecycle_ledger_required"
    return result[columns]
