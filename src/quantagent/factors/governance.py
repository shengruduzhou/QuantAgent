from __future__ import annotations

import pandas as pd

from quantagent.factors.lifecycle import FactorLifecycleReport


def factor_group_metrics(reports: list[FactorLifecycleReport], group_map: dict[str, str] | None = None) -> pd.DataFrame:
    if not reports:
        return pd.DataFrame(
            columns=[
                "group",
                "mean_rank_ic",
                "mean_rank_icir",
                "active_ratio",
                "mean_turnover",
                "mean_capacity_proxy",
                "mean_crowding_proxy",
            ]
        )
    frame = pd.DataFrame([report.__dict__ for report in reports])
    frame["group"] = frame["factor_name"].map(group_map or {}).fillna("ungrouped")
    frame["is_active"] = frame["recommended_status"].eq("active").astype(float)
    return (
        frame.groupby("group", sort=True)
        .agg(
            mean_rank_ic=("rolling_rank_ic", "mean"),
            mean_rank_icir=("rank_icir", "mean"),
            active_ratio=("is_active", "mean"),
            mean_turnover=("turnover", "mean"),
            mean_capacity_proxy=("capacity_proxy", "mean"),
            mean_crowding_proxy=("crowding_proxy", "mean"),
        )
        .reset_index()
    )
