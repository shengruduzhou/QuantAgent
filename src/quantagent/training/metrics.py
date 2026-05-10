from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic_by_date(
    frame: pd.DataFrame,
    prediction_column: str,
    target_column: str,
    date_column: str = "trade_date",
) -> pd.Series:
    """Daily Spearman rank correlation between predictions and realized returns."""
    subset = frame[[date_column, prediction_column, target_column]]
    return subset.groupby(date_column).apply(
        lambda x: x[prediction_column].rank().corr(x[target_column].rank()),
        include_groups=False,
    )


def information_coefficient_summary(rank_ic: pd.Series) -> dict[str, float]:
    clean = rank_ic.dropna()
    if clean.empty:
        return {"rank_ic_mean": np.nan, "rank_ic_std": np.nan, "icir": np.nan}
    std = clean.std(ddof=1)
    return {
        "rank_ic_mean": float(clean.mean()),
        "rank_ic_std": float(std),
        "icir": float(clean.mean() / std) if std and not np.isnan(std) else np.nan,
    }


def alpha_evaluation_summary(
    frame: pd.DataFrame,
    prediction_column: str = "alpha",
    target_column: str = "target",
    weight_column: str | None = None,
) -> dict[str, float]:
    rank_ic = rank_ic_by_date(frame, prediction_column, target_column)
    ic = frame[prediction_column].corr(frame[target_column])
    turnover = np.nan
    if weight_column and weight_column in frame.columns:
        wide = frame.pivot_table(index="trade_date", columns="symbol", values=weight_column, aggfunc="last").fillna(0.0)
        turnover = float(wide.diff().abs().sum(axis=1).mean())
    summary = information_coefficient_summary(rank_ic)
    summary.update(
        {
            "ic": float(ic) if ic == ic else np.nan,
            "turnover": turnover,
            "calibration_error": float((frame[prediction_column] - frame[target_column]).abs().mean()),
        }
    )
    return summary
