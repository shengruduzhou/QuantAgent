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
    return frame.groupby(date_column).apply(
        lambda x: x[prediction_column].rank().corr(x[target_column].rank())
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
