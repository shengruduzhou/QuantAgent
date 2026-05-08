from __future__ import annotations

import numpy as np
import pandas as pd


def ic_by_date(
    frame: pd.DataFrame,
    signal_column: str,
    return_column: str,
    date_column: str = "trade_date",
    method: str = "pearson",
) -> pd.Series:
    subset = frame[[date_column, signal_column, return_column]]
    return subset.groupby(date_column).apply(
        lambda x: x[signal_column].corr(x[return_column], method=method),
        include_groups=False,
    )


def rank_ic_by_date(
    frame: pd.DataFrame,
    signal_column: str,
    return_column: str,
    date_column: str = "trade_date",
) -> pd.Series:
    return ic_by_date(frame, signal_column, return_column, date_column, method="spearman")


def ic_summary(ic: pd.Series, max_lag: int | None = None) -> dict[str, float]:
    """IC summary with Newey-West HAC t-stat for autocorrelated IC series."""
    from quantagent.quant_math.performance import newey_west_t_stat

    clean = ic.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "ir": np.nan,
            "positive_ratio": np.nan,
            "t_stat": np.nan,
            "t_stat_nw": np.nan,
        }
    std = clean.std(ddof=1)
    naive_t = float(clean.mean() / (std / np.sqrt(len(clean)))) if std else np.nan
    return {
        "mean": float(clean.mean()),
        "std": float(std),
        "ir": float(clean.mean() / std) if std and not np.isnan(std) else np.nan,
        "positive_ratio": float((clean > 0).mean()),
        "t_stat": naive_t,
        "t_stat_nw": newey_west_t_stat(clean, max_lag=max_lag),
    }


def decay_curve(
    frame: pd.DataFrame,
    signal_column: str,
    return_columns: list[str],
    date_column: str = "trade_date",
) -> pd.DataFrame:
    rows = []
    for return_column in return_columns:
        rank_ic = rank_ic_by_date(frame, signal_column, return_column, date_column)
        summary = ic_summary(rank_ic)
        summary["return_column"] = return_column
        rows.append(summary)
    return pd.DataFrame(rows)


def dynamic_model_weights(
    model_ic: pd.DataFrame,
    model_column: str = "model",
    ic_column: str = "rank_ic",
    error_variance_column: str = "error_variance",
    min_ic: float = 0.0,
) -> pd.Series:
    """Weight models by positive IC divided by historical error variance."""
    latest = model_ic.copy()
    latest["edge"] = latest[ic_column].clip(lower=min_ic)
    latest["precision"] = 1.0 / (latest[error_variance_column].clip(lower=1e-12))
    raw = latest["edge"] * latest["precision"]
    if raw.sum() <= 0:
        return pd.Series(1.0 / len(latest), index=latest[model_column])
    return pd.Series(raw.to_numpy() / raw.sum(), index=latest[model_column])
