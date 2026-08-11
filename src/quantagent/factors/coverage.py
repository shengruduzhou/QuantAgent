from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UsableFactorCoverage:
    """Coverage of finite factor/return pairs used by predictive validity gates."""

    counts_by_date: pd.Series
    eligible_dates: pd.DatetimeIndex
    coverage_dates: int
    median_symbols_per_date: float


def usable_factor_coverage(
    frame: pd.DataFrame,
    factor_column: str,
    return_column: str,
    *,
    min_symbols_per_date: int,
    date_column: str = "trade_date",
    symbol_column: str = "symbol",
) -> UsableFactorCoverage:
    """Count only finite factor/return pairs, never the surrounding market panel.

    A dense market panel must not make a sparse factor look well covered. Rows are
    eligible for predictive-validity coverage only when date, symbol, factor and
    target return are all present and the factor/return values are finite. Symbol
    counts are unique per date so accidental duplicate rows cannot inflate sample
    breadth (callers still own their stricter duplicate-row validation).
    """

    required = {date_column, symbol_column, factor_column, return_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"factor coverage frame missing columns: {missing}")
    if int(min_symbols_per_date) < 1:
        raise ValueError("min_symbols_per_date must be >= 1")

    work = frame[[date_column, symbol_column, factor_column, return_column]].copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work[factor_column] = pd.to_numeric(work[factor_column], errors="coerce")
    work[return_column] = pd.to_numeric(work[return_column], errors="coerce")
    work[[factor_column, return_column]] = work[[factor_column, return_column]].replace(
        [np.inf, -np.inf], np.nan
    )
    work = work.dropna(
        subset=[date_column, symbol_column, factor_column, return_column]
    )

    if work.empty:
        counts = pd.Series(dtype="int64")
        eligible_dates = pd.DatetimeIndex([], dtype="datetime64[ns]")
        return UsableFactorCoverage(counts, eligible_dates, 0, 0.0)

    counts = (
        work.groupby(date_column, sort=True)[symbol_column]
        .nunique()
        .astype("int64")
        .sort_index()
    )
    eligible = counts[counts >= int(min_symbols_per_date)]
    eligible_dates = pd.DatetimeIndex(eligible.index)
    return UsableFactorCoverage(
        counts_by_date=counts,
        eligible_dates=eligible_dates,
        coverage_dates=int(len(eligible)),
        median_symbols_per_date=float(counts.median()),
    )


__all__ = ["UsableFactorCoverage", "usable_factor_coverage"]
