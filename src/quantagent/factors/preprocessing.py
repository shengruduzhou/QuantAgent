from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from quantagent.quant_math.ashare import limit_down_mask, limit_up_mask, suspension_mask


def winsorize_by_date(
    frame: pd.DataFrame,
    columns: Sequence[str],
    n_mad: float = 5.0,
    date_column: str = "trade_date",
) -> pd.DataFrame:
    data = frame.copy()
    for column in columns:
        data[column] = data.groupby(date_column, sort=False)[column].transform(
            lambda s: _winsorize_series_mad(s, n_mad)
        )
    return data


def zscore_by_date(
    frame: pd.DataFrame,
    columns: Sequence[str],
    date_column: str = "trade_date",
    suffix: str = "",
) -> pd.DataFrame:
    data = frame.copy()
    for column in columns:
        target = f"{column}{suffix}" if suffix else column
        data[target] = data.groupby(date_column, sort=False)[column].transform(_zscore)
    return data.replace([np.inf, -np.inf], np.nan)


def rank_by_date(
    frame: pd.DataFrame,
    columns: Sequence[str],
    date_column: str = "trade_date",
    pct: bool = True,
    suffix: str = "",
) -> pd.DataFrame:
    data = frame.copy()
    for column in columns:
        target = f"{column}{suffix}" if suffix else column
        data[target] = data.groupby(date_column, sort=False)[column].rank(method="average", pct=pct)
    return data


def neutralize_by_date(
    frame: pd.DataFrame,
    target_column: str,
    exposure_columns: Sequence[str] | None = None,
    industry_column: str | None = None,
    date_column: str = "trade_date",
    output_column: str | None = None,
) -> pd.DataFrame:
    output_column = output_column or f"{target_column}_neutral"
    exposure_columns = tuple(exposure_columns or ())
    data = frame.copy()
    residuals: list[pd.Series] = []
    for _, group in data.groupby(date_column, sort=False):
        y = group[target_column].astype(float)
        design_parts = [pd.Series(1.0, index=group.index, name="intercept")]
        if exposure_columns:
            design_parts.append(group[list(exposure_columns)].astype(float))
        if industry_column is not None and industry_column in group.columns:
            dummies = pd.get_dummies(group[industry_column], prefix=industry_column, dtype=float)
            if len(dummies.columns) > 1:
                design_parts.append(dummies.iloc[:, 1:])
        x = pd.concat(design_parts, axis=1)
        valid = ~(x.isna().any(axis=1) | y.isna())
        resid = pd.Series(np.nan, index=group.index, dtype=float)
        if valid.sum() > x.loc[valid].shape[1]:
            beta, *_ = np.linalg.lstsq(x.loc[valid].to_numpy(), y.loc[valid].to_numpy(), rcond=None)
            resid.loc[valid] = y.loc[valid] - x.loc[valid].to_numpy() @ beta
        residuals.append(resid)
    data[output_column] = pd.concat(residuals).sort_index() if residuals else np.nan
    return data


def orthogonalize_factors(
    frame: pd.DataFrame,
    factor_columns: Sequence[str],
    date_column: str = "trade_date",
    suffix: str = "_orth",
) -> pd.DataFrame:
    data = frame.copy()
    output_columns = [f"{column}{suffix}" for column in factor_columns]
    for column in output_columns:
        data[column] = np.nan
    for _, group in data.groupby(date_column, sort=False):
        matrix = group[list(factor_columns)].astype(float)
        valid = ~matrix.isna().any(axis=1)
        if valid.sum() < len(factor_columns):
            continue
        centered = matrix.loc[valid] - matrix.loc[valid].mean(axis=0)
        q, _ = np.linalg.qr(centered.to_numpy())
        data.loc[matrix.loc[valid].index, output_columns] = q[:, : len(factor_columns)]
    return data


def fill_missing_by_industry_median(
    frame: pd.DataFrame,
    columns: Sequence[str],
    industry_column: str = "industry",
    date_column: str = "trade_date",
) -> pd.DataFrame:
    data = frame.copy()
    for column in columns:
        industry_median = data.groupby([date_column, industry_column], sort=False)[column].transform("median")
        date_median = data.groupby(date_column, sort=False)[column].transform("median")
        data[column] = data[column].fillna(industry_median).fillna(date_median)
    return data


def mask_untradable(
    frame: pd.DataFrame,
    columns: Sequence[str],
    tradable_column: str | None = None,
    date_column: str = "trade_date",
    symbol_column: str = "symbol",
) -> pd.DataFrame:
    data = frame.copy()
    if tradable_column and tradable_column in data.columns:
        tradable = data[tradable_column].fillna(False).astype(bool)
    else:
        ordered = data.sort_values([symbol_column, date_column]).copy()
        up = limit_up_mask(ordered, symbol_column=symbol_column).reindex(ordered.index)
        down = limit_down_mask(ordered, symbol_column=symbol_column).reindex(ordered.index)
        suspended = suspension_mask(ordered, symbol_column=symbol_column).reindex(ordered.index)
        tradable = (~up & ~down & ~suspended).reindex(data.index).fillna(False)
    for column in columns:
        data.loc[~tradable, column] = np.nan
    return data


def apply_factor_pipeline(
    frame: pd.DataFrame,
    factor_columns: Sequence[str],
    config: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    config = config or {}
    data = frame.copy()
    if config.get("fill_missing_by_industry", False):
        data = fill_missing_by_industry_median(
            data,
            factor_columns,
            industry_column=str(config.get("industry_column", "industry")),
        )
    if config.get("winsorize", True):
        data = winsorize_by_date(data, factor_columns, n_mad=float(config.get("n_mad", 5.0)))
    if config.get("neutralize", False):
        industry = config.get("industry_column", "industry")
        exposures = config.get("exposure_columns", ())
        exposure_columns = tuple(exposures) if isinstance(exposures, Sequence) and not isinstance(exposures, str) else ()
        for column in factor_columns:
            data = neutralize_by_date(
                data,
                column,
                exposure_columns=exposure_columns,
                industry_column=str(industry) if isinstance(industry, str) and industry in data.columns else None,
                output_column=column,
            )
    if config.get("zscore", True):
        data = zscore_by_date(data, factor_columns)
    if config.get("mask_untradable", True):
        data = mask_untradable(data, factor_columns, tradable_column=config.get("tradable_column"))
    return data


def _winsorize_series_mad(series: pd.Series, n_mad: float) -> pd.Series:
    median = series.median()
    mad = np.median(np.abs(series - median))
    if not np.isfinite(mad) or mad <= 1e-12:
        return series.astype(float)
    radius = n_mad * 1.4826 * mad
    return series.clip(median - radius, median + radius)


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (series - series.mean()) / std
