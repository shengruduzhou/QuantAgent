from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_by_date(
    frame: pd.DataFrame,
    columns: list[str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    date_column: str = "trade_date",
) -> pd.DataFrame:
    data = frame.copy()
    for column in columns:
        bounds = data.groupby(date_column)[column].transform(
            lambda x: x.clip(x.quantile(lower_quantile), x.quantile(upper_quantile))
        )
        data[column] = bounds
    return data


def robust_zscore_by_date(
    frame: pd.DataFrame,
    columns: list[str],
    date_column: str = "trade_date",
    suffix: str = "_rz",
) -> pd.DataFrame:
    """Median/MAD z-score by cross section."""
    data = frame.copy()
    for column in columns:
        median = data.groupby(date_column)[column].transform("median")
        mad = data.groupby(date_column)[column].transform(lambda x: np.median(np.abs(x - np.median(x))))
        data[f"{column}{suffix}"] = (data[column] - median) / (1.4826 * mad + 1e-12)
    return data.replace([np.inf, -np.inf], np.nan)


def neutralize_cross_section(
    frame: pd.DataFrame,
    target_column: str,
    exposure_columns: list[str] | None = None,
    category_columns: list[str] | None = None,
    date_column: str = "trade_date",
    output_column: str | None = None,
) -> pd.DataFrame:
    """Regress a signal on exposures per date and keep residuals."""
    exposure_columns = exposure_columns or []
    category_columns = category_columns or []
    output_column = output_column or f"{target_column}_neutral"
    data = frame.copy()
    residuals = []
    for _, group in data.groupby(date_column, sort=False):
        y = group[target_column].astype(float)
        design_parts = [pd.Series(1.0, index=group.index, name="intercept")]
        if exposure_columns:
            design_parts.append(group[exposure_columns].astype(float))
        for category in category_columns:
            dummies = pd.get_dummies(group[category], prefix=category, dtype=float)
            if len(dummies.columns) > 1:
                dummies = dummies.iloc[:, 1:]
            design_parts.append(dummies)
        x = pd.concat(design_parts, axis=1)
        valid = ~(x.isna().any(axis=1) | y.isna())
        group_residual = pd.Series(np.nan, index=group.index, dtype=float)
        if valid.sum() > x.shape[1]:
            beta, *_ = np.linalg.lstsq(x.loc[valid].to_numpy(), y.loc[valid].to_numpy(), rcond=None)
            fitted = x.loc[valid].to_numpy() @ beta
            group_residual.loc[valid] = y.loc[valid] - fitted
        residuals.append(group_residual)
    data[output_column] = pd.concat(residuals).sort_index()
    return data
