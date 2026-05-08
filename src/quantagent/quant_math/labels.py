from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_return_labels(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5, 20, 60),
    price_column: str = "close",
    benchmark_price_column: str | None = None,
    date_column: str = "trade_date",
    symbol_column: str = "symbol",
) -> pd.DataFrame:
    """Add future log-return and optional benchmark-relative labels."""
    data = frame.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values([symbol_column, date_column]).reset_index(drop=True)
    grouped = data.groupby(symbol_column, group_keys=False)

    for horizon in horizons:
        future_price = grouped[price_column].shift(-horizon)
        data[f"future_{horizon}d_log_return"] = np.log(future_price / data[price_column])
        if benchmark_price_column:
            benchmark_return = _future_benchmark_log_return(
                data,
                benchmark_price_column,
                horizon,
                date_column,
            )
            data[f"future_{horizon}d_log_excess_return"] = (
                data[f"future_{horizon}d_log_return"] - benchmark_return
            )
    return data.replace([np.inf, -np.inf], np.nan)


def add_future_risk_labels(
    frame: pd.DataFrame,
    horizons: tuple[int, ...] = (5, 20, 60),
    return_column: str = "ret_1d",
    price_column: str = "close",
    symbol_column: str = "symbol",
) -> pd.DataFrame:
    """Add future volatility and drawdown labels over multiple horizons."""
    data = frame.copy()
    grouped = data.groupby(symbol_column, group_keys=False)
    for horizon in horizons:
        data[f"future_{horizon}d_volatility"] = grouped[return_column].transform(
            lambda returns: returns.rolling(horizon).std().shift(-horizon)
        )
        data[f"future_{horizon}d_drawdown"] = grouped[price_column].transform(
            lambda close: _future_drawdown(close, horizon)
        )
    return data


def add_sector_neutral_return_label(
    frame: pd.DataFrame,
    return_column: str,
    sector_column: str = "sector",
    date_column: str = "trade_date",
) -> pd.DataFrame:
    """Subtract same-date same-sector mean return from a future return label."""
    if sector_column not in frame.columns:
        raise ValueError(f"Missing sector column: {sector_column}")
    data = frame.copy()
    sector_mean = data.groupby([date_column, sector_column])[return_column].transform("mean")
    data[f"{return_column}_sector_neutral"] = data[return_column] - sector_mean
    return data


def _future_drawdown(close: pd.Series, horizon: int) -> pd.Series:
    future_min = close.shift(-1).rolling(horizon).min().shift(-(horizon - 1))
    return future_min / close - 1.0


def _future_benchmark_log_return(
    data: pd.DataFrame,
    benchmark_price_column: str,
    horizon: int,
    date_column: str,
) -> pd.Series:
    by_date = data[[date_column, benchmark_price_column]].drop_duplicates(date_column).sort_values(date_column)
    by_date[f"future_{horizon}d_benchmark_log_return"] = np.log(
        by_date[benchmark_price_column].shift(-horizon) / by_date[benchmark_price_column]
    )
    return data[[date_column]].merge(
        by_date[[date_column, f"future_{horizon}d_benchmark_log_return"]],
        on=date_column,
        how="left",
    )[f"future_{horizon}d_benchmark_log_return"]
