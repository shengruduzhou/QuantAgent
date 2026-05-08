from __future__ import annotations

import pandas as pd


def add_forward_return_labels(
    features: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 20),
    benchmark_return_column: str = "benchmark_ret_1d",
) -> pd.DataFrame:
    """Add future raw and benchmark-relative returns without look-ahead features."""
    frame = features.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    grouped = frame.groupby("symbol", group_keys=False)

    for horizon in horizons:
        future_close = grouped["close"].shift(-horizon)
        raw = future_close / frame["close"] - 1.0
        frame[f"future_{horizon}d_return"] = raw

        benchmark_future = _future_compound_return(frame, benchmark_return_column, horizon)
        frame[f"future_{horizon}d_excess_return"] = raw - benchmark_future

    frame["future_5d_volatility"] = grouped["ret_1d"].transform(
        lambda returns: returns.rolling(5).std().shift(-5)
    )
    frame["future_20d_drawdown"] = grouped["close"].transform(_future_20d_drawdown)
    return frame


def _future_compound_return(frame: pd.DataFrame, column: str, horizon: int) -> pd.Series:
    by_date = frame[["trade_date", column]].drop_duplicates("trade_date").sort_values("trade_date")
    future = (1.0 + by_date[column]).rolling(horizon).apply(lambda values: values.prod(), raw=True)
    by_date[f"future_{horizon}d_benchmark_return"] = future.shift(-horizon) - 1.0
    return frame[["trade_date"]].merge(
        by_date[["trade_date", f"future_{horizon}d_benchmark_return"]],
        on="trade_date",
        how="left",
    )[f"future_{horizon}d_benchmark_return"]


def _future_20d_drawdown(close: pd.Series) -> pd.Series:
    future_min = close.shift(-1).rolling(20).min().shift(-19)
    return future_min / close - 1.0
