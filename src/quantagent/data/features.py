from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_PRICE_COLUMNS = {
    "trade_date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def validate_price_frame(prices: pd.DataFrame) -> None:
    missing = REQUIRED_PRICE_COLUMNS.difference(prices.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")


def add_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build point-in-time daily technical features from OHLCV data."""
    validate_price_frame(prices)
    frame = prices.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame = frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    grouped = frame.groupby("symbol", group_keys=False)

    frame["ret_1d"] = grouped["close"].pct_change()
    frame["ret_5d"] = grouped["close"].pct_change(5)
    frame["ret_20d"] = grouped["close"].pct_change(20)
    frame["volatility_20d"] = grouped["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)

    amount = frame["amount"] if "amount" in frame.columns else frame["close"] * frame["volume"]
    frame["amount"] = amount
    frame["volume_zscore_20d"] = _rolling_zscore(grouped["volume"], 20)
    frame["amount_zscore_20d"] = _rolling_zscore(frame.groupby("symbol")["amount"], 20)

    ma_5 = grouped["close"].rolling(5).mean().reset_index(level=0, drop=True)
    ma_20 = grouped["close"].rolling(20).mean().reset_index(level=0, drop=True)
    std_20 = grouped["close"].rolling(20).std().reset_index(level=0, drop=True)
    frame["ma_gap_5d"] = frame["close"] / ma_5 - 1.0
    frame["ma_gap_20d"] = frame["close"] / ma_20 - 1.0
    frame["bollinger_zscore_20d"] = (frame["close"] - ma_20) / (std_20 + 1e-12)
    frame["rsi_14d"] = grouped["close"].transform(_rsi_14)
    return frame.replace([np.inf, -np.inf], np.nan)


def add_benchmark_features(
    features: pd.DataFrame,
    benchmark: pd.DataFrame,
    benchmark_symbol: str,
) -> pd.DataFrame:
    """Join daily benchmark returns by date for excess-return labels and features."""
    benchmark_frame = benchmark.copy()
    benchmark_frame["trade_date"] = pd.to_datetime(benchmark_frame["trade_date"])
    benchmark_frame = benchmark_frame.sort_values("trade_date")
    benchmark_frame["benchmark_ret_1d"] = benchmark_frame["close"].pct_change()
    benchmark_frame = benchmark_frame[["trade_date", "benchmark_ret_1d"]]

    frame = features.copy()
    frame["benchmark_symbol"] = benchmark_symbol
    return frame.merge(benchmark_frame, on="trade_date", how="left")


def _rolling_zscore(series_group: pd.core.groupby.SeriesGroupBy, window: int) -> pd.Series:
    mean = series_group.rolling(window).mean().reset_index(level=0, drop=True)
    std = series_group.rolling(window).std().reset_index(level=0, drop=True)
    values = series_group.obj
    return (values - mean) / (std + 1e-12)


def _rsi_14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)
