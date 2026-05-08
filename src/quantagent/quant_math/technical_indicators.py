from __future__ import annotations

import numpy as np
import pandas as pd


def add_advanced_technical_indicators(
    prices: pd.DataFrame,
    date_column: str = "trade_date",
    symbol_column: str = "symbol",
) -> pd.DataFrame:
    """Add Phase-1 no-training technical indicators for daily strategies."""
    required = {date_column, symbol_column, "open", "high", "low", "close", "volume"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")

    frame = prices.copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    frame = frame.sort_values([symbol_column, date_column]).reset_index(drop=True)
    grouped = frame.groupby(symbol_column, group_keys=False)

    amount = frame["amount"] if "amount" in frame.columns else frame["close"] * frame["volume"]
    frame["amount"] = amount
    frame["ret_1d"] = grouped["close"].pct_change()
    frame["rv_5d"] = grouped["ret_1d"].transform(lambda x: x.rolling(5).std())
    frame["rv_20d"] = grouped["ret_1d"].transform(lambda x: x.rolling(20).std())
    frame["rv_ratio_5_20"] = frame["rv_5d"] / (frame["rv_20d"] + 1e-12)

    frame["rsi_14d"] = grouped["close"].transform(_rsi)
    frame["macd_line"] = grouped["close"].transform(lambda x: _ema(x, 12) - _ema(x, 26))
    frame["macd_signal"] = frame.groupby(symbol_column)["macd_line"].transform(lambda x: _ema(x, 9))
    frame["macd_hist"] = frame["macd_line"] - frame["macd_signal"]
    frame["macd_hist_delta"] = frame.groupby(symbol_column)["macd_hist"].diff()

    ma_20 = grouped["close"].transform(lambda x: x.rolling(20).mean())
    std_20 = grouped["close"].transform(lambda x: x.rolling(20).std())
    frame["ma_20"] = ma_20
    frame["bb_upper_20_2"] = ma_20 + 2.0 * std_20
    frame["bb_lower_20_2"] = ma_20 - 2.0 * std_20
    frame["bollinger_zscore_20d"] = (frame["close"] - ma_20) / (std_20 + 1e-12)

    frame["atr_14d"] = _atr(frame, symbol_column)
    adx_frame = _adx(frame, symbol_column)
    frame["adx_14d"] = adx_frame["adx_14d"]
    frame["plus_di_14d"] = adx_frame["plus_di_14d"]
    frame["minus_di_14d"] = adx_frame["minus_di_14d"]

    frame["donchian_high_20d"] = grouped["high"].transform(lambda x: x.rolling(20).max().shift(1))
    frame["donchian_low_20d"] = grouped["low"].transform(lambda x: x.rolling(20).min().shift(1))
    frame["vwap_20d"] = _rolling_vwap(frame, symbol_column, 20)
    frame["volume_zscore_20d"] = grouped["volume"].transform(_zscore_20)
    frame["amount_zscore_20d"] = frame.groupby(symbol_column)["amount"].transform(_zscore_20)
    frame["amount_mean_20d"] = frame.groupby(symbol_column)["amount"].transform(lambda x: x.rolling(20).mean())
    frame["amihud_20d"] = frame.groupby(symbol_column).apply(_amihud).reset_index(level=0, drop=True)
    return frame.replace([np.inf, -np.inf], np.nan)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(frame: pd.DataFrame, symbol_column: str, window: int = 14) -> pd.Series:
    previous_close = frame.groupby(symbol_column)["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.groupby(frame[symbol_column]).transform(lambda x: x.rolling(window).mean())


def _adx(frame: pd.DataFrame, symbol_column: str, window: int = 14) -> pd.DataFrame:
    high_diff = frame.groupby(symbol_column)["high"].diff()
    low_diff = -frame.groupby(symbol_column)["low"].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)
    atr = _atr(frame, symbol_column, window)
    plus_di = 100.0 * plus_dm.groupby(frame[symbol_column]).transform(lambda x: x.rolling(window).sum()) / (
        atr * window + 1e-12
    )
    minus_di = 100.0 * minus_dm.groupby(frame[symbol_column]).transform(lambda x: x.rolling(window).sum()) / (
        atr * window + 1e-12
    )
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = dx.groupby(frame[symbol_column]).transform(lambda x: x.rolling(window).mean())
    return pd.DataFrame({"adx_14d": adx, "plus_di_14d": plus_di, "minus_di_14d": minus_di})


def _rolling_vwap(frame: pd.DataFrame, symbol_column: str, window: int) -> pd.Series:
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    value = typical * frame["volume"]
    numerator = value.groupby(frame[symbol_column]).transform(lambda x: x.rolling(window).sum())
    denominator = frame.groupby(symbol_column)["volume"].transform(lambda x: x.rolling(window).sum())
    return numerator / (denominator + 1e-12)


def _zscore_20(series: pd.Series) -> pd.Series:
    return (series - series.rolling(20).mean()) / (series.rolling(20).std() + 1e-12)


def _amihud(group: pd.DataFrame) -> pd.Series:
    illiquidity = group["ret_1d"].abs() / (group["amount"] + 1e-12)
    return illiquidity.rolling(20).mean()
