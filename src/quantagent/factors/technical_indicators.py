"""Classic technical indicators (Bollinger / RSI / MACD) as PIT-safe factors.

These factors are registered into ``default_registry`` under the
``technical_indicators`` category. Each indicator is computed per symbol
on the daily close panel and exposes one or more factor outputs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from quantagent.factors.registry import FactorMeta, default_registry

BASE_COLUMNS = ("close",)
EPS = 1e-12


def _base(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    return data


def _format(data: pd.DataFrame, name: str, values: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": data["trade_date"].to_numpy(),
            "symbol": data["symbol"].to_numpy(),
            "factor_name": name,
            "factor_value": values.to_numpy(dtype=float),
        }
    )


# ---------------------------------------------------------------------------
# Bollinger Bands (%b — position of price within the 2σ envelope)
# ---------------------------------------------------------------------------


def bollinger_percent_b(frame: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """%b = (close − lower) / (upper − lower), clipped to a finite range.

    Values near 1 mean price is at the upper band (overbought), near 0
    at the lower band (oversold), 0.5 at the midline.
    """
    data = _base(frame)
    grouped = data.groupby("symbol", sort=False)["close"]
    mean = grouped.transform(lambda s: s.rolling(window, min_periods=window).mean())
    std = grouped.transform(lambda s: s.rolling(window, min_periods=window).std(ddof=0))
    upper = mean + num_std * std
    lower = mean - num_std * std
    denom = (upper - lower).replace(0.0, np.nan)
    values = (data["close"] - lower) / denom
    return _format(data, "boll_percent_b_20", values)


def bollinger_bandwidth(frame: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bandwidth = (upper − lower) / mean — a volatility-regime signal."""
    data = _base(frame)
    grouped = data.groupby("symbol", sort=False)["close"]
    mean = grouped.transform(lambda s: s.rolling(window, min_periods=window).mean())
    std = grouped.transform(lambda s: s.rolling(window, min_periods=window).std(ddof=0))
    denom = mean.replace(0.0, np.nan)
    values = (2.0 * num_std * std) / denom
    return _format(data, "boll_bandwidth_20", values)


# ---------------------------------------------------------------------------
# Relative Strength Index (Wilder smoothing)
# ---------------------------------------------------------------------------


def _rsi_series(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder's smoothing ≈ EMA with alpha = 1/window
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When average loss is zero (pure up moves), RSI is 100 by convention.
    rsi = rsi.where(avg_loss != 0, 100.0)
    # When both gain and loss are zero (flat), leave NaN until warmed.
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def rsi_14(frame: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    data = _base(frame)
    values = data.groupby("symbol", sort=False)["close"].transform(lambda s: _rsi_series(s, window))
    return _format(data, "rsi_14", values)


# ---------------------------------------------------------------------------
# MACD (12, 26, 9) histogram
# ---------------------------------------------------------------------------


def _macd_hist_series(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd - sig


def macd_hist(frame: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Standard MACD histogram (price-scale)."""
    data = _base(frame)
    values = data.groupby("symbol", sort=False)["close"].transform(
        lambda s: _macd_hist_series(s, fast, slow, signal)
    )
    return _format(data, "macd_hist_12_26_9", values)


def macd_hist_normalized(frame: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD histogram divided by close — comparable across price levels."""
    data = _base(frame)
    hist = data.groupby("symbol", sort=False)["close"].transform(
        lambda s: _macd_hist_series(s, fast, slow, signal)
    )
    values = hist / data["close"].replace(0.0, np.nan)
    return _format(data, "macd_hist_norm_12_26_9", values)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register(
    name: str,
    func: Callable[[pd.DataFrame], pd.DataFrame],
    description: str,
    *,
    horizon: int = 5,
    direction: int = 1,
    lookback: int = 26,
) -> None:
    default_registry.add(
        FactorMeta(
            name=name,
            category="technical_indicators",
            horizon_days=horizon,
            required_columns=BASE_COLUMNS,
            direction=direction,
            description=description,
            source="Classic TA (Bollinger 1980 / Wilder RSI / Appel MACD)",
            group="technical_classic",
            lookback=lookback,
        ),
        func,
    )


_register(
    "boll_percent_b_20",
    bollinger_percent_b,
    "Bollinger %b (close position within 20d 2σ envelope)",
    horizon=5,
    direction=-1,
    lookback=20,
)
_register(
    "boll_bandwidth_20",
    bollinger_bandwidth,
    "Bollinger bandwidth (envelope width / mean)",
    horizon=10,
    direction=0,
    lookback=20,
)
_register(
    "rsi_14",
    rsi_14,
    "Wilder RSI(14)",
    horizon=5,
    direction=-1,
    lookback=14,
)
_register(
    "macd_hist_12_26_9",
    macd_hist,
    "MACD histogram (12/26/9 EMA)",
    horizon=5,
    direction=1,
    lookback=26,
)
_register(
    "macd_hist_norm_12_26_9",
    macd_hist_normalized,
    "MACD histogram divided by close",
    horizon=5,
    direction=1,
    lookback=26,
)


__all__ = [
    "bollinger_percent_b",
    "bollinger_bandwidth",
    "rsi_14",
    "macd_hist",
    "macd_hist_normalized",
]
