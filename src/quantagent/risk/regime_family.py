"""Three-family PIT market regime classifier: bull / neutral / bear."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeFamilyConfig:
    lookback_return_days: int = 60
    short_return_days: int = 20
    breadth_ma_days: int = 20
    bull_return_threshold: float = 0.08
    bear_return_threshold: float = -0.08
    bull_breadth_threshold: float = 0.55
    bear_breadth_threshold: float = 0.45


def compute_regime_family(
    market_panel: pd.DataFrame,
    *,
    config: RegimeFamilyConfig | None = None,
) -> pd.Series:
    """Classify each date into bull / neutral / bear using trailing data only."""
    cfg = config or RegimeFamilyConfig()
    if market_panel is None or market_panel.empty:
        return pd.Series(dtype="object", name="regime_family")
    df = market_panel[["trade_date", "symbol", "close"]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["trade_date", "symbol", "close"]).sort_values(["symbol", "trade_date"])
    df["_ret1d"] = df.groupby("symbol", sort=False)["close"].pct_change(fill_method=None)
    mkt_ret = df.groupby("trade_date")["_ret1d"].mean().fillna(0.0)
    index = (1.0 + mkt_ret).cumprod()
    ret_long = index / index.shift(int(cfg.lookback_return_days)) - 1.0
    ret_short = index / index.shift(int(cfg.short_return_days)) - 1.0
    ma = (
        df.groupby("symbol", sort=False)["close"]
        .transform(lambda s: s.rolling(int(cfg.breadth_ma_days), min_periods=5).mean())
    )
    df["_above_ma"] = df["close"] >= ma
    breadth = df.groupby("trade_date")["_above_ma"].mean()

    labels = []
    for date in index.index:
        r_long = float(ret_long.loc[date]) if pd.notna(ret_long.loc[date]) else 0.0
        r_short = float(ret_short.loc[date]) if pd.notna(ret_short.loc[date]) else 0.0
        b = float(breadth.loc[date]) if date in breadth.index and pd.notna(breadth.loc[date]) else 0.5
        if r_long >= cfg.bull_return_threshold and r_short >= 0.0 and b >= cfg.bull_breadth_threshold:
            labels.append("bull")
        elif r_long <= cfg.bear_return_threshold and r_short <= 0.0 and b <= cfg.bear_breadth_threshold:
            labels.append("bear")
        else:
            labels.append("neutral")
    return pd.Series(labels, index=pd.DatetimeIndex(index.index), name="regime_family")


__all__ = ["RegimeFamilyConfig", "compute_regime_family"]
