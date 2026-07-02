"""Stage 8 — risk overlay for the sector-reversal book.

The laggard-sector reversal edge is real (+3-9% OOS excess vs equal-sector) but
fully-invested concentrated baskets draw down 23-34% because laggard sectors
still fall in a *market-wide* crash (2018, 2022H1, 2024-01). This overlay scales
gross exposure on each rebalance date to cut that systemic drawdown while
keeping the bull-market capture:

* market-trend gate  — cut exposure when the broad market (equal-sector bench)
  is in a downtrend; full risk-on when it trends up. This is the high-leverage
  fix: it sidesteps systemic crashes, and because the benchmark stays fully
  invested, going to cash in a bear *adds* excess.
* volatility target  — de-lever (long-only, never above 1.0) when realised
  basket vol runs hot, damping the lumpy episode risk.

All inputs are trailing / known at the rebalance close (no lookahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 244


def market_trend_gate(bench_daily: pd.Series, *, lookback: int = 60,
                      ma_window: int = 120, floor: float = 0.0,
                      mid: float = 0.5, soft: float = -0.05) -> pd.Series:
    """0..1 gross scalar from broad-market trend (equal-sector benchmark).

    risk-on (1.0) when trailing `lookback` return > 0 and price above its MA;
    half (`mid`) in a mild pullback; `floor` (cash) in a confirmed downtrend
    (return < `soft` and below MA). Smooth, hysteresis-free, fully causal.
    """
    cum = (1.0 + bench_daily.fillna(0.0)).cumprod()
    trail = cum / cum.shift(lookback) - 1.0
    ma = cum.rolling(ma_window, min_periods=ma_window // 2).mean()
    above = cum > ma
    g = pd.Series(mid, index=bench_daily.index)
    g[(trail > 0) & above] = 1.0
    g[(trail < soft) & (~above)] = floor
    return g.clip(0.0, 1.0)


def vol_target_scalar(book_daily: pd.Series, *, target_ann: float = 0.25,
                      window: int = 20, cap: float = 1.0) -> pd.Series:
    """Long-only de-lever scalar = min(cap, target / realised vol)."""
    rv = book_daily.rolling(window, min_periods=window // 2).std() * np.sqrt(ANN)
    s = (target_ann / (rv + 1e-9)).clip(upper=cap)
    return s.fillna(cap)


def combined_overlay(bench_daily: pd.Series, book_daily: pd.Series | None = None,
                     *, use_trend: bool = True, use_voltarget: bool = False,
                     target_ann: float = 0.25, **trend_kw) -> pd.Series:
    """Per-date gross scalar = trend_gate * vol_target (each optional)."""
    g = pd.Series(1.0, index=bench_daily.index)
    if use_trend:
        g = g * market_trend_gate(bench_daily, **trend_kw)
    if use_voltarget and book_daily is not None:
        g = g * vol_target_scalar(book_daily, target_ann=target_ann).reindex(g.index).fillna(1.0)
    return g.clip(0.0, 1.0)


def apply_overlay_to_nav(book_daily: pd.Series, gross: pd.Series) -> pd.Series:
    """Re-derive a NAV where each day's book return is scaled by yesterday's gross.

    Cheap way to evaluate an overlay at the sector-basket level without rebuilding
    the book: gross is applied with a 1-day lag (set at close t, governs t+1).
    """
    g = gross.reindex(book_daily.index).shift(1).fillna(1.0).clip(0.0, 1.0)
    net = book_daily.fillna(0.0) * g
    return (1.0 + net).cumprod()
