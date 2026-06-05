"""Index-hedge overlay for a long-only A-share book.

A-share single-stock shorting is infeasible, so the execution engine
(`ashare_execution_simulator`) is long-only — it silently drops negative
target weights. The institutional way to neutralise market beta is instead
to short a **stock-index future** (CSI300 / CSI500 / CSI1000, i.e. IF / IC /
IM) against the long book. Futures are liquid, shortable and cheap.

This module applies that hedge as an overlay on the realised daily returns
of the long book: it does not need the simulator to short anything. Given
the long book's daily returns and a hedge-index daily return series,

    hedged_ret_t = long_ret_t − hedge_ratio · index_ret_t − daily_cost

``hedge_ratio`` is the fraction of book notional sold in the index
(1.0 ≈ fully market-neutral when the book's beta to the index is ≈ 1).
The small ``annual_cost_bps`` captures roll / basis / commission drag.

The point of the overlay (validated on the v8 bear OOS): it converts the
long decile book — which eats the full market drawdown — into a
market-neutral book whose P&L is the *cross-sectional* alpha, with
dramatically lower drawdown in falling / choppy regimes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight_market_return(panel: pd.DataFrame) -> pd.Series:
    """Daily equal-weight all-A return — a proxy hedge index from the panel.

    Mirrors the index the regime detector and the equal-weight benchmark use,
    so the hedge, the benchmark and the regime signal are mutually consistent.
    """
    p = panel[["symbol", "trade_date", "close"]].copy()
    p["trade_date"] = pd.to_datetime(p["trade_date"], errors="coerce")
    piv = p.pivot_table(index="trade_date", columns="symbol", values="close")
    return piv.pct_change(fill_method=None).mean(axis=1)


def apply_index_hedge(
    long_nav: pd.Series,
    index_returns: pd.Series,
    *,
    hedge_ratio: float = 1.0,
    annual_cost_bps: float = 50.0,
) -> pd.Series:
    """Return the hedged NAV series for a long book.

    ``long_nav`` is the realised NAV of the long-only book (from the strict
    simulator). ``index_returns`` is the daily return of the hedge index,
    aligned by date. ``hedge_ratio`` scales the short index leg; 0 disables
    the hedge (returns ``long_nav`` unchanged).
    """
    nav = long_nav.sort_index().dropna().astype(float)
    if hedge_ratio == 0.0 or len(nav) < 2:
        return nav
    long_ret = nav.pct_change()
    idx = index_returns.reindex(nav.index).fillna(0.0).astype(float)
    daily_cost = annual_cost_bps / 1e4 / 252.0
    hedged_ret = long_ret - hedge_ratio * idx - daily_cost
    hedged_ret = hedged_ret.fillna(0.0)
    hedged_nav = (1.0 + hedged_ret).cumprod() * float(nav.iloc[0])
    hedged_nav.iloc[0] = float(nav.iloc[0])
    return hedged_nav


def apply_dynamic_index_hedge(
    long_nav: pd.Series,
    index_returns: pd.Series,
    hedge_ratio_by_date: pd.Series,
    *,
    annual_cost_bps: float = 50.0,
) -> pd.Series:
    """Return hedged NAV with a per-date hedge ratio.

    This is the report/backtest overlay for regime-aware hedging. It does
    not emit stock orders or futures orders; it only marks a hypothetical
    short-index leg against the realised long-book NAV so drawdown control
    can be evaluated before any execution path is discussed.
    """
    nav = long_nav.sort_index().dropna().astype(float)
    if len(nav) < 2:
        return nav
    ratio = hedge_ratio_by_date.copy()
    ratio.index = pd.to_datetime(ratio.index, errors="coerce")
    ratio = ratio.reindex(nav.index).ffill().fillna(0.0).clip(lower=0.0)
    long_ret = nav.pct_change()
    idx = index_returns.reindex(nav.index).fillna(0.0).astype(float)
    daily_cost = (annual_cost_bps / 1e4 / 252.0) * ratio
    hedged_ret = long_ret - ratio * idx - daily_cost
    hedged_ret = hedged_ret.fillna(0.0)
    hedged_nav = (1.0 + hedged_ret).cumprod() * float(nav.iloc[0])
    hedged_nav.iloc[0] = float(nav.iloc[0])
    return hedged_nav


def nav_metrics(nav: pd.Series, *, periods: int = 252) -> dict[str, float]:
    """Annualised return / Sharpe / max-DD / vol for a NAV series."""
    nav = nav.sort_index().dropna().astype(float)
    if len(nav) < 2:
        return {"ann": 0.0, "sharpe": 0.0, "max_dd": 0.0, "vol": 0.0, "total_return": 0.0}
    ret = nav.pct_change().dropna()
    n = len(ret)
    total = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    ann = float((1.0 + total) ** (periods / n) - 1.0)
    std = float(ret.std(ddof=1))
    sharpe = float(ret.mean() / std * (periods ** 0.5)) if std > 1e-12 else 0.0
    vol = float(std * (periods ** 0.5))
    max_dd = float(abs((nav / nav.cummax() - 1.0).min()))
    return {"ann": ann, "sharpe": sharpe, "max_dd": max_dd, "vol": vol, "total_return": total}


__all__ = ["equal_weight_market_return", "apply_index_hedge", "apply_dynamic_index_hedge", "nav_metrics"]
