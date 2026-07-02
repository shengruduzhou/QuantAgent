"""Stage 8 — sector-rotation book construction.

Turns a sector signal + a per-stock feature panel into a per-stock
``target_weights`` pivot (index=trade_date, columns=symbol) ready for the
trusted ``run_strict_backtest_v8`` engine, so sector-rotation results are
cost/tradability-honest and directly comparable to the v8.9 stock baseline.

Construction knobs (the search grid):
  signal            sector score column (mom_60, rs_60, rmom_60, breadth_ma60, ...)
  top_n             number of sectors held {1,2,3,5,8}
  rebalance_days    rebalance cadence in trading days (5=weekly, 20, 21=monthly)
  sector_weighting  'equal' | 'volparity' | 'momentum'
  within_sector     'top_liquid' | 'top_momentum' | 'defensive'
  n_within          stocks per sector
  exposure          'full' | 'overlay' (overlay scales gross by market regime)

All selection uses only lookback information (signals are trailing; eligibility
flags are observable at the close). The book is forward-filled to every eval
trading day so positions are held between rebalances and turnover comes only
from rebalances.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTOR_COL = "sector_level_1"


def rebalance_dates(all_dates: list[pd.Timestamp], step: int) -> list[pd.Timestamp]:
    d = sorted(pd.DatetimeIndex(all_dates).unique())
    return list(d[::step])


def _eligible(feat_day: pd.DataFrame) -> pd.DataFrame:
    """Drop names not buyable at signal close: suspended / ST / limit-up sealed."""
    bad = pd.Series(False, index=feat_day.index)
    for c in ("is_suspended", "is_st", "is_limit_up"):
        if c in feat_day.columns:
            bad = bad | feat_day[c].fillna(False).astype(bool)
    return feat_day[~bad]


def select_sectors(sector_day: pd.DataFrame, signal: str, top_n: int,
                   *, reverse: bool = False) -> pd.DataFrame:
    """Top-N sectors by signal on one rebalance date; returns sector + raw signal.

    ``reverse=True`` selects the LOW-signal laggards (mean-reversion / 高低切),
    which the Stage 8 diagnostic showed is the side with positive sector-timing
    edge in A-shares.
    """
    s = sector_day.dropna(subset=[signal])
    s = s.sort_values(signal, ascending=reverse).head(top_n)
    return s[[SECTOR_COL, signal, "vol_60"]].copy()


def _sector_weights(sel: pd.DataFrame, signal: str, weighting: str) -> pd.Series:
    if weighting == "equal" or len(sel) == 1:
        w = pd.Series(1.0, index=sel[SECTOR_COL])
    elif weighting == "volparity":
        iv = 1.0 / (sel.set_index(SECTOR_COL)["vol_60"].abs() + 1e-9)
        w = iv
    elif weighting == "momentum":
        raw = sel.set_index(SECTOR_COL)[signal].clip(lower=0.0)
        w = raw if raw.sum() > 0 else pd.Series(1.0, index=sel[SECTOR_COL])
    else:
        raise ValueError(f"unknown sector weighting {weighting!r}")
    return w / w.sum()


def _pick_within(feat_day: pd.DataFrame, sector: str, mode: str, n: int) -> list[str]:
    g = _eligible(feat_day[feat_day[SECTOR_COL] == sector])
    if g.empty:
        return []
    if mode == "top_liquid":
        g = g.sort_values("amt20", ascending=False)
    elif mode == "top_momentum":
        g = g.sort_values("mom60", ascending=False)
    elif mode == "defensive":  # lowest realized vol among liquid enough
        liq = g.sort_values("amt20", ascending=False).head(max(n * 3, n))
        g = liq.sort_values("vol60", ascending=True)
    else:
        raise ValueError(f"unknown within_sector mode {mode!r}")
    return list(g["symbol"].head(n))


def build_rotation_book(
    sector_panel: pd.DataFrame,
    stock_feat: pd.DataFrame,
    *,
    signal: str,
    top_n: int,
    rebalance_days: int,
    sector_weighting: str = "equal",
    within_sector: str = "top_liquid",
    n_within: int = 5,
    exposure: str = "full",
    reverse: bool = False,
    gross_overlay: pd.Series | None = None,
    eval_dates: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Build the forward-filled per-stock target-weight pivot for one config.

    ``reverse`` picks laggard sectors. ``gross_overlay`` (a date->[0,1] Series)
    scales total exposure on each rebalance date (risk overlay); the remainder
    stays in cash.
    """
    if eval_dates is None:
        eval_dates = sorted(stock_feat["trade_date"].unique())
    eval_dates = sorted(pd.DatetimeIndex(eval_dates).unique())
    rebals = rebalance_dates(eval_dates, rebalance_days)

    sp_by_date = {d: g for d, g in sector_panel.groupby("trade_date")}
    sf_by_date = {d: g for d, g in stock_feat.groupby("trade_date")}

    weight_rows: dict[pd.Timestamp, dict[str, float]] = {}
    for d in rebals:
        sd = sp_by_date.get(d)
        fd = sf_by_date.get(d)
        if sd is None or fd is None:
            continue
        sel = select_sectors(sd, signal, top_n, reverse=reverse)
        if sel.empty:
            continue
        sw = _sector_weights(sel, signal, sector_weighting)
        gross = 1.0
        if exposure == "overlay" and gross_overlay is not None:
            reg = gross_overlay.reindex([d]).iloc[0] if d in gross_overlay.index else 1.0
            gross = float(np.clip(reg, 0.0, 1.0)) if np.isfinite(reg) else 1.0
        day_w: dict[str, float] = {}
        for sector, secw in sw.items():
            picks = _pick_within(fd, sector, within_sector, n_within)
            if not picks:
                continue
            per = secw * gross / len(picks)
            for sym in picks:
                day_w[sym] = day_w.get(sym, 0.0) + per
        if day_w:
            weight_rows[d] = day_w

    if not weight_rows:
        return pd.DataFrame()
    tw = pd.DataFrame.from_dict(weight_rows, orient="index").fillna(0.0).sort_index()
    # forward-fill to every eval trading day so positions are held between rebalances
    full = pd.DatetimeIndex([d for d in eval_dates if d >= tw.index.min()])
    tw = tw.reindex(full).ffill().fillna(0.0)
    tw.index.name = "trade_date"
    return tw
