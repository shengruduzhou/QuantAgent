"""Stage 8 — PIT-safe SW1 sector aggregate panel.

Turns the per-stock daily price panel + the SW1/SW2 ``sector_map`` into a
*sector-level* daily panel: one row per ``(trade_date, sector_level_1)`` with
equal-weight and liquidity(amount)-weighted sector returns, breadth, momentum,
volatility, drawdown and relative-strength series. This is the data foundation
for sector-rotation signal search.

Honesty / PIT caveats (documented, not hidden):

* The ``sector_map`` we have is a **current snapshot** (one ``available_at``
  date). Joining today's SW1 label onto historical prices therefore assumes a
  stock's SW1 membership is constant back through history. SW1 (申万一级) is
  stable enough that reclassification drift is small at the *cross-sector*
  (ranking) level, but two biases remain and are inherited, not removed:
    - survivorship: stocks delisted before the snapshot are absent, so absolute
      sector returns are biased mildly upward;
    - reclassification: a name moved between SW1 sectors recently is mislabeled
      in the past.
  These are the *same* universe biases the v8.9 stock baseline carries, so
  cross-comparison between the sector book and that baseline stays fair.
* All per-day aggregates use only information available at the close of that
  day. Amount weights use the *prior* day's amount (no same-day lookahead).
* Suspended names (``is_suspended``) are dropped from the membership each day.

Pure functions: panel in, frame out, no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTOR_COL = "sector_level_1"


def _prep_prices(panel: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    """Join SW1 labels, drop suspended/unclassified, compute per-symbol return."""
    sm = sector_map[["symbol", SECTOR_COL]].dropna(subset=[SECTOR_COL]).drop_duplicates("symbol")
    cols = ["symbol", "trade_date", "close", "amount"]
    opt = [c for c in ("is_suspended", "is_st", "is_limit_up", "is_limit_down") if c in panel.columns]
    px = panel[cols + opt].copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.merge(sm, on="symbol", how="inner")
    if "is_suspended" in px.columns:
        px = px[~px["is_suspended"].fillna(False).astype(bool)]
    px = px.sort_values(["symbol", "trade_date"])
    g = px.groupby("symbol", sort=False)
    px["ret"] = g["close"].pct_change(fill_method=None)
    # prior-day amount = today's liquidity weight known at signal time
    px["amt_lag"] = g["amount"].shift(1)
    # per-symbol trailing highs / MAs for breadth (lookback only)
    px["ma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    px["ma60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=30).mean())
    px["high60"] = g["close"].transform(lambda s: s.rolling(60, min_periods=30).max())
    px["above_ma20"] = (px["close"] >= px["ma20"]).astype(float)
    px["above_ma60"] = (px["close"] >= px["ma60"]).astype(float)
    px["at_high60"] = (px["close"] >= px["high60"] * 0.999).astype(float)
    px["up"] = (px["ret"] > 0).astype(float)
    return px


def build_sector_panel(panel: pd.DataFrame, sector_map: pd.DataFrame, *,
                       min_members: int = 5) -> pd.DataFrame:
    """Daily SW1 sector aggregate panel.

    Returns one row per ``(trade_date, sector_level_1)`` with:
      ret_eqw            equal-weight member return
      ret_amtw           prior-amount-weighted member return (cap/liquidity proxy)
      amount             total sector turnover (CNY)
      n_members          tradable members that day
      breadth_up         fraction of members with positive return
      breadth_ma20/ma60  fraction above 20/60d MA
      breadth_high60     fraction at/near 60d high
    Momentum / vol / drawdown / RS are added by :func:`add_sector_signals`.
    """
    px = _prep_prices(panel, sector_map)
    px = px.dropna(subset=["ret"])

    def _amtw(grp: pd.DataFrame) -> float:
        w = grp["amt_lag"]
        if w.notna().sum() == 0 or w.sum() <= 0:
            return float(grp["ret"].mean())
        w = w.fillna(0.0).clip(lower=0.0)
        if w.sum() <= 0:
            return float(grp["ret"].mean())
        return float(np.average(grp["ret"], weights=w))

    rows = px.groupby(["trade_date", SECTOR_COL], sort=True)
    agg = rows.agg(
        ret_eqw=("ret", "mean"),
        amount=("amount", "sum"),
        n_members=("ret", "size"),
        breadth_up=("up", "mean"),
        breadth_ma20=("above_ma20", "mean"),
        breadth_ma60=("above_ma60", "mean"),
        breadth_high60=("at_high60", "mean"),
    )
    amtw = rows.apply(_amtw, include_groups=False).rename("ret_amtw")
    out = agg.join(amtw).reset_index()
    out = out[out["n_members"] >= min_members].reset_index(drop=True)
    return out.sort_values(["trade_date", SECTOR_COL]).reset_index(drop=True)


def add_sector_signals(sector_panel: pd.DataFrame, *,
                       ret_col: str = "ret_eqw") -> pd.DataFrame:
    """Add momentum / vol / drawdown / relative-strength signals per sector.

    All windows are trailing (lookback-only). Relative strength is vs the
    cross-sector equal-weight mean return that day (an internal all-sector bench).
    """
    df = sector_panel.copy().sort_values([SECTOR_COL, "trade_date"])
    g = df.groupby(SECTOR_COL, sort=False)[ret_col]
    logret = np.log1p(df[ret_col].clip(lower=-0.99))
    df["_logret"] = logret
    gl = df.groupby(SECTOR_COL, sort=False)["_logret"]
    for w in (20, 60, 120):
        df[f"mom_{w}"] = np.expm1(gl.transform(lambda s, w=w: s.rolling(w, min_periods=w // 2).sum()))
    df["vol_20"] = g.transform(lambda s: s.rolling(20, min_periods=10).std())
    df["vol_60"] = g.transform(lambda s: s.rolling(60, min_periods=30).std())
    # risk-adjusted 60d momentum
    df["rmom_60"] = df["mom_60"] / (df["vol_60"] * np.sqrt(60) + 1e-9)
    # drawdown from trailing 120d cumulative high + recovery slope
    cum = np.expm1(gl.transform(lambda s: s.rolling(252, min_periods=20).sum()))
    df["nav"] = 1.0 + cum.fillna(0.0)
    roll_max = df.groupby(SECTOR_COL, sort=False)["nav"].transform(lambda s: s.cummax())
    df["drawdown"] = df["nav"] / roll_max - 1.0
    df["dd_recover_20"] = g.transform(  # avg return over last 20d while in drawdown = recovery thrust
        lambda s: s.rolling(20, min_periods=10).mean())
    # volume / turnover acceleration: 5d vs 20d amount
    ga = df.groupby(SECTOR_COL, sort=False)["amount"]
    df["amt_accel"] = (ga.transform(lambda s: s.rolling(5, min_periods=3).mean())
                       / (ga.transform(lambda s: s.rolling(20, min_periods=10).mean()) + 1e-9) - 1.0)
    # relative strength vs cross-sector mean (computed per date)
    mkt = df.groupby("trade_date")[ret_col].transform("mean")
    df["_rs_excess"] = df[ret_col] - mkt
    grs = df.groupby(SECTOR_COL, sort=False)["_rs_excess"]
    for w in (20, 60, 120):
        df[f"rs_{w}"] = grs.transform(lambda s, w=w: s.rolling(w, min_periods=w // 2).sum())
    df = df.drop(columns=["_logret", "_rs_excess"])
    return df.sort_values(["trade_date", SECTOR_COL]).reset_index(drop=True)
