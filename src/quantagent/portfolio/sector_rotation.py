"""板块轮动 + 高低切 + 做T overlay.

A-share capital rotates fast: overheated (高位) sectors top out while lagging
(低位) sectors that start stabilizing attract inflows (高低切). This module turns
that into signals layered ON TOP of the factor-dominant stock pool — it never
overrides the factor alpha, it manages risk/timing around it:

* ``sector_heat`` — per 申万一级 sector: momentum + proximity to 60d high.
  高位 = near high + strong 60d momentum; 低位 = far below high + weak momentum.
* ``rotation_score`` — favors 低位但企稳改善 (lagging yet turning up = inflow
  target), penalizes 高位且动能衰减 (overheated + decelerating = outflow risk).
* ``do_t_flag`` — per stock: high intraday range + liquidity + sits in a 高位
  sector ⇒ good 做T(T+0) candidate to trim cost / hedge risk while holding a
  good factor name through rotation. Trend is up, so we hold and manage, not sell.

Pure functions (panel in / frame out); no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 1e-12 else s * 0.0


def compute_sector_heat(
    panel: pd.DataFrame,
    sector_map: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    high_window: int = 60,
) -> pd.DataFrame:
    """Per-申万一级 sector heat / 高低 metrics as of ``as_of`` (lookahead-safe)."""
    px = panel[panel["trade_date"] <= as_of].copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    sm = sector_map[["symbol", "sector_level_1"]].drop_duplicates("symbol")
    px = px.merge(sm, on="symbol", how="inner")
    # per-symbol returns
    px = px.sort_values(["symbol", "trade_date"])
    g = px.groupby("symbol")["close"]
    last = px.groupby("symbol").tail(1).set_index("symbol")
    ret = {w: g.apply(lambda s, w=w: s.iloc[-1] / s.iloc[-w - 1] - 1.0 if len(s) > w else np.nan) for w in (5, 20, 60)}
    high60 = g.apply(lambda s: s.tail(high_window).max())
    sym = pd.DataFrame({
        "sector_level_1": last["sector_level_1"],
        "ret_5d": ret[5], "ret_20d": ret[20], "ret_60d": ret[60],
        "pct_from_high": last["close"] / high60 - 1.0,
    }).dropna(subset=["sector_level_1"])
    sec = sym.groupby("sector_level_1").agg(
        ret_5d=("ret_5d", "median"), ret_20d=("ret_20d", "median"),
        ret_60d=("ret_60d", "median"), pct_from_high=("pct_from_high", "median"),
        n=("ret_5d", "size")).reset_index()
    sec = sec[sec["n"] >= 5].copy()
    # 高位: 近高点 + 60d强动能 ; 低位: 远离高点 + 弱动能
    sec["heat"] = _zscore(sec["ret_60d"]) + _zscore(sec["pct_from_high"])  # high = overheated
    sec["accel"] = sec["ret_5d"] - sec["ret_20d"] / 4.0                    # >0 = 近端加速
    # 轮动: 低位(heat低)但企稳改善(accel>0, ret_5d>0) = 流入目标; 高位且衰减 = 流出风险
    sec["rotation_score"] = (-0.6 * _zscore(sec["heat"]) + 0.4 * _zscore(sec["accel"])).round(3)
    sec["regime_tag"] = np.where(sec["heat"] > sec["heat"].quantile(0.66), "高位",
                          np.where(sec["heat"] < sec["heat"].quantile(0.33), "低位", "中位"))
    sec["rotation_tag"] = np.where((sec["regime_tag"] == "低位") & (sec["accel"] > 0), "低位企稳_流入",
                            np.where((sec["regime_tag"] == "高位") & (sec["accel"] < 0), "高位衰减_流出风险", "观望"))
    return sec.sort_values("rotation_score", ascending=False).reset_index(drop=True)


def attach_rotation_and_dot(
    pool: pd.DataFrame,
    sector_heat: pd.DataFrame,
    *,
    intraday_range_col: str = "intraday_range_pos",
) -> pd.DataFrame:
    """Layer rotation_score + 做T flag onto a stock pool (factor pool stays primary).

    - ``rotation_score`` joined by sector → a small tilt / awareness signal.
    - ``do_t_flag`` = the name sits in a 高位 sector (rotation risk) ⇒ hold but
      manage with 做T; the factor rank is unchanged.
    """
    out = pool.copy()
    heat = sector_heat.set_index("sector_level_1")
    out["sector_rotation_score"] = out["sector_level_1"].map(heat["rotation_score"]).fillna(0.0)
    out["sector_regime_tag"] = out["sector_level_1"].map(heat["regime_tag"]).fillna("中位")
    out["sector_rotation_tag"] = out["sector_level_1"].map(heat["rotation_tag"]).fillna("观望")
    # 做T: 高位板块的持仓 → 用T+0管理风险(规避高低切回撤), 趋势向上故不清仓
    in_hot = out["sector_regime_tag"].eq("高位")
    dot = pd.to_numeric(out.get("do_t_suitability_score", 0.5), errors="coerce").fillna(0.5)
    out["do_t_action"] = np.where(in_hot & (dot >= 0.45), "做T管理_高位",
                          np.where(dot >= 0.6, "做T可选", "持有"))
    return out


def attach_intra_sector_high_low(
    pool: pd.DataFrame,
    panel: pd.DataFrame,
    sector_map: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    high_window: int = 60,
) -> pd.DataFrame:
    """板块内部高低切: within each 申万一级, rank每只股票的高低位 (vs 60d high) and tilt.

    A-share intra-sector rotation: 高位个股(已大涨/贴近自身高点) 资金少拉, 反而拉起
    同板块的低位个股. So WITHIN a sector we mildly PREFER the lagging (低位) names
    over the already-extended (高位) ones — the factor rank stays primary; this is
    a small tilt + a 做T flag for 高位 names. Lookahead-safe (uses past closes only).
    """
    out = pool.copy()
    px = panel[panel["trade_date"] <= as_of].sort_values(["symbol", "trade_date"])
    g = px.groupby("symbol")["close"]
    last = px.groupby("symbol").tail(1).set_index("symbol")["close"]
    high = g.apply(lambda s: s.tail(high_window).max())
    ret20 = g.apply(lambda s: s.iloc[-1] / s.iloc[-21] - 1.0 if len(s) > 21 else np.nan)
    pos = (last / high - 1.0)  # 0=贴高点(高位), 越负=越低位
    sm = sector_map[["symbol", "sector_level_1"]].drop_duplicates("symbol")
    if "sector_level_1" not in out.columns:
        out = out.merge(sm, on="symbol", how="left")
    out["pct_from_high"] = out["symbol"].map(pos)
    out["ret_20d_stk"] = out["symbol"].map(ret20)
    # within-sector percentile of 高低位 (1 = 最高位)
    out["intra_sector_highpos"] = out.groupby("sector_level_1")["pct_from_high"].rank(pct=True)
    out["intra_sector_tag"] = np.where(out["intra_sector_highpos"] >= 0.70, "板块内高位_少拉/做T",
                               np.where(out["intra_sector_highpos"] <= 0.40, "板块内低位_偏好补涨", "板块内中位"))
    # small tilt: prefer 低位 within sector (负向 highpos), bounded so factor stays primary
    out["intra_sector_tilt"] = (0.5 - out["intra_sector_highpos"].fillna(0.5)).round(3) * 0.10
    return out
