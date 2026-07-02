"""Stage 10 — pure concept-strength & stock-hardness scoring functions.

Shared by the daily scanner (10.1), the PIT snapshot store (10.2) and the
forward paper-trader (10.4) so there is ONE implementation, no drift. No I/O,
no network — frames in, frames out. All features are trailing / point-in-time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------- concept strength ----------------------------
def board_strength(h: pd.DataFrame) -> dict:
    """Per-board strength features from its index OHLCV history (东财 hist)."""
    h = h.sort_values("日期")
    c = h["收盘"].astype(float).values
    amt = h["成交额"].astype(float).values
    if len(c) < 25:
        return {}
    def ret(n): return c[-1] / c[-n - 1] - 1.0 if len(c) > n else np.nan
    hi60 = np.nanmax(c[-60:]) if len(c) >= 60 else np.nanmax(c)
    hi120 = np.nanmax(c[-120:]) if len(c) >= 120 else np.nanmax(c)
    amt5, amt20 = np.nanmean(amt[-5:]), np.nanmean(amt[-20:])
    vol20 = np.nanstd(np.diff(np.log(c[-21:]))) if len(c) >= 22 else np.nan
    return {
        "ret_1": ret(1), "ret_5": ret(5), "ret_20": ret(20), "ret_60": ret(60),
        "amt_today": float(amt[-1]), "amt_expand": float(amt5 / (amt20 + 1e-9) - 1.0),
        "new_high_60": float(c[-1] >= hi60 * 0.999), "new_high_120": float(c[-1] >= hi120 * 0.999),
        "vol_20": float(vol20), "from_high_60": float(c[-1] / hi60 - 1.0),
    }


def composite_strength(df: pd.DataFrame) -> pd.Series:
    def z(col):
        s = df[col]; return (s - s.mean()) / (s.std() + 1e-9)
    return (1.0 * z("ret_5") + 1.5 * z("ret_20") + 1.0 * z("ret_60")
            + 1.0 * z("amt_expand") + 1.0 * z("mainflow_pct")
            + 0.5 * z("breadth") + 0.5 * df["new_high_60"]
            + 0.5 * df["new_high_120"] + 0.3 * df["lead_limitup"])


def classify_state(r: pd.Series) -> str:
    strong = r["ret_20"] > 0.05 and r["ret_60"] > 0
    near_high = r["from_high_60"] > -0.05
    expanding = r["amt_expand"] > 0.10
    inflow = r["mainflow_pct"] > 0
    recent_up = r["ret_5"] > 0.02
    fading = r["ret_5"] < -0.02 and r["amt_expand"] < 0
    if r["new_high_60"] and expanding and recent_up:
        return "启动"
    if strong and near_high and (expanding or inflow):
        return "主升"
    if r["ret_60"] < 0 and recent_up and r["amt_expand"] > 0:
        return "高低切"
    if (r["ret_20"] > 0.10) and fading:
        return "退潮"
    if r["ret_20"] < -0.03 and not recent_up:
        return "弱势"
    return "震荡"


def market_regime(df: pd.DataFrame) -> tuple[str, dict]:
    pct_strong = float((df["ret_20"] > 0.05).mean())
    pct_newhigh = float((df["new_high_60"] > 0).mean())
    med_amt_expand = float(df["amt_expand"].median())
    n_ignite = int((df["state"] == "启动").sum())
    if pct_strong > 0.45 and pct_newhigh > 0.20:
        regime = "强趋势牛市(追主线)"
    elif pct_strong < 0.15 and df["ret_20"].median() < 0:
        regime = "退潮/弱市(降仓防御)"
    elif n_ignite >= 5 and med_amt_expand > 0:
        regime = "新主线启动(看突破扩散)"
    else:
        regime = "震荡轮动(高低切)"
    return regime, {"pct_strong_20d": round(pct_strong, 3), "pct_new_high_60": round(pct_newhigh, 3),
                    "median_amt_expand": round(med_amt_expand, 3), "n_igniting": n_ignite}


# ------------------------------ stock hardness ------------------------------
def score_hardness(df: pd.DataFrame) -> pd.DataFrame:
    """Add 概念硬度 components + total. df has: mktcap, ret60, liangbi, pe, pb,
    mainflow_pct, rev_yoy, profit_yoy, gross_margin, yj_forecast, board, order_label.
    order_label drives the 订单 score (Stage 10.3 hard labels)."""
    d = df.copy()
    cl = np.clip
    # 概念纯度 from VERIFIED 主营收入占比 (revenue_exposure_pct, 0..1); NO market-cap proxy.
    # Unknown -> neutral 10 + purity_status flag (never fabricated).
    if "revenue_exposure_pct" in d.columns and d["revenue_exposure_pct"].notna().any():
        rev = pd.to_numeric(d["revenue_exposure_pct"], errors="coerce")
        d["score_purity"] = np.where(rev.notna(), cl(rev / 0.40 * 20.0, 0, 20), 10.0)
        d["purity_status"] = np.where(rev.notna(), "verified", "unknown")
    else:
        d["score_purity"] = 10.0
        d["purity_status"] = "unknown"
    # 订单 score from hard labels (Stage 10.3); fallback neutral when unverified
    order_pts = {"confirmed_order": 18, "confirmed_customer": 15, "revenue_exposure": 13,
                 "earnings_verified": 11, "rumor_only": 4, "fake_concept": 0}
    d["score_order"] = d.get("order_label", pd.Series("unverified", index=d.index)).map(
        lambda x: order_pts.get(x, 8.0))
    perf = (cl(d["rev_yoy"].fillna(0) / 5.0, -4, 8) + cl(d["profit_yoy"].fillna(0) / 10.0, -6, 10)
            + cl((d["gross_margin"].fillna(20) - 20) / 5.0, -2, 4))
    perf = perf + d["yj_forecast"].fillna("").apply(
        lambda t: 3.0 if any(k in str(t) for k in ("预增", "扭亏", "续盈")) else
        (-3.0 if any(k in str(t) for k in ("预减", "首亏", "续亏")) else 0.0))
    d["score_perf"] = cl(perf + 8.0, 0, 20)
    cap_liq = (cl(d["mainflow_pct"].fillna(0) * 2, -6, 8) + cl(d["ret60"].fillna(0) / 8, -4, 8)
               + cl((d["liangbi"].fillna(1) - 1) * 2, -2, 4))
    d["score_capital"] = cl(cap_liq + 8.0, 0, 20)
    risk = (cl((d["pe"].fillna(40) - 60) / 20, 0, 6) + cl((d["pb"].fillna(5) - 8) / 3, 0, 4)
            + (d["profit_yoy"].fillna(0) < -20).astype(float) * 4
            + (d["score_purity"] < 10).astype(float) * 3
            + (d.get("order_label", "") == "fake_concept").astype(float) * 5)
    d["score_risk"] = -cl(risk, 0, 20)
    d["hardness"] = d[["score_purity", "score_order", "score_perf", "score_capital", "score_risk"]].sum(axis=1)
    d["短线波段分"] = d["score_capital"] + cl((d["liangbi"].fillna(1) - 1) * 3, 0, 6)
    d["中线趋势分"] = d["score_perf"] * 0.6 + cl(d["ret60"].fillna(0) / 5, -4, 8)
    d["业绩兑现分"] = d["score_perf"]
    return d


def classify_role(df: pd.DataFrame) -> pd.Series:
    out = []
    for i, r in df.iterrows():
        peers = df[df.board == r["board"]]
        capr = peers["mktcap"].rank(pct=True).loc[i]
        med_ret = peers["ret60"].median()
        if r.get("order_label") == "fake_concept" or (
                r["score_perf"] < 6 and r["score_capital"] < 6 and r["score_purity"] < 11):
            out.append("伪概念")
        elif r["score_perf"] >= 11 and r["score_capital"] >= 11 and capr >= 0.6:
            out.append("核心龙头")
        elif capr >= 0.75:
            out.append("中军")
        elif pd.notna(r["ret60"]) and r["ret60"] < med_ret and r["score_perf"] >= 8:
            out.append("补涨")
        else:
            out.append("弹性")
    return pd.Series(out, index=df.index)


def buy_reject_reason(r: pd.Series) -> str:
    if r["role"] == "伪概念":
        return "伪概念:无业绩+无资金+纯度低"
    if pd.notna(r.get("profit_yoy")) and r["profit_yoy"] < -20:
        return "业绩不兑现(净利大幅负增)"
    if r["score_risk"] < -6:
        return "估值过高(PE/PB极端)"
    if r.get("order_label") == "rumor_only":
        return "仅传闻未公告核实"
    return ""
