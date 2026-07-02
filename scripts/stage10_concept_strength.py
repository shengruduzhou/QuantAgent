#!/usr/bin/env python3
"""Stage 10 step 1 — concept STRENGTH ranking + market state (live).

For every taxonomy-mapped 东财 concept board, combines:
  * board-index multi-horizon return (1/5/20/60d) + relative strength vs the
    cross-concept market, new-60/120d-high breakout
  * volume/turnover expansion (5d vs 20d 成交额), 换手率
  * 主力/超大单 net inflow (concept fund-flow rank)
  * breadth (上涨/下跌家数), 领涨股 limit-up as 涨停扩散 proxy
into a composite concept-strength score, classifies each concept's wave state
(主升 / 启动 / 高低切 / 退潮 / 弱势), and the overall market regime. This is the
live "current strongest 细分概念" screen — no look-ahead, current snapshot.

Board history is cached per board so reruns are instant. All amounts in CNY.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantagent.concept.taxonomy import EXCLUDE, board_to_leaves  # noqa: E402

RAW = Path("runtime/stage10_concept/raw")
HIST = RAW / "board_hist"
OUT = Path("runtime/stage10_concept")


def _ak():
    import akshare as ak
    return ak


def fetch_board_hist(boards: list[str], days: int = 200) -> dict[str, pd.DataFrame]:
    ak = _ak()
    HIST.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=days * 2)).strftime("%Y%m%d")
    out = {}
    miss = 0
    for i, b in enumerate(boards):
        f = HIST / f"{b.replace('/', '_')}.parquet"
        if f.exists():
            out[b] = pd.read_parquet(f)
            continue
        try:
            h = ak.stock_board_concept_hist_em(symbol=b, period="daily",
                                               start_date=start, end_date=end, adjust="")
            if h is not None and not h.empty:
                h.to_parquet(f)
                out[b] = h
            time.sleep(0.25)
        except Exception as e:
            miss += 1
            print(f"   miss {b}: {repr(e)[:60]}")
        if (i + 1) % 40 == 0:
            print(f"   fetched {i+1}/{len(boards)} boards")
    print(f"[hist] {len(out)} boards ({miss} missing)")
    return out


def board_strength(h: pd.DataFrame) -> dict:
    h = h.sort_values("日期")
    c = h["收盘"].astype(float).values
    amt = h["成交额"].astype(float).values
    if len(c) < 25:
        return {}
    def ret(n): return c[-1] / c[-n - 1] - 1.0 if len(c) > n else np.nan
    hi60 = np.nanmax(c[-60:]) if len(c) >= 60 else np.nanmax(c)
    hi120 = np.nanmax(c[-120:]) if len(c) >= 120 else np.nanmax(c)
    amt5 = np.nanmean(amt[-5:]); amt20 = np.nanmean(amt[-20:])
    vol20 = np.nanstd(np.diff(np.log(c[-21:]))) if len(c) >= 22 else np.nan
    return {
        "ret_1": ret(1), "ret_5": ret(5), "ret_20": ret(20), "ret_60": ret(60),
        "amt_today": float(amt[-1]), "amt_expand": float(amt5 / (amt20 + 1e-9) - 1.0),
        "turnover": float(h["换手率"].iloc[-1]) if "换手率" in h else np.nan,
        "new_high_60": float(c[-1] >= hi60 * 0.999),
        "new_high_120": float(c[-1] >= hi120 * 0.999),
        "vol_20": float(vol20),
        "from_high_60": float(c[-1] / hi60 - 1.0),
    }


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spot = pd.read_parquet(RAW / "em_concept_boards.parquet")
    b2l = board_to_leaves()
    mapped = sorted(set(b2l) & set(spot["板块名称"]))
    print(f"[concept] {len(mapped)} taxonomy-mapped boards (of {len(spot)} live)")

    hist = fetch_board_hist(mapped)
    ak = _ak()
    try:
        ff = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="概念资金流")
        ff = ff.set_index("名称")["今日主力净流入-净占比"].to_dict()
    except Exception as e:
        print("fund flow ERR", repr(e)[:80]); ff = {}

    spot_i = spot.set_index("板块名称")
    rows = []
    for b in mapped:
        if b not in hist:
            continue
        st = board_strength(hist[b])
        if not st:
            continue
        sp = spot_i.loc[b] if b in spot_i.index else None
        up = float(sp["上涨家数"]) if sp is not None else np.nan
        dn = float(sp["下跌家数"]) if sp is not None else np.nan
        breadth = up / (up + dn + 1e-9) if not np.isnan(up) else np.nan
        lead_lu = float(sp["领涨股票-涨跌幅"] >= 9.8) if sp is not None and "领涨股票-涨跌幅" in sp else 0.0
        leaves = b2l[b]
        rows.append({
            "board": b, "industry": leaves[0].industry, "track": leaves[0].track,
            "segment": leaves[0].segment, "position": leaves[0].position,
            **st, "breadth": breadth, "lead_limitup": lead_lu,
            "mainflow_pct": float(ff.get(b, 0.0)),
            "mktcap": float(sp["总市值"]) if sp is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    # composite strength (cross-concept z-scores; momentum-weighted)
    def z(col):
        s = df[col]; return (s - s.mean()) / (s.std() + 1e-9)
    df["strength"] = (1.0 * z("ret_5") + 1.5 * z("ret_20") + 1.0 * z("ret_60")
                      + 1.0 * z("amt_expand") + 1.0 * z("mainflow_pct")
                      + 0.5 * z("breadth") + 0.5 * df["new_high_60"]
                      + 0.5 * df["new_high_120"] + 0.3 * df["lead_limitup"])
    # relative strength vs cross-concept market
    df["rs_20"] = df["ret_20"] - df["ret_20"].median()
    df["state"] = df.apply(classify_state, axis=1)
    df = df.sort_values("strength", ascending=False).reset_index(drop=True)
    df.to_csv(OUT / "concept_strength.csv", index=False)

    # ---- market regime from aggregate ----
    n = len(df)
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

    print(f"\n{'='*70}\n市场状态: {regime}")
    print(f"  依据: {pct_strong:.0%} 概念20日涨>5% | {pct_newhigh:.0%} 创60日新高 | "
          f"中位成交额环比 {med_amt_expand:+.0%} | {n_ignite} 个概念处于启动")
    print(f"\n=== 当前最强细分概念 TOP 25 ===")
    cols = ["board", "industry", "track", "segment", "ret_5", "ret_20", "ret_60",
            "amt_expand", "mainflow_pct", "breadth", "new_high_60", "state", "strength"]
    show = df[cols].head(25).copy()
    for c in ["ret_5", "ret_20", "ret_60", "amt_expand"]:
        show[c] = (show[c] * 100).round(1)
    show["mainflow_pct"] = show["mainflow_pct"].round(2)
    show["breadth"] = (show["breadth"] * 100).round(0)
    show["strength"] = show["strength"].round(2)
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.unicode.east_asian_width", True):
        print(show.to_string(index=False))
    # industry roll-up
    print(f"\n=== 大产业强度 (mean strength, n concepts) ===")
    ind = df.groupby("industry").agg(strength=("strength", "mean"), n=("board", "size"),
                                     ret20=("ret_20", "mean")).sort_values("strength", ascending=False)
    ind["ret20"] = (ind["ret20"] * 100).round(1)
    ind["strength"] = ind["strength"].round(2)
    print(ind.to_string())
    print(f"\n[write] {OUT/'concept_strength.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
