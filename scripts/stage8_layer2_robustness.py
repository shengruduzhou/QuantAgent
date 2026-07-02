#!/usr/bin/env python3
"""Stage 8 step 8 — DECISIVE size-matched, multi-phase layer-2 vs plain v8.9.

The layer-2 grid was confounded: hyper-concentrated books (2 sectors x 3 stocks
= ~6 names) showed huge "excess" that flipped sign between rb20 and rb21 — pure
concentration + timing luck, not sector alpha, and both laggard AND leader
directions "won". This test removes both confounds:

  * SIZE-MATCHED   : layer-2 holds the same #names N as the plain v8.9 control
                     (ns x ps = N), equal-stock weight (no score concentration).
  * MULTI-PHASE    : each config is run at 5 rebalance phase offsets (period 20,
                     start day 0/4/8/12/16) and averaged, so timing luck is
                     diversified away. We report mean +/- std across phases.

Verdict rule (user's): if size-matched layer-2 does not beat plain v8.9 on
after-cost CAGR / excess with stability across phases, the sector layer adds no
enhancement value. All strict A-share engine, same OOS dates / universe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

SCORE = "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/ensemble_composite.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
SECTOR_PANEL = "runtime/stage8_sector_rotation/sector_panel.parquet"
OUT_DIR = Path("runtime/stage8_sector_rotation")
ANN = 244
SCORE_COL = "composite_score"
PHASES = [0, 4, 8, 12, 16]
PERIOD = 20


def _ann(d):
    n = len(d); return float((1 + d).prod() ** (ANN / n) - 1) if n else float("nan")


def build_tw(stock_day, sector_day, *, rebal_dates, eval_dates, size,
             sector_signal=None, n_sectors=0, per_sector=0, reverse=True):
    rows = {}
    for d in rebal_dates:
        sd = stock_day.get(d)
        if sd is None or sd.empty:
            continue
        sd = sd[~sd["bad"]].dropna(subset=["score"])
        if sector_signal is not None:
            secf = sector_day.get(d)
            if secf is None or secf.empty:
                continue
            sf = secf.dropna(subset=[sector_signal]).sort_values(sector_signal, ascending=reverse)
            sd = sd[sd["sector"].isin(list(sf["sector_level_1"].head(n_sectors)))]
            sd = sd.sort_values("score", ascending=False).groupby("sector", group_keys=False).head(per_sector)
        sd = sd.sort_values("score", ascending=False).head(size)
        if sd.empty:
            continue
        w = 1.0 / len(sd)
        rows[d] = {s: w for s in sd["symbol"]}
    if not rows:
        return pd.DataFrame()
    tw = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()
    full = pd.DatetimeIndex([d for d in eval_dates if d >= tw.index.min()])
    return tw.reindex(full).ffill().fillna(0.0).rename_axis("trade_date")


def strict_cagr_dd(tw, win, smap_full):
    if tw.empty:
        return None
    a = run_strict_backtest_v8(tw, win, sector_map=smap_full)
    m = a.metrics
    return {"cagr": m.annualized_return, "maxdd": m.max_drawdown,
            "calmar": m.calmar, "turnover": m.turnover, "nav": a.nav}


def main():
    sc = pd.read_parquet(SCORE)[["trade_date", "symbol", SCORE_COL]]
    sc["trade_date"] = pd.to_datetime(sc["trade_date"])
    start, end = sc.trade_date.min(), sc.trade_date.max()
    smap_full = pd.read_parquet(SECTOR)
    smap = smap_full[["symbol", "sector_level_1"]].dropna().drop_duplicates("symbol")
    panel = pd.read_parquet(PANEL); panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    win = panel[(panel.trade_date >= start) & (panel.trade_date <= end)].copy()
    sp = pd.read_parquet(SECTOR_PANEL); sp["trade_date"] = pd.to_datetime(sp["trade_date"])
    sp = sp[(sp.trade_date >= start) & (sp.trade_date <= end)]
    eval_dates = sorted(win.trade_date.unique())

    flags = win[["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up"]]
    df = sc.merge(smap, on="symbol", how="left").merge(flags, on=["symbol", "trade_date"], how="left")
    df["bad"] = df[["is_st", "is_suspended", "is_limit_up"]].fillna(False).astype(bool).any(axis=1)
    df = df.rename(columns={SCORE_COL: "score", "sector_level_1": "sector"})
    stock_day = {d: g for d, g in df.groupby("trade_date")}
    sector_day = {d: g for d, g in sp.groupby("trade_date")}
    dsorted = sorted(pd.DatetimeIndex(eval_dates).unique())

    # configs: (label, sector_signal, ns, ps, reverse) at each target size
    plans = {
        30: [("plain_v89", None, 0, 0, True),
             ("L2 mom60-lag ns3", "mom_60", 3, 10, True),
             ("L2 mom60-lag ns5", "mom_60", 5, 6, True),
             ("L2 rs20-lag ns5", "rs_20", 5, 6, True),
             ("L2 mom60-lead ns5", "mom_60", 5, 6, False)],
        50: [("plain_v89", None, 0, 0, True),
             ("L2 mom60-lag ns5", "mom_60", 5, 10, True),
             ("L2 mom60-lag ns8", "mom_60", 8, 7, True),
             ("L2 rs20-lag ns5", "rs_20", 5, 10, True),
             ("L2 mom60-lead ns5", "mom_60", 5, 10, False)],
    }

    results = []
    for size, cfgs in plans.items():
        print(f"\n##### target size = {size} names (equal-weight, strict, 5 rebalance phases) #####")
        # plain navs per phase (for excess)
        plain_navs = {}
        for label, sig, ns, ps, rev in cfgs:
            cagrs, dds, calmars, turns, navs = [], [], [], [], {}
            for ph in PHASES:
                rebal = dsorted[ph::PERIOD]
                tw = build_tw(stock_day, sector_day, rebal_dates=rebal, eval_dates=eval_dates,
                              size=size, sector_signal=sig, n_sectors=ns, per_sector=ps, reverse=rev)
                r = strict_cagr_dd(tw, win, smap_full)
                if r is None:
                    continue
                cagrs.append(r["cagr"]); dds.append(r["maxdd"]); calmars.append(r["calmar"])
                turns.append(r["turnover"]); navs[ph] = r["nav"]
            if sig is None:
                plain_navs = navs
            # excess vs plain per phase
            excs = []
            for ph in PHASES:
                if ph in navs and ph in plain_navs:
                    rr = navs[ph].pct_change().dropna()
                    pp = plain_navs[ph].pct_change().dropna()
                    idx = rr.index.intersection(pp.index)
                    excs.append(_ann(rr.reindex(idx)) - _ann(pp.reindex(idx)))
            row = {"size": size, "config": label,
                   "cagr_mean": round(float(np.mean(cagrs)), 4),
                   "cagr_std": round(float(np.std(cagrs)), 4),
                   "maxdd_mean": round(float(np.mean(dds)), 4),
                   "calmar_mean": round(float(np.mean(calmars)), 3),
                   "turnover_mean": round(float(np.mean(turns)), 4),
                   "exc_v89_mean": round(float(np.mean(excs)), 4) if excs else None,
                   "exc_v89_std": round(float(np.std(excs)), 4) if excs else None,
                   "exc_v89_winrate": round(float(np.mean([e > 0 for e in excs])), 2) if excs else None}
            results.append(row)
            print(f"  {label:<22} CAGR={row['cagr_mean']:+.1%}±{row['cagr_std']:.1%}  "
                  f"DD={row['maxdd_mean']:.1%}  Calmar={row['calmar_mean']}  "
                  f"excV89={row['exc_v89_mean']:+.1%}±{row['exc_v89_std']:.1%} "
                  f"(win {row['exc_v89_winrate']})" if row['exc_v89_mean'] is not None
                  else f"  {label:<22} CAGR={row['cagr_mean']:+.1%}±{row['cagr_std']:.1%} (plain baseline)")

    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / "layer2_robustness.csv", index=False)
    (OUT_DIR / "layer2_robustness.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[write] {OUT_DIR/'layer2_robustness.csv'}")
    print("\nVERDICT GUIDE: layer-2 enhances v8.9 only if exc_v89_mean > 0 AND")
    print("exc_v89_mean > exc_v89_std (stable across phases) AND winrate >= 0.6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
