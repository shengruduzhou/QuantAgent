#!/usr/bin/env python3
"""Stage 9 step 3 — temporal OOS validation of the size-30 wave edge.

wave_full (regime-aware sector selection) beat plain by +7.6% (win 1.0) at
size 30 over the full 2020-2025. But the state-machine design + thresholds were
chosen AFTER seeing that span, so the edge could be full-sample fitting. Two
honest checks:

  A. UN-TUNED temporal stability — with the *default* (un-fit) WaveConfig,
     measure the size-30 wave_full excess vs plain PER YEAR and on two halves
     P1=2020-2022 / P2=2023-2025. If positive in both halves / most years, the
     edge is structural, not one-period luck.
  B. TRUE held-out calibration — lightly tune the regime thresholds on P1 only
     (fast engine), FREEZE, then evaluate ONCE on P2 with the strict engine.
     If the frozen-on-past config still beats plain on the untouched future, the
     edge survives honest OOS.

All size-30, equal-weight, 5 rebalance phases, strict A-share engine for the
headline numbers; fast per-stock engine only to rank the P1 calibration grid.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402
from quantagent.strategy.sector_wave_state import WaveConfig, select_wave  # noqa: E402

SCORE = "runtime/stage6_classical_2018/wf/walkforward_predictions.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
WAVE_PANEL = "runtime/stage8_sector_rotation/sector_panel_wave.parquet"
OUT_DIR = Path("runtime/stage8_sector_rotation")
ANN = 244
SCORE_COL = "alpha_20d"
SEC = "sector_level_1"
PHASES = [0, 4, 8, 12, 16]
PERIOD = 20
SIZE = 30
NSEC = 5


def _ann(d):
    n = len(d); return float((1 + d).prod() ** (ANN / n) - 1) if n else float("nan")


def build_tw(stock_day, sector_day, mkt_trend, *, rebal_dates, eval_dates, size,
             wave=False, cfg=WaveConfig()):
    rows = {}
    for d in rebal_dates:
        sd = stock_day.get(d)
        if sd is None or sd.empty:
            continue
        sd = sd[~sd["bad"]].dropna(subset=["score"])
        if wave:
            secf = sector_day.get(d)
            if secf is None or secf.empty:
                continue
            sel = select_wave(secf, mkt_trend.get(d, 0.0), n_sectors=NSEC, cfg=cfg)
            sd = sd[sd["sector"].isin(sel["sectors"])]
        if sd.empty:
            continue
        sd = sd.sort_values("score", ascending=False).head(size)
        w = 1.0 / len(sd)
        rows[d] = {s: w for s in sd["symbol"]}
    if not rows:
        return pd.DataFrame()
    tw = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()
    full = pd.DatetimeIndex([x for x in eval_dates if x >= tw.index.min()])
    return tw.reindex(full).ffill().fillna(0.0).rename_axis("trade_date")


def fast_nav(tw, ret_mat, cost_bps=18.0):
    if tw.empty:
        return pd.Series(dtype=float)
    cols = tw.columns.intersection(ret_mat.columns)
    tw = tw[cols].reindex(ret_mat.index).fillna(0.0)
    R = ret_mat[cols].reindex(tw.index)
    gross = (tw.shift(1).fillna(0.0) * R).sum(axis=1)
    turn = (tw - tw.shift(1)).abs().sum(axis=1) * 0.5
    cost = turn.shift(1).fillna(0.0) * cost_bps / 1e4
    return (1 + (gross - cost).fillna(0.0)).cumprod()


def excess(nav_w, nav_p, lo=None, hi=None):
    rw = nav_w.pct_change().dropna(); rp = nav_p.pct_change().dropna()
    idx = rw.index.intersection(rp.index)
    if lo is not None:
        idx = idx[(idx >= pd.Timestamp(lo)) & (idx <= pd.Timestamp(hi))]
    if len(idx) < 20:
        return None
    return _ann(rw.reindex(idx)) - _ann(rp.reindex(idx))


def load():
    sc = pd.read_parquet(SCORE)[["trade_date", "symbol", SCORE_COL]]
    sc["trade_date"] = pd.to_datetime(sc["trade_date"])
    sc = sc[(sc.trade_date >= "2020-01-23") & (sc.trade_date <= "2025-12-31")]
    smap_full = pd.read_parquet(SECTOR)
    smap = smap_full[["symbol", SEC]].dropna().drop_duplicates("symbol")
    panel = pd.read_parquet(PANEL); panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    win = panel[(panel.trade_date >= "2020-01-23") & (panel.trade_date <= "2025-12-31")].copy()
    wp = pd.read_parquet(WAVE_PANEL); wp["trade_date"] = pd.to_datetime(wp["trade_date"])
    wp = wp[(wp.trade_date >= "2020-01-23") & (wp.trade_date <= "2025-12-31")]
    eval_dates = sorted(win.trade_date.unique())
    flags = win[["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up"]]
    df = sc.merge(smap, on="symbol", how="left").merge(flags, on=["symbol", "trade_date"], how="left")
    df["bad"] = df[["is_st", "is_suspended", "is_limit_up"]].fillna(False).astype(bool).any(axis=1)
    df = df.rename(columns={SCORE_COL: "score", SEC: "sector"})
    stock_day = {d: g for d, g in df.groupby("trade_date")}
    sector_day = {d: g for d, g in wp.groupby("trade_date")}
    eqsector = wp.pivot_table(index="trade_date", columns=SEC, values="ret_eqw").mean(axis=1).dropna()
    cum = (1 + eqsector).cumprod()
    mkt_trend = (cum / cum.shift(60) - 1.0).to_dict()
    ret_mat = win.pivot_table(index="trade_date", columns="symbol", values="close").pct_change(fill_method=None).reindex(pd.DatetimeIndex(eval_dates))
    return dict(stock_day=stock_day, sector_day=sector_day, mkt_trend=mkt_trend,
                eval_dates=eval_dates, win=win, smap_full=smap_full, ret_mat=ret_mat,
                dsorted=sorted(pd.DatetimeIndex(eval_dates).unique()))


def main():
    D = load()
    rep = {}

    # ---------- A. un-tuned temporal stability (strict, default cfg) ----------
    print("=== A. UN-TUNED size-30 wave_full vs plain — per year + halves (strict) ===")
    wave_navs, plain_navs = {}, {}
    for ph in PHASES:
        rebal = D["dsorted"][ph::PERIOD]
        twp = build_tw(D["stock_day"], D["sector_day"], D["mkt_trend"], rebal_dates=rebal,
                       eval_dates=D["eval_dates"], size=SIZE, wave=False)
        tww = build_tw(D["stock_day"], D["sector_day"], D["mkt_trend"], rebal_dates=rebal,
                       eval_dates=D["eval_dates"], size=SIZE, wave=True)
        plain_navs[ph] = run_strict_backtest_v8(twp, D["win"], sector_map=D["smap_full"]).nav
        wave_navs[ph] = run_strict_backtest_v8(tww, D["win"], sector_map=D["smap_full"]).nav
        print(f"  phase {ph} done")
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    yr_exc = {y: [] for y in years}
    p1_exc, p2_exc = [], []
    for ph in PHASES:
        for y in years:
            e = excess(wave_navs[ph], plain_navs[ph], f"{y}-01-01", f"{y}-12-31")
            if e is not None:
                yr_exc[y].append(e)
        e1 = excess(wave_navs[ph], plain_navs[ph], "2020-01-01", "2022-12-31")
        e2 = excess(wave_navs[ph], plain_navs[ph], "2023-01-01", "2025-12-31")
        if e1 is not None: p1_exc.append(e1)
        if e2 is not None: p2_exc.append(e2)
    print("  per-year wave_full excess vs plain (mean over phases):")
    for y in years:
        v = yr_exc[y]
        if v:
            print(f"    {y}: {np.mean(v):+.1%}  (win {np.mean([x>0 for x in v]):.1f})")
    rep["per_year_excess"] = {str(y): round(float(np.mean(v)), 4) for y, v in yr_exc.items() if v}
    rep["P1_2020_2022_excess"] = {"mean": round(float(np.mean(p1_exc)), 4), "std": round(float(np.std(p1_exc)), 4),
                                  "win": round(float(np.mean([x > 0 for x in p1_exc])), 2)}
    rep["P2_2023_2025_excess"] = {"mean": round(float(np.mean(p2_exc)), 4), "std": round(float(np.std(p2_exc)), 4),
                                  "win": round(float(np.mean([x > 0 for x in p2_exc])), 2)}
    print(f"  P1 2020-2022: {rep['P1_2020_2022_excess']}")
    print(f"  P2 2023-2025: {rep['P2_2023_2025_excess']}")

    # ---------- B. calibrate thresholds on P1 (fast), freeze, strict-test P2 ----------
    print("\n=== B. calibrate WaveConfig on P1 (fast), freeze, strict-test on P2 ===")
    grid = list(itertools.product([0.50, 0.55, 0.60], [0.25, 0.30, 0.35], [0.03, 0.05, 0.08]))
    # P1 fast: plain navs per phase
    p1_lo, p1_hi = "2020-01-23", "2022-12-31"
    plain_fast_p1 = {ph: fast_nav(build_tw(D["stock_day"], D["sector_day"], D["mkt_trend"],
                                           rebal_dates=D["dsorted"][ph::PERIOD], eval_dates=D["eval_dates"],
                                           size=SIZE, wave=False), D["ret_mat"]) for ph in PHASES}
    best, best_exc = None, -1e9
    for bb, wb, ie in grid:
        cfg = WaveConfig(bull_breadth=bb, weak_breadth=wb, ignition_breadth_exp=ie)
        excs = []
        for ph in PHASES:
            tww = build_tw(D["stock_day"], D["sector_day"], D["mkt_trend"], rebal_dates=D["dsorted"][ph::PERIOD],
                           eval_dates=D["eval_dates"], size=SIZE, wave=True, cfg=cfg)
            e = excess(fast_nav(tww, D["ret_mat"]), plain_fast_p1[ph], p1_lo, p1_hi)
            if e is not None:
                excs.append(e)
        me = float(np.mean(excs)) if excs else -1e9
        if me > best_exc:
            best_exc, best = me, (bb, wb, ie)
    print(f"  best P1 cfg (fast): bull_breadth={best[0]} weak_breadth={best[1]} ignition_exp={best[2]} "
          f"(P1 fast excess {best_exc:+.1%})")
    cfg_best = WaveConfig(bull_breadth=best[0], weak_breadth=best[1], ignition_breadth_exp=best[2])
    # strict test on P2 with frozen cfg
    p2_lo, p2_hi = "2023-01-01", "2025-12-31"
    p2_exc_cal = []
    for ph in PHASES:
        tww = build_tw(D["stock_day"], D["sector_day"], D["mkt_trend"], rebal_dates=D["dsorted"][ph::PERIOD],
                       eval_dates=D["eval_dates"], size=SIZE, wave=True, cfg=cfg_best)
        nav_w = run_strict_backtest_v8(tww, D["win"], sector_map=D["smap_full"]).nav
        e = excess(nav_w, plain_navs[ph], p2_lo, p2_hi)
        if e is not None:
            p2_exc_cal.append(e)
    rep["calibration"] = {"best_cfg_P1": {"bull_breadth": best[0], "weak_breadth": best[1], "ignition_breadth_exp": best[2]},
                          "P1_fast_excess": round(best_exc, 4),
                          "P2_strict_excess_mean": round(float(np.mean(p2_exc_cal)), 4),
                          "P2_strict_excess_std": round(float(np.std(p2_exc_cal)), 4),
                          "P2_win": round(float(np.mean([x > 0 for x in p2_exc_cal])), 2)}
    print(f"  FROZEN cfg on held-out P2 (strict): excess={np.mean(p2_exc_cal):+.1%}"
          f"±{np.std(p2_exc_cal):.1%} win={np.mean([x>0 for x in p2_exc_cal]):.1f}")

    (OUT_DIR / "stage9_wave_oos.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"\n[write] {OUT_DIR/'stage9_wave_oos.json'}")
    print("\nVERDICT: time-stable if P1 AND P2 un-tuned excess > 0 (win>=0.6), and the")
    print("frozen-on-P1 config still beats plain on held-out P2 (strict).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
