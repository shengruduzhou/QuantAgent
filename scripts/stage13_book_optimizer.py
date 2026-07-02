#!/usr/bin/env python3
"""Stage 13 — beta-aware book optimizer over v8.9 (nested walk-forward).

Searches BOOK CONSTRUCTION on the fixed (already-OOS) v8.9 composite score to
maximise after-cost absolute CAGR, then decomposes beta/alpha — WITHOUT tuning
and reporting on the same window. The v8.9 score is not re-fit; only the book
rules are searched:

  topK        10/20/30/50/80/100
  rebalance   5/10/20 trading days
  weighting   equal / score / vol_cap(inverse-vol) / liquidity(amount)
  single_cap  max per-name weight (0.05 / 0.10 / none)

Nested walk-forward: for each fold, the fast per-stock engine ranks the grid on
the VALIDATION window (phase-averaged, cost-charged) and picks the best config;
that single config is then run on the UNTOUCHED TEST window through the STRICT
A-share engine. Test segments are stitched into one OOS NAV and beta-decomposed.
A full-window in-sample grid gives the Pareto frontier shape (clearly labelled).

PIT rule: no concept membership used (Line-A only). Score is fixed OOS.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantagent.backtest import beta_decomposition as bd  # noqa: E402
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

SCORE = "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/ensemble_composite.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
FEAT = "runtime/stage8_sector_rotation/stock_features.parquet"
INDEX = "runtime/data/v7/raw/akshare/index/equity_index.parquet"
OUT = Path("runtime/stage13")
PHASES_FAST = [0, 10]     # 2 phases for fast validation ranking (timing-luck guard)

TOPK = [10, 20, 30, 50, 80, 100]
REBAL = [5, 10, 20]
WEIGHTING = ["equal", "score", "vol_cap", "liquidity"]
SINGLE_CAP = [0.10, None]


def build_weights(sd: pd.DataFrame, size: int, weighting: str, single_cap):
    d = sd.sort_values("composite_score", ascending=False).head(size).copy()
    if d.empty:
        return {}
    if weighting == "equal":
        w = pd.Series(1.0, index=d["symbol"])
    elif weighting == "score":
        s = d.set_index("symbol")["composite_score"]
        w = (s - s.min() + 1e-6)
    elif weighting == "vol_cap":
        iv = 1.0 / (d.set_index("symbol")["vol60"].abs().fillna(d["vol60"].median()) + 1e-6)
        w = iv.clip(upper=iv.quantile(0.9))
    elif weighting == "liquidity":
        w = d.set_index("symbol")["amt20"].fillna(d["amt20"].median()).clip(lower=0)
    else:
        raise ValueError(weighting)
    w = w / w.sum()
    if single_cap:
        for _ in range(3):
            over = w > single_cap
            if not over.any():
                break
            excess = (w[over] - single_cap).sum()
            w[over] = single_cap
            under = ~over
            w[under] = w[under] + excess * w[under] / w[under].sum()
        w = w / w.sum()
    return w.to_dict()


def build_book(stock_day, *, rebal_dates, eval_dates, size, weighting, single_cap):
    rows = {}
    for d in rebal_dates:
        sd = stock_day.get(d)
        if sd is None or sd.empty:
            continue
        sd = sd[~sd["bad"]].dropna(subset=["composite_score"])
        if sd.empty:
            continue
        wd = build_weights(sd, size, weighting, single_cap)
        if wd:
            rows[d] = wd
    if not rows:
        return pd.DataFrame()
    tw = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()
    full = pd.DatetimeIndex([x for x in eval_dates if x >= tw.index.min()])
    return tw.reindex(full).ffill().fillna(0.0).rename_axis("trade_date")


def fast_nav(tw, ret_mat, cost_bps=18.0):
    cols = tw.columns.intersection(ret_mat.columns)
    tw = tw[cols].reindex(ret_mat.index).fillna(0.0)
    R = ret_mat[cols].reindex(tw.index)
    gross = (tw.shift(1).fillna(0.0) * R).sum(axis=1)
    turn = (tw - tw.shift(1)).abs().sum(axis=1) * 0.5
    cost = turn.shift(1).fillna(0.0) * cost_bps / 1e4
    return (1 + (gross - cost).fillna(0.0)).cumprod()


def index_daily(label, dates):
    idx = pd.read_parquet(INDEX); idx = idx[idx["label"] == label].copy()
    idx["observation_date"] = pd.to_datetime(idx["observation_date"])
    return idx.set_index("observation_date")["close"].sort_index().pct_change().reindex(pd.DatetimeIndex(sorted(dates))).dropna()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sc = pd.read_parquet(SCORE)[["trade_date", "symbol", "composite_score"]]
    sc["trade_date"] = pd.to_datetime(sc["trade_date"])
    start, end = sc.trade_date.min(), sc.trade_date.max()
    panel = pd.read_parquet(PANEL); panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    win = panel[(panel.trade_date >= start) & (panel.trade_date <= end)].copy()
    smap = pd.read_parquet(SECTOR)
    feat = pd.read_parquet(FEAT, columns=["symbol", "trade_date", "vol60", "amt20"])
    feat["trade_date"] = pd.to_datetime(feat["trade_date"])
    eval_dates = sorted(win.trade_date.unique())
    flags = win[["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up"]]
    df = (sc.merge(flags, on=["symbol", "trade_date"], how="left")
          .merge(feat, on=["symbol", "trade_date"], how="left"))
    df["bad"] = df[["is_st", "is_suspended", "is_limit_up"]].fillna(False).astype(bool).any(axis=1)
    stock_day = {d: g for d, g in df.groupby("trade_date")}
    dsorted = sorted(pd.DatetimeIndex(eval_dates).unique())
    ret_mat = win.pivot_table(index="trade_date", columns="symbol", values="close").pct_change(fill_method=None).reindex(pd.DatetimeIndex(eval_dates))
    all_a = ret_mat.mean(axis=1).dropna()
    benches = {"all_a": all_a, "csi300": index_daily("csi300", eval_dates),
               "csi500": index_daily("csi500", eval_dates), "csi1000": index_daily("csi1000", eval_dates)}
    print(f"[opt] v8.9 OOS {start.date()}..{end.date()} ({len(eval_dates)}d) | all-A ann={bd.ann_return(all_a):+.1%}")

    configs = list(itertools.product(TOPK, REBAL, WEIGHTING, SINGLE_CAP))
    print(f"[grid] {len(configs)} book configs")

    def fast_cagr(cfg, lo, hi):
        K, RB, W, C = cfg
        cagrs = []
        for ph in PHASES_FAST:
            rebal = [d for d in dsorted[ph::RB] if lo <= d <= hi]
            tw = build_book(stock_day, rebal_dates=rebal, eval_dates=[d for d in eval_dates if lo <= d <= hi],
                            size=K, weighting=W, single_cap=C)
            if tw.empty:
                continue
            nav = fast_nav(tw, ret_mat.loc[(ret_mat.index >= lo) & (ret_mat.index <= hi)])
            cagrs.append(bd.ann_return(nav.pct_change().dropna()))
        return float(np.mean(cagrs)) if cagrs else -1e9

    # ---- nested walk-forward ----
    folds = [("2024-08-09", "2025-03-31", "2025-04-01", "2025-07-31"),
             ("2024-08-09", "2025-07-31", "2025-08-01", "2025-11-30"),
             ("2024-08-09", "2025-11-30", "2025-12-01", "2026-05-07")]
    test_navs = []
    picks = []
    for vlo, vhi, tlo, thi in folds:
        vlo, vhi, tlo, thi = map(pd.Timestamp, (vlo, vhi, tlo, thi))
        best = max(configs, key=lambda c: fast_cagr(c, vlo, vhi))
        K, RB, W, C = best
        # strict on untouched test
        test_dates = [d for d in eval_dates if tlo <= d <= thi]
        rebal = [d for d in dsorted[0::RB] if tlo <= d <= thi]
        tw = build_book(stock_day, rebal_dates=rebal, eval_dates=test_dates, size=K, weighting=W, single_cap=C)
        wtest = win[(win.trade_date >= tlo) & (win.trade_date <= thi)]
        arts = run_strict_backtest_v8(tw, wtest, sector_map=smap)
        test_navs.append(arts.nav)
        picks.append({"val": f"{vlo.date()}..{vhi.date()}", "test": f"{tlo.date()}..{thi.date()}",
                      "config": {"topK": K, "rebalance": RB, "weighting": W, "single_cap": C},
                      "test_cagr": round(arts.metrics.annualized_return, 4),
                      "test_maxdd": round(arts.metrics.max_drawdown, 4),
                      "test_turnover": round(arts.metrics.turnover, 4)})
        print(f"  fold test {tlo.date()}..{thi.date()}: picked K={K} rb={RB} {W} cap={C} "
              f"-> test CAGR={arts.metrics.annualized_return:+.1%} DD={arts.metrics.max_drawdown:.1%}")

    # stitch OOS test NAV
    stitched = pd.concat([n.pct_change().dropna() for n in test_navs]).sort_index()
    stitched = stitched[~stitched.index.duplicated()]
    oos_nav = (1 + stitched).cumprod()
    panel = bd.full_panel(stitched, oos_nav, benches, primary="all_a")
    config_stable = len({tuple(p["config"].items()) for p in picks}) == 1
    print("\n=== NESTED WALK-FORWARD OOS (stitched test segments) ===")
    print(f"  CAGR={panel['cagr']:+.1%} MaxDD={panel['maxdd']:.1%} Calmar={panel['calmar']} Sharpe={panel['sharpe']}")
    print(f"  vs all-A : beta={panel['beta_all_a']} Jensen-alpha={panel['alpha_all_a']:+.1%} excess={panel['excess_all_a']:+.1%}")
    print(f"  vs csi300: alpha={panel['alpha_csi300']:+.1%} | up_cap={panel['up_capture']} down_cap={panel['down_capture']}")
    print(f"  selected-config stable across folds: {config_stable}  (picks: {[p['config'] for p in picks]})")

    # ---- in-sample Pareto (full window, labelled) ----
    print("\n=== IN-SAMPLE PARETO (full window — for frontier shape only) ===")
    full_lo, full_hi = pd.Timestamp(start), pd.Timestamp(end)
    grid_rows = []
    for cfg in configs:
        K, RB, W, C = cfg
        rebal = [d for d in dsorted[0::RB]]
        tw = build_book(stock_day, rebal_dates=rebal, eval_dates=eval_dates, size=K, weighting=W, single_cap=C)
        nav = fast_nav(tw, ret_mat)
        r = nav.pct_change().dropna()
        p = bd.full_panel(r, nav, {"all_a": all_a}, primary="all_a")
        grid_rows.append({"topK": K, "rebalance": RB, "weighting": W, "single_cap": C,
                          "cagr": p["cagr"], "calmar": p["calmar"], "alpha_all_a": p["alpha_all_a"],
                          "beta_all_a": p["beta_all_a"], "up_capture": p["up_capture"]})
    g = pd.DataFrame(grid_rows)
    g.to_csv(OUT / "book_grid_insample.csv", index=False)
    for lab, col in [("max CAGR", "cagr"), ("max alpha", "alpha_all_a"), ("max Calmar", "calmar"), ("max bull-cap", "up_capture")]:
        r = g.sort_values(col, ascending=False).iloc[0]
        print(f"  {lab:<14}: K={int(r.topK)} rb={int(r.rebalance)} {r.weighting} cap={r.single_cap} "
              f"CAGR={r.cagr:+.1%} alpha={r.alpha_all_a:+.1%} Calmar={r.calmar} beta={r.beta_all_a}")

    (OUT / "book_optimizer.json").write_text(json.dumps(
        {"walk_forward_picks": picks, "oos_panel": panel, "config_stable": config_stable}, indent=2, default=str))
    print(f"\n[write] {OUT/'book_optimizer.json'} , {OUT/'book_grid_insample.csv'}")
    print("NOTE: OOS = nested walk-forward stitched test (honest). Pareto = in-sample frontier shape.")
    print("      Window is 2024-08..2026-05 momentum bull; bear regime under-sampled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
