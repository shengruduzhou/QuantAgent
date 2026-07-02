#!/usr/bin/env python3
"""Stage 11 step 1 — beta/alpha decomposition of v8.9 + fundamental variants.

Answers the question the whole arc skipped: how much of v8.9's return is market
BETA vs real selection ALPHA? Runs each strategy through the strict engine (size
+ 5-phase matched), then decomposes daily returns vs all-A eqw + CSI300/500/1000
into beta, annualised Jensen alpha, r2, up/down capture, plus CAGR/Calmar/Sharpe/
MaxDD/turnover. Tests window-robustness over sub-periods and emits the Pareto
frontier + the user's strategy classification (beta_strategy / research_signal /
production_candidate / window_artifact).

PIT rule 九.1: NO current concept membership backfilled into history — benchmarks
here are all-A + cap indices only. (Concept-routed strategies are forward-only.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantagent.backtest.beta_decomposition import (  # noqa: E402
    ann_return, classify_strategy, full_panel)
from quantagent.backtest.strict_v8 import run_strict_backtest_v8  # noqa: E402

SCORE = "runtime/reports/v89_closed_loop/retrain_plus7_20260620_0300/ensemble_composite.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
FUND = "runtime/data/v7/silver/market_panel/market_panel_fund.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
INDEX = "runtime/data/v7/raw/akshare/index/equity_index.parquet"
OUT = Path("runtime/stage11")
PHASES, PERIOD = [0, 4, 8, 12, 16], 20
SUBWINDOWS = {"2024H2": ("2024-08-09", "2024-12-31"), "2025": ("2025-01-01", "2025-12-31"),
              "2026": ("2026-01-01", "2026-05-07")}


def _csrank(s): return s.rank(pct=True)


def build_book(stock_day, *, rebal_dates, eval_dates, size, score_col):
    rows = {}
    for d in rebal_dates:
        sd = stock_day.get(d)
        if sd is None or sd.empty:
            continue
        sd = sd[~sd["bad"]].dropna(subset=[score_col]).sort_values(score_col, ascending=False).head(size)
        if sd.empty:
            continue
        rows[d] = {s: 1.0 / len(sd) for s in sd["symbol"]}
    if not rows:
        return pd.DataFrame()
    tw = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()
    full = pd.DatetimeIndex([x for x in eval_dates if x >= tw.index.min()])
    return tw.reindex(full).ffill().fillna(0.0).rename_axis("trade_date")


def index_daily(label, dates):
    idx = pd.read_parquet(INDEX)
    idx = idx[idx["label"] == label].copy()
    idx["observation_date"] = pd.to_datetime(idx["observation_date"])
    return idx.set_index("observation_date")["close"].sort_index().pct_change().reindex(pd.DatetimeIndex(sorted(dates))).dropna()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[load] v8.9 score + fundamentals + indices ...")
    sc = pd.read_parquet(SCORE)[["trade_date", "symbol", "composite_score"]]
    sc["trade_date"] = pd.to_datetime(sc["trade_date"])
    start, end = sc.trade_date.min(), sc.trade_date.max()
    panel = pd.read_parquet(PANEL); panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    win = panel[(panel.trade_date >= start) & (panel.trade_date <= end)].copy()
    fund = pd.read_parquet(FUND, columns=["symbol", "trade_date", "roe", "gross_margin", "debt_to_asset", "pb"])
    fund["trade_date"] = pd.to_datetime(fund["trade_date"])
    fund = fund[(fund.trade_date >= start) & (fund.trade_date <= end)]
    smap = pd.read_parquet(SECTOR)
    eval_dates = sorted(win.trade_date.unique())
    print(f"  OOS {start.date()}..{end.date()} ({len(eval_dates)} days)")

    flags = win[["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up", "close"]]
    df = sc.merge(flags, on=["symbol", "trade_date"], how="left").merge(fund, on=["symbol", "trade_date"], how="left")
    df["bad"] = df[["is_st", "is_suspended", "is_limit_up"]].fillna(False).astype(bool).any(axis=1)
    g = df.groupby("trade_date")
    hard = (g["roe"].transform(_csrank) + g["gross_margin"].transform(_csrank)
            + (1 - g["debt_to_asset"].transform(_csrank)) + (1 - g["pb"].transform(_csrank))) / 4
    df["hardness"] = hard
    df["v89_x_hard_q"] = 0.75 * g["composite_score"].transform(_csrank) + 0.25 * g["hardness"].transform(_csrank)
    stock_day = {d: gg for d, gg in df.groupby("trade_date")}
    dsorted = sorted(pd.DatetimeIndex(eval_dates).unique())

    # benchmarks
    all_a = win.pivot_table(index="trade_date", columns="symbol", values="close").pct_change(fill_method=None).mean(axis=1).dropna()
    benches = {"all_a": all_a, "csi300": index_daily("csi300", eval_dates),
               "csi500": index_daily("csi500", eval_dates), "csi1000": index_daily("csi1000", eval_dates)}
    print("[bench] ann:", {k: round(ann_return(v), 3) for k, v in benches.items()})

    variants = {"composite_score": "plain_v89", "hardness": "hardness_only", "v89_x_hard_q": "v89_x_hard(75/25)"}
    results = []
    for size in (30, 50):
        for col, label in variants.items():
            navs, turns = [], []
            for ph in PHASES:
                tw = build_book(stock_day, rebal_dates=dsorted[ph::PERIOD], eval_dates=eval_dates, size=size, score_col=col)
                if tw.empty:
                    continue
                arts = run_strict_backtest_v8(tw, win, sector_map=smap)
                navs.append(arts.nav); turns.append(arts.metrics.turnover)
            if not navs:
                continue
            # average panel across phases; phase_std of excess_all_a for stability
            panels = []
            for nav in navs:
                r = nav.pct_change().dropna()
                panels.append(full_panel(r, nav, benches, turnover=float(np.mean(turns)), primary="all_a"))
            avg = {k: round(float(np.mean([p[k] for p in panels if p[k] is not None])), 4)
                   if any(p[k] is not None for p in panels) else None for k in panels[0]}
            phase_std = float(np.std([p["excess_all_a"] for p in panels]))
            # window robustness: alpha sign positive in >=2 of 3 sub-windows (phase-0 nav)
            nav0 = navs[0]; r0 = nav0.pct_change().dropna()
            sub_alpha = {}
            for wname, (a, z) in SUBWINDOWS.items():
                m = (r0.index >= pd.Timestamp(a)) & (r0.index <= pd.Timestamp(z))
                if m.sum() < 20:
                    continue
                sp = full_panel(r0[m], (1 + r0[m]).cumprod(), {"all_a": all_a}, primary="all_a")
                sub_alpha[wname] = sp.get("alpha_all_a")
            n_pos = sum(1 for v in sub_alpha.values() if v is not None and v > 0)
            multi_ok = n_pos >= 2
            label_cls, cflags = classify_strategy(avg, multi_window_ok=multi_ok, phase_std=phase_std,
                                                  max_weight=1.0 / size, primary="all_a")
            row = {"strategy": label, "size": size, "class": label_cls, "flags": ",".join(cflags),
                   "phase_std_exc": round(phase_std, 4), "sub_alpha": sub_alpha, "n_alpha_pos_windows": n_pos, **avg}
            results.append(row)
            print(f"\n[{label} size{size}] -> {label_cls} {cflags}")
            print(f"   CAGR={avg['cagr']:+.1%} DD={avg['maxdd']:.1%} Calmar={avg['calmar']} Sharpe={avg['sharpe']}")
            print(f"   vs all-A: beta={avg['beta_all_a']} alpha={avg['alpha_all_a']:+.1%} r2={avg['r2_all_a']} excess={avg['excess_all_a']:+.1%}")
            print(f"   vs csi300: beta={avg['beta_csi300']} alpha={avg['alpha_csi300']:+.1%} | up_cap={avg['up_capture']} down_cap={avg['down_capture']}")
            print(f"   sub-window alpha vs all-A: {sub_alpha}  (positive in {n_pos}/3)")

    res = pd.DataFrame(results)
    res.to_csv(OUT / "beta_decomposition.csv", index=False)
    (OUT / "beta_decomposition.json").write_text(json.dumps(results, indent=2, default=str))

    # ---- Pareto frontier ----
    print("\n" + "=" * 70 + "\n=== PARETO FRONTIER ===")
    def best(metric, lab):
        r = res.sort_values(metric, ascending=False).iloc[0]
        print(f"  {lab:<22}: {r['strategy']} size{r['size']} ({r['class']}) "
              f"CAGR={r['cagr']:+.1%} alpha_allA={r['alpha_all_a']:+.1%} Calmar={r['calmar']} upcap={r['up_capture']}")
    best("cagr", "max absolute CAGR")
    best("alpha_all_a", "max alpha (vs all-A)")
    best("calmar", "max Calmar")
    best("up_capture", "max bull capture")
    prod = res[res["class"] == "production_candidate"]
    bal = (prod if not prod.empty else res).sort_values(["calmar", "alpha_all_a"], ascending=False).iloc[0]
    print(f"  {'balanced production':<22}: {bal['strategy']} size{bal['size']} ({bal['class']}) "
          f"CAGR={bal['cagr']:+.1%} alpha={bal['alpha_all_a']:+.1%} Calmar={bal['calmar']}")
    print(f"\n[write] {OUT/'beta_decomposition.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
