#!/usr/bin/env python3
"""Stage 10 step 3 — PIT-able validation: does 概念硬度's fundamental signal add to v8.9?

The hardness rubric's NEW ingredient vs v8.9 (a price/alpha-factor model) is the
FUNDAMENTAL layer: quality (ROE, 毛利率, low leverage) + value (low PB) + earnings.
These are PIT-able from market_panel_fund (roe/gross_margin/debt_to_asset/pb,
99% populated, quarterly-updating). This isolates and tests that orthogonal
signal — without the look-ahead-contaminated concept-membership filter.

Variants (equal-weight, strict A-share engine, size + 5-rebalance-phase matched,
vs plain v8.9 on SAME dates/universe — the Stage 9 rigor):
  plain_v89        top-K by composite_score (baseline)
  hardness_only    top-K by fundamental hardness alone
  v89_x_hardness   top-K by rank-avg(v89, hardness)  <- does fundamentals ADD?
  v89_x_hard_q     v89 with a mild hardness tilt (75/25)

Verdict (user's rule): the fundamental hardness layer adds value only if a blend
beats plain v8.9 on excess with exc_mean > exc_std and win >= 0.6. If not, the
concept-hardness fundamental component adds nothing over v8.9.
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
FUND = "runtime/data/v7/silver/market_panel/market_panel_fund.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
OUT = Path("runtime/stage10_concept")
ANN = 244
PHASES = [0, 4, 8, 12, 16]
PERIOD = 20


def _ann(d):
    n = len(d); return float((1 + d).prod() ** (ANN / n) - 1) if n else float("nan")


def _csrank(s):  # cross-sectional percentile rank within a date
    return s.rank(pct=True)


def build_book(stock_day, *, rebal_dates, eval_dates, size, score_col):
    rows = {}
    for d in rebal_dates:
        sd = stock_day.get(d)
        if sd is None or sd.empty:
            continue
        sd = sd[~sd["bad"]].dropna(subset=[score_col])
        if sd.empty:
            continue
        sd = sd.sort_values(score_col, ascending=False).head(size)
        w = 1.0 / len(sd)
        rows[d] = {s: w for s in sd["symbol"]}
    if not rows:
        return pd.DataFrame()
    tw = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()
    full = pd.DatetimeIndex([x for x in eval_dates if x >= tw.index.min()])
    return tw.reindex(full).ffill().fillna(0.0).rename_axis("trade_date")


def main():
    print("[load] v8.9 score + fundamentals ...")
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
    df = sc.merge(flags, on=["symbol", "trade_date"], how="left").merge(
        fund, on=["symbol", "trade_date"], how="left")
    df["bad"] = df[["is_st", "is_suspended", "is_limit_up"]].fillna(False).astype(bool).any(axis=1)

    # PIT fundamental hardness = quality (roe, margin, -leverage) + value (-pb), cross-sectional per date
    g = df.groupby("trade_date")
    df["q_roe"] = g["roe"].transform(_csrank)
    df["q_gm"] = g["gross_margin"].transform(_csrank)
    df["q_lev"] = 1 - g["debt_to_asset"].transform(_csrank)
    df["v_pb"] = 1 - g["pb"].transform(_csrank)
    df["hardness"] = df[["q_roe", "q_gm", "q_lev", "v_pb"]].mean(axis=1)
    # blends with v8.9 (cross-sectional rank average)
    df["r_v89"] = g["composite_score"].transform(_csrank)
    df["r_hard"] = g["hardness"].transform(_csrank)
    df["v89_x_hardness"] = (df["r_v89"] + df["r_hard"]) / 2.0
    df["v89_x_hard_q"] = 0.75 * df["r_v89"] + 0.25 * df["r_hard"]

    stock_day = {d: gg for d, gg in df.groupby("trade_date")}
    dsorted = sorted(pd.DatetimeIndex(eval_dates).unique())

    variants = ["composite_score", "hardness", "v89_x_hardness", "v89_x_hard_q"]
    labels = {"composite_score": "plain_v89", "hardness": "hardness_only",
              "v89_x_hardness": "v89_x_hardness(50/50)", "v89_x_hard_q": "v89_x_hard(75/25)"}
    results = []
    for size in (30, 50):
        print(f"\n##### size={size}, 5 phases, strict #####")
        plain = {}
        for col in variants:
            navs = {}
            for ph in PHASES:
                tw = build_book(stock_day, rebal_dates=dsorted[ph::PERIOD], eval_dates=eval_dates,
                                size=size, score_col=col)
                nav = run_strict_backtest_v8(tw, win, sector_map=smap).nav if not tw.empty else None
                if nav is not None:
                    navs[ph] = nav
            if col == "composite_score":
                plain = navs
            cagrs, dds, excs = [], [], []
            for ph, nav in navs.items():
                r = nav.pct_change().dropna(); peak = nav.cummax()
                dds.append(float(abs((nav / peak - 1).min()))); cagrs.append(_ann(r))
                if ph in plain:
                    pr = plain[ph].pct_change().dropna()
                    idx = r.index.intersection(pr.index)
                    excs.append(_ann(r.reindex(idx)) - _ann(pr.reindex(idx)))
            row = {"size": size, "variant": labels[col],
                   "cagr": round(float(np.mean(cagrs)), 4), "cagr_std": round(float(np.std(cagrs)), 4),
                   "maxdd": round(float(np.mean(dds)), 4),
                   "calmar": round(float(np.mean(cagrs)) / float(np.mean(dds)), 3) if np.mean(dds) > 1e-9 else None,
                   "exc_v89": round(float(np.mean(excs)), 4) if excs else None,
                   "exc_std": round(float(np.std(excs)), 4) if excs else None,
                   "win": round(float(np.mean([e > 0 for e in excs])), 2) if excs else None}
            results.append(row)
            ex = (f"excV89={row['exc_v89']:+.1%}±{row['exc_std']:.1%}(win{row['win']})"
                  if row["exc_v89"] is not None else "(baseline)")
            print(f"  {row['variant']:<22} CAGR={row['cagr']:+.1%}±{row['cagr_std']:.1%} "
                  f"DD={row['maxdd']:.1%} Calmar={row['calmar']} {ex}")

    pd.DataFrame(results).to_csv(OUT / "hardness_validation.csv", index=False)
    (OUT / "hardness_validation.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[write] {OUT/'hardness_validation.csv'}")
    print("VERDICT: fundamental hardness adds to v8.9 only if a blend has exc_v89>0, exc>std, win>=0.6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
