#!/usr/bin/env python3
"""Track C factor batch 1 (DUAL_TRACK_FACTOR_BATCH_PLAN.md): PIT-safe defensive /
low-turnover factors evaluated on the pre-quarantine H-008 fold span.

CPU-only, no fresh/burned holdout (all data < 2025-09-01, asserted). Writes
FACTOR_CANDIDATE_LEDGER.csv + a per-factor verdict.
"""
from __future__ import annotations
import sys
import time
import resource
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import baseline_protocol as bp  # noqa: E402
from quantagent.factors import expr as E  # noqa: E402
from quantagent.factors.evaluation import (  # noqa: E402
    forward_return_labels, information_coefficient, quantile_group_backtest, capacity_proxy,
)

QUAR = pd.Timestamp("2025-09-01")
WIN = (pd.Timestamp("2023-07-03"), pd.Timestamp("2025-08-29"))
F2 = (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-06-28"))
EPS = E.Constant(1e-6)


def eps_add(x):
    return E.Add(x, EPS)


FACTORS = {
    "D1_low_vol_20": ("defensive", E.Mul(E.Constant(-1.0), E.TsStd(E.Returns(E.Close, 1), 20))),
    "D2_trend_quality_60": ("low_turnover", E.Div(E.Returns(E.Close, 60), eps_add(E.TsStd(E.Returns(E.Close, 1), 60)))),
    "D3_near_high_120": ("defensive", E.Div(E.Close, E.TsMax(E.Close, 120))),
    "D4_liquidity_amount_60": ("liquidity", E.TsMean(E.Amount, 60)),
    "D5_amihud_illiq_neg_20": ("liquidity", E.Mul(E.Constant(-1.0),
        E.TsMean(E.Div(E.Abs(E.Returns(E.Close, 1)), E.Add(E.Amount, E.Constant(1.0))), 20))),
    "D6_vol_compression": ("defensive", E.Mul(E.Constant(-1.0),
        E.Div(E.TsStd(E.Returns(E.Close, 1), 5), eps_add(E.TsStd(E.Returns(E.Close, 1), 60))))),
    "D7_downside_range_neg_20": ("defensive", E.Mul(E.Constant(-1.0),
        E.TsMean(E.Div(E.Sub(E.High, E.Low), E.Close), 20))),
}
# references for decorrelation (technical winners, so a fundamental batch's
# orthogonality to the existing signal is measured directly)
REF = {"mom20": E.Returns(E.Close, 20), "liq": E.TsMean(E.Amount, 20),
       "lowvol20": E.Mul(E.Constant(-1.0), E.TsStd(E.Returns(E.Close, 1), 20))}
FIN = REPO / "runtime/data/v7/gold/training_dataset/tickflow_fin_features.parquet"
# fundamental quality/growth factors (batch=fundamental). PIT-safe: tickflow
# features step on publication dates; an extra 1-day per-symbol lag is applied.
FUND_COLS = ["roe", "net_margin", "gross_margin", "revenue_yoy", "net_income_yoy"]
FUND_FACTORS = {  # name -> (class, source spec)
    "QF_roe": ("fundamental", ["roe"]),
    "QF_net_margin": ("fundamental", ["net_margin"]),
    "QF_gross_margin": ("fundamental", ["gross_margin"]),
    "QF_revenue_yoy": ("fundamental", ["revenue_yoy"]),
    "QF_net_income_yoy": ("fundamental", ["net_income_yoy"]),
    "QF_quality": ("fundamental", ["roe", "net_margin", "gross_margin"]),  # rank-mean composite
    "QF_growth": ("fundamental", ["revenue_yoy", "net_income_yoy"]),        # rank-mean composite
}


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def _load_panel():
    panel = pd.read_parquet(REPO / bp.PANEL,
        columns=["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"],
        filters=[("trade_date", ">=", WIN[0] - pd.Timedelta(days=200)), ("trade_date", "<=", WIN[1])])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    assert panel["trade_date"].max() < QUAR, "quarantine breach"
    return panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def build_dsl_frame():
    panel = _load_panel()
    cols = {n: ex.evaluate(panel).to_numpy() for n, (_, ex) in FACTORS.items()}
    for n, ex in REF.items():
        cols[n] = ex.evaluate(panel).to_numpy()
    for k, v in cols.items():
        panel[k] = v
    return panel, {n: c for n, (c, _) in FACTORS.items()}, list(REF)


def build_fundamental_frame():
    panel = _load_panel()
    # refs (technical) still computed on price for orthogonality measurement
    for n, ex in REF.items():
        panel[n] = ex.evaluate(panel).to_numpy()
    fin = pd.read_parquet(FIN, columns=["symbol", "trade_date"] + FUND_COLS)
    fin["trade_date"] = pd.to_datetime(fin["trade_date"])
    # extra PIT safety: 1-trading-day per-symbol lag (feature at t uses info <= t-1)
    fin = fin.sort_values(["symbol", "trade_date"])
    fin[FUND_COLS] = fin.groupby("symbol", sort=False)[FUND_COLS].shift(1)
    n0 = len(panel)
    panel = panel.merge(fin, on=["symbol", "trade_date"], how="left")
    assert len(panel) == n0, f"fundamental merge fan-out {n0}->{len(panel)}"
    # per-date cross-sectional rank of each raw fundamental (robust to ROE outliers),
    # then factor columns = rank (single) or rank-mean (composite)
    for c in FUND_COLS:
        panel[f"_r_{c}"] = panel.groupby("trade_date")[c].rank(pct=True)
    for name, (_, srcs) in FUND_FACTORS.items():
        panel[name] = panel[[f"_r_{c}" for c in srcs]].mean(axis=1)
    return panel, {n: c for n, (c, _) in FUND_FACTORS.items()}, list(REF)


def score_factors(lab, factor_meta, ref_names, out_csv):
    """Shared scoring: per-factor IC/turnover/cost/crash + decorrelation
    clustering (keep-best) + verdict. `lab` already has factor + ref columns and
    forward_return_{10,20}d; `factor_meta` = {name: class}."""
    fac_names = list(factor_meta)
    rank_frame = {n: lab.groupby("trade_date")[n].rank(pct=True) for n in fac_names + ref_names}
    corr = pd.DataFrame(rank_frame).corr(method="spearman")

    rows = []
    for name, cls in factor_meta.items():
        sub = lab.dropna(subset=[name, "forward_return_10d"])
        ic10 = information_coefficient(sub, name, "forward_return_10d").summary
        sub20 = lab.dropna(subset=[name, "forward_return_20d"])
        ic20 = information_coefficient(sub20, name, "forward_return_20d").summary
        qb8 = quantile_group_backtest(sub, name, "forward_return_10d", quantiles=5, cost_bps=8.0)
        qb15 = quantile_group_backtest(sub, name, "forward_return_10d", quantiles=5, cost_bps=15.0)
        qb25 = quantile_group_backtest(sub, name, "forward_return_10d", quantiles=5, cost_bps=25.0)
        f2 = lab[(lab["trade_date"] >= F2[0]) & (lab["trade_date"] <= F2[1])].dropna(subset=[name, "forward_return_10d"])
        f2ic = information_coefficient(f2, name, "forward_return_10d").summary.mean_rank_ic if len(f2) else np.nan
        cap = capacity_proxy(sub, name)
        others = [c for c in fac_names if c != name]
        max_decorr = float(max(abs(corr.loc[name, o]) for o in others)) if others else 0.0
        max_ref = float(max(abs(corr.loc[name, r]) for r in ref_names))
        to = float(qb8.turnover.mean())
        ls8, ls25 = float(qb8.cost_adjusted_long_short.mean()), float(qb25.cost_adjusted_long_short.mean())
        # gates. NOTE: factors are oriented high=good, so acceptance requires a
        # POSITIVE oriented IC (long side predicts higher returns). A negative IC
        # would need a sign-flip whose long side here is illiquid/reversal names
        # = capacity trap for a capacity-aware long book -> rejected, not silently
        # sign-flipped. Decorrelation clustering (keep-best) is applied post-loop.
        g_ic = ic10.mean_rank_ic >= 0.015 and (ic10.rank_icir >= 0.20 or ic20.rank_icir >= 0.20)
        g_turn = to <= 0.15
        g_cost = ls8 > 0 and ls25 > 0  # oriented long-short survives costs, both positive
        g_crash = (f2ic >= 0) if cls == "defensive" else True
        rows.append({
            "factor": name, "class": cls,
            "rank_ic_h10": round(ic10.mean_rank_ic, 4), "rank_icir_h10": round(ic10.rank_icir, 3),
            "rank_ic_h20": round(ic20.mean_rank_ic, 4), "rank_icir_h20": round(ic20.rank_icir, 3),
            "pos_ratio_h10": round(ic10.positive_ratio, 3),
            "topq_turnover": round(to, 4), "avg_hold_days": round(1.0 / to, 1) if to > 0 else np.nan,
            "ls_costadj_8bps": round(ls8, 5), "ls_costadj_15bps": round(float(qb15.cost_adjusted_long_short.mean()), 5),
            "ls_costadj_25bps": round(ls25, 5), "monotonicity_h10": round(qb8.monotonicity, 3),
            "f2_crash_ic_h10": round(float(f2ic), 4),
            "max_decorr_other": round(max_decorr, 3), "max_corr_ref": round(max_ref, 3),
            "capacity_rmb": round(cap.capacity_rmb / 1e8, 3),  # 亿元
            "g_ic_pos": bool(g_ic), "g_turn": bool(g_turn), "g_cost": bool(g_cost),
            "g_crash": bool(g_crash),
        })

    # ---- decorrelation clustering (keep-best): among factors that PASS the
    # standalone gates and are >0.90 correlated, keep only the highest |ICIR|.
    df = pd.DataFrame(rows).set_index("factor")
    passes_solo = {n: bool(df.loc[n, "g_ic_pos"] and df.loc[n, "g_turn"]
                           and df.loc[n, "g_cost"] and df.loc[n, "g_crash"]) for n in df.index}
    verdicts, g_decorr = {}, {}
    for n in df.index:
        if not passes_solo[n]:
            verdicts[n] = "discard" if abs(df.loc[n, "rank_ic_h10"]) < 0.008 else "reject"
            g_decorr[n] = None
            continue
        # cluster of solo-passers correlated > 0.90 with n (incl. n)
        cluster = [n] + [o for o in df.index if o != n and passes_solo[o] and abs(corr.loc[n, o]) > 0.90]
        best = max(cluster, key=lambda c: abs(df.loc[c, "rank_icir_h10"]))
        redundant = len(cluster) > 1 and n != best
        g_decorr[n] = not redundant
        verdicts[n] = "redundant" if redundant else "accept"
    df["g_decorr"] = [g_decorr[n] for n in df.index]
    df["verdict"] = [verdicts[n] for n in df.index]
    df = df.reset_index()
    df.to_csv(out_csv, index=False)
    for _, r in df.iterrows():
        print(f"{r['factor']:26s}[{r['class']:12s}] IC10 {r['rank_ic_h10']:+.4f} ICIR {r['rank_icir_h10']:+.2f} "
              f"turn {r['topq_turnover']:.3f} F2ic {r['f2_crash_ic_h10']:+.4f} LS25 {r['ls_costadj_25bps']:+.5f} "
              f"decorr {r['max_decorr_other']:.2f} refcorr {r['max_corr_ref']:.2f} -> {r['verdict']}", flush=True)
    print(f"\nwrote {out_csv.name} | accept={sum(df.verdict=='accept')} "
          f"redundant={sum(df.verdict=='redundant')} reject={sum(df.verdict=='reject')} "
          f"discard={sum(df.verdict=='discard')}")
    return df


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", choices=["dsl", "fundamental"], default="dsl")
    args = ap.parse_args()
    t0 = time.time()
    if args.batch == "dsl":
        panel, meta, refs = build_dsl_frame()
        out = REPO / "FACTOR_CANDIDATE_LEDGER.csv"
    else:
        panel, meta, refs = build_fundamental_frame()
        out = REPO / "FACTOR_CANDIDATE_LEDGER_fundamental.csv"
    lab = forward_return_labels(panel, horizons=(10, 20))
    lab = lab[(lab["trade_date"] >= WIN[0]) & (lab["trade_date"] <= WIN[1])].copy()
    score_factors(lab, meta, refs, out)
    print(f"peak RSS {rss_gib():.2f} GiB, {time.time()-t0:.0f}s [batch={args.batch}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
