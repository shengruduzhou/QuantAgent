#!/usr/bin/env python3
"""H-027 gate evaluation (preregistered, commit 4ca88e2 — applied verbatim).

Loads stage-1/2 daily IC series + fold metrics, computes the 8 preregistered
gates vs B0 for every candidate, decides conditional blends E2/E3, and writes
gate_verdicts.json + candidate_metrics.csv. Cost convention (declared):
costadj_daily = topdecile_h20/20 - topdecile_turnover*2*cost; gate6/7 use
25 bps. No gate is altered after observing results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runtime/reports/full_universe/fu_20260713_h027"
COST25 = 25 / 1e4


def load():
    ics = [pd.read_parquet(OUT / "daily_ic_stage1.parquet")]
    rows = [pd.read_csv(OUT / "stage1_fold_metrics.csv")]
    for tag in ("stage1b", "stage2"):
        if (OUT / f"daily_ic_{tag}.parquet").exists():
            ics.append(pd.read_parquet(OUT / f"daily_ic_{tag}.parquet"))
            rows.append(pd.read_csv(OUT / f"{tag}_fold_metrics.csv"))
    return pd.concat(ics, ignore_index=True), pd.concat(rows, ignore_index=True)


def costadj(row, bps=COST25):
    return row["topdecile_long_h20"] / 20.0 - row["topdecile_turnover"] * 2.0 * bps


def main() -> int:
    ics, fm = load()
    fm["costadj25"] = fm.apply(costadj, axis=1)
    b0 = fm[fm["candidate"] == "B0"].set_index("fold")
    b0_ic = {f: g.set_index("d")["ic"] for f, g in ics[ics["candidate"] == "B0"].groupby("fold")}
    verdicts = {}
    for cand, g in fm[fm["candidate"] != "B0"].groupby("candidate"):
        g = g.set_index("fold")
        folds = sorted(set(g.index) & set(b0.index))
        if len(folds) < 3:
            verdicts[cand] = {"status": "INCOMPLETE", "folds": len(folds), "passes": False}
            continue
        d_ic, diffs_pooled, d_dec, d_to, d_ca = [], [], [], [], []
        for f in folds:
            ci = ics[(ics["candidate"] == cand) & (ics["fold"] == f)].set_index("d")["ic"]
            al = pd.concat({"c": ci, "b": b0_ic[f]}, axis=1).dropna()
            d_ic.append(float((al["c"] - al["b"]).mean()))
            diffs_pooled.append(al["c"] - al["b"])
            d_dec.append(float(g.loc[f, "costadj25"] - b0.loc[f, "costadj25"]))
            d_to.append(float(g.loc[f, "topdecile_turnover"] / b0.loc[f, "topdecile_turnover"]))
            d_ca.append(float(g.loc[f, "costadj25"]))
        pooled = pd.concat(diffs_pooled)
        tstat, pval = stats.ttest_1samp(pooled, 0.0)
        gates = {
            "g1_median_dic_ge_005": bool(np.median(d_ic) >= 0.005),
            "g2_pos_folds_ge2": bool(sum(d > 0 for d in d_ic) >= 2),
            "g3_pooled_p_lt_010": bool(pval < 0.10 and tstat > 0),
            "g4_crash_guard": bool(d_ic[0] >= -0.002),
            "g5_sign_stable": bool(sum(d > 0 for d in d_ic) >= 2 or sum(d < 0 for d in d_ic) >= 2),
            "g6_costadj_improve": bool(np.median(d_dec) > 0),
            "g7_costadj25_positive": bool(np.median(d_ca) > 0),
            "g8_turnover_le_110pct": bool(np.median(d_to) <= 1.10),
        }
        verdicts[cand] = {
            "delta_ic_by_fold": [round(x, 5) for x in d_ic],
            "median_delta_ic": round(float(np.median(d_ic)), 5),
            "pooled_t": round(float(tstat), 2), "pooled_p": round(float(pval), 4),
            "delta_costadj25_by_fold": [round(x, 5) for x in d_dec],
            "costadj25_median": round(float(np.median(d_ca)), 5),
            "turnover_ratio_median": round(float(np.median(d_to)), 3),
            "gates": gates, "passes": bool(all(gates.values())),
        }
    passing = [c for c, v in verdicts.items() if v.get("passes")]
    summary = {
        "b0_mean_ic_by_fold": {int(f): round(float(r["mean_ic"]), 5) for f, r in b0.iterrows()},
        "b0_costadj25_by_fold": {int(f): round(float(r["costadj25"]), 5) for f, r in b0.iterrows()},
        "verdicts": verdicts, "passing": passing,
        "e2_e3_conditional": "components did not pass; E2/E3 not run" if not passing else "see blends",
        "strict_progression": "NOT entered (no passer)" if not passing else "S1 eligible",
        "trust": "searched_validation / candidate_research_only_not_fresh_holdout_validated",
    }
    (OUT / "gate_verdicts.json").write_text(json.dumps(summary, indent=2))
    fm.to_csv(OUT / "candidate_metrics.csv", index=False)
    print(json.dumps({k: {"median_delta_ic": v.get("median_delta_ic"),
                          "pooled_p": v.get("pooled_p"),
                          "passes": v.get("passes")} for k, v in verdicts.items()}, indent=1))
    print("passing:", passing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
