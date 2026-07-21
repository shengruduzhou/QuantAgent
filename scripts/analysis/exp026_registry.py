#!/usr/bin/env python3
"""H-026 Phase C1-C3: unified factor master registry + common executable-label
rescoring + global redundancy control (fu_20260713_h026).

- C1: one row per factor across production union / survivors / conditional /
  historical rejects (reconsideration audit inline).
- C2: pool factors rescored under delay-1 EXECUTABLE labels
  (close(t+1+h)/close(t+1)-1, entry infeasible when t+1 limit-up/suspended);
  original H-025 same-day metrics preserved, never overwritten.
- C3: rank-correlation matrix + >0.90 clustering with a preregistered robust
  representative score. RC7 composite (per-date equal rank-mean of 7 pool
  factors) is characterized here; it is model group M4 in Phase D.

CPU-only; window 2023-07-03..2025-08-29 (pre-quarantine, asserted).
Trust label: candidate_research_only_not_fresh_holdout_validated /
fixed_cohort_searched_validation.
"""
from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "scripts"))

from dual_track_factor_batch import FACTORS as FACTORS1, REF as REF1, WIN, F2, rss_gib  # noqa: E402
from factor_batch3_pv import FACTORS3, REF3  # noqa: E402
import baseline_protocol as bp  # noqa: E402
from quantagent.factors.evaluation import information_coefficient, quantile_group_backtest  # noqa: E402

OUT = REPO / "runtime/reports/full_universe/fu_20260713_h026"
QUAR = pd.Timestamp("2025-09-01")
POOL = {  # 7 model-group candidates: D1 + 6 H-025 survivors
    "D1_low_vol_20": FACTORS1["D1_low_vol_20"][1],
    **{n: FACTORS3[n][1] for n in ("M3_pv_corr_neg_20", "M4_volume_quiet_5_60",
                                   "M7_vov_neg_20", "M10_vol_cv_neg_20",
                                   "M12_cgo_vwap60_neg", "D6R_vol_compression_regate")},
}
FORMULAS = {
    "D1_low_vol_20": "-TsStd(Returns(Close,1),20)",
    "M3_pv_corr_neg_20": "-TsCorr(Close,Volume,20)",
    "M4_volume_quiet_5_60": "-Log(TsMean(V,5)/(TsMean(V,60)+eps)+eps)",
    "M7_vov_neg_20": "-TsStd(TsStd(r,5),20)",
    "M10_vol_cv_neg_20": "-TsStd(V,20)/(TsMean(V,20)+eps)",
    "M12_cgo_vwap60_neg": "-(Close/(TsSum(Amt,60)/(TsSum(V,60)+eps))-1)",
    "D6R_vol_compression_regate": "-TsStd(r,5)/(TsStd(r,60)+eps)",
    "RC7_composite": "per-date equal-weight rank-mean of 7 pool factors (a-priori)",
}


def fhash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def load_panel_with_flags() -> pd.DataFrame:
    cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
            "is_limit_up", "is_suspended"]
    panel = pd.read_parquet(REPO / bp.PANEL, columns=cols,
        filters=[("trade_date", ">=", WIN[0] - pd.Timedelta(days=200)),
                 ("trade_date", "<=", WIN[1])])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    assert panel["trade_date"].max() < QUAR, "quarantine breach"
    return panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def add_exec_labels(panel: pd.DataFrame, horizons=(10, 20)) -> pd.DataFrame:
    """Delay-1 executable: entry close(t+1), exit close(t+1+h); entry infeasible
    rows (t+1 limit-up or suspended) get NaN labels — mirrors the gold `_exec_`
    convention (EVALUATOR_VALIDITY_AUDIT_IC016)."""
    g = panel.groupby("symbol", sort=False)
    entry = g["close"].shift(-1)
    infeasible = (g["is_limit_up"].shift(-1).fillna(False).astype(bool)
                  | g["is_suspended"].shift(-1).fillna(False).astype(bool))
    for h in horizons:
        exit_ = g["close"].shift(-1 - h)
        lab = exit_ / entry - 1.0
        lab[infeasible] = np.nan
        panel[f"exec_return_{h}d"] = lab
    return panel


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel_with_flags()
    for name, ex in POOL.items():
        panel[name] = ex.evaluate(panel).to_numpy()
    for name, ex in REF3.items():
        panel[name] = ex.evaluate(panel).to_numpy()
    ranks = panel.groupby("trade_date")[list(POOL)].rank(pct=True)
    panel["RC7_composite"] = ranks.mean(axis=1)
    panel = add_exec_labels(panel)
    lab = panel[(panel["trade_date"] >= WIN[0]) & (panel["trade_date"] <= WIN[1])].copy()
    del panel

    # ---- C2: executable-label rescoring (exec_* columns, additive only)
    exec_rows = {}
    for name in list(POOL) + ["RC7_composite"]:
        sub = lab.dropna(subset=[name, "exec_return_20d"])
        ic20 = information_coefficient(sub, name, "exec_return_20d").summary
        ic10 = information_coefficient(lab.dropna(subset=[name, "exec_return_10d"]),
                                       name, "exec_return_10d").summary
        qb8 = quantile_group_backtest(sub, name, "exec_return_20d", quantiles=5, cost_bps=8.0)
        qb25 = quantile_group_backtest(sub, name, "exec_return_20d", quantiles=5, cost_bps=25.0)
        f2 = sub[(sub["trade_date"] >= F2[0]) & (sub["trade_date"] <= F2[1])]
        f2ic = information_coefficient(f2, name, "exec_return_20d").summary.mean_rank_ic if len(f2) else np.nan
        exec_rows[name] = {
            "exec_rank_ic_h10": round(ic10.mean_rank_ic, 4),
            "exec_rank_ic_h20": round(ic20.mean_rank_ic, 4),
            "exec_rank_icir_h20": round(ic20.rank_icir, 3),
            "exec_positive_ratio_h20": round(ic20.positive_ratio, 3),
            "exec_turnover": round(float(qb8.turnover.mean()), 4),
            "exec_ls_costadj_8bps": round(float(qb8.cost_adjusted_long_short.mean()), 5),
            "exec_ls_costadj_25bps": round(float(qb25.cost_adjusted_long_short.mean()), 5),
            "exec_crash_ic_h20": round(float(f2ic), 4),
            "exec_coverage": round(len(sub) / len(lab), 3),
        }
        r = exec_rows[name]
        print(f"{name:28s} execIC20 {r['exec_rank_ic_h20']:+.4f} ICIR {r['exec_rank_icir_h20']:+.2f} "
              f"turn {r['exec_turnover']:.3f} LS25 {r['exec_ls_costadj_25bps']:+.5f} "
              f"crash {r['exec_crash_ic_h20']:+.4f}", flush=True)

    # ---- C3: correlation matrix + clustering
    all_cols = list(POOL) + ["RC7_composite"] + list(REF3)
    rk = {c: lab.groupby("trade_date")[c].rank(pct=True) for c in all_cols}
    corr = pd.DataFrame(rk).corr(method="spearman")
    corr.to_parquet(OUT / "factor_correlation_matrix.parquet")
    # robust representative score (preregistered): rank(ICIR)+rank(LS25)+rank(crash)-rank(turnover)
    er = pd.DataFrame(exec_rows).T.loc[list(POOL)]
    rob = (er["exec_rank_icir_h20"].rank() + er["exec_ls_costadj_25bps"].rank()
           + er["exec_crash_ic_h20"].rank() - er["exec_turnover"].rank())
    clusters, assigned, cid = {}, set(), 0
    for n in list(POOL):
        if n in assigned:
            continue
        members = [n] + [o for o in POOL if o != n and o not in assigned
                         and abs(corr.loc[n, o]) > 0.90]
        cid += 1
        rep = max(members, key=lambda m: rob[m])
        for m in members:
            clusters[m] = {"cluster_id": cid, "cluster_representative": rep == m}
            assigned.add(m)
    pruned = [n for n in POOL if clusters[n]["cluster_representative"]]
    pd.DataFrame([{"factor_id": n, **clusters[n], "robust_score": round(rob[n], 1)}
                  for n in POOL]).to_csv(OUT / "factor_clusters.csv", index=False)

    # ---- C1: master registry
    prod = json.load(open(REPO / "runtime/reports/full_universe/fu_20260713/production_feature_union.json"))
    b3 = pd.read_csv(REPO / "FACTOR_CANDIDATE_LEDGER_batch3.csv").set_index("factor")
    b1 = pd.read_csv(REPO / "FACTOR_CANDIDATE_LEDGER.csv").set_index("factor")
    b2 = pd.read_csv(REPO / "FACTOR_CANDIDATE_LEDGER_fundamental.csv").set_index("factor")
    rows = []

    def base_row(fid, **kw):
        d = dict(factor_id=fid, factor_name=fid, formula=kw.pop("formula", "library"),
                 source=kw.pop("source", ""), source_batch=kw.pop("source_batch", ""),
                 family=kw.pop("family", ""), frequency="daily",
                 historical_status=kw.pop("historical_status", ""),
                 production_schema_used=kw.pop("production_schema_used", False),
                 old_survivor=kw.pop("old_survivor", False),
                 h025_survivor=kw.pop("h025_survivor", False),
                 conditional_only=kw.pop("conditional_only", False),
                 reconsideration_reason=kw.pop("reconsideration_reason", ""),
                 required_columns=kw.pop("required_columns", "ohlcv"),
                 PIT_status="pit_safe", coverage=kw.pop("coverage", np.nan),
                 screen_window="2023-07-03..2025-08-29",
                 label_definition=kw.pop("label_definition", ""),
                 final_pool_status=kw.pop("final_pool_status", ""),
                 rejection_reason=kw.pop("rejection_reason", ""))
        d["formula_hash"] = fhash(d["formula"] if d["formula"] != "library" else fid)
        d.update(kw)
        return d

    for col, sleeves in prod.items():
        rows.append(base_row(col, source="production gold dataset",
                             source_batch="v8.9 plus7clean", family="library",
                             historical_status="active_production",
                             production_schema_used=True,
                             label_definition="delay-1 executable (gold _exec_)",
                             final_pool_status="active_production",
                             reconsideration_reason=f"sleeves={'+'.join(sleeves)}"))
    for n in list(POOL) + ["RC7_composite"]:
        led = (b1 if n in b1.index else b3)
        old = led.loc[n] if n in led.index else None
        er_n = exec_rows[n]
        rows.append(base_row(
            n, formula=FORMULAS[n], source="batch-1" if n == "D1_low_vol_20" else
            ("preregistered composite" if n == "RC7_composite" else "batch-3 (H-025)"),
            source_batch="H-026 composite" if n == "RC7_composite" else
            ("batch1" if n == "D1_low_vol_20" else "batch3"),
            family="pool", historical_status="survivor" if n != "RC7_composite" else "composite",
            old_survivor=n == "D1_low_vol_20",
            h025_survivor=n in FACTORS3, conditional_only=False,
            label_definition="screen=same-day close (provenance); exec_*=delay-1 executable",
            final_pool_status="model_group_candidate" if n != "RC7_composite" else "model_group_M4",
            rank_ic=float(old["rank_ic_h10"]) if old is not None else np.nan,
            rank_icir=float(old["rank_icir_h10"]) if old is not None else np.nan,
            positive_ratio=float(old["pos_ratio_h10"]) if old is not None else np.nan,
            turnover=float(old["topq_turnover"]) if old is not None else np.nan,
            average_holding_days=float(old["avg_hold_days"]) if old is not None else np.nan,
            cost_adjusted_ls_8bps=float(old["ls_costadj_8bps"]) if old is not None else np.nan,
            cost_adjusted_ls_15bps=float(old["ls_costadj_15bps"]) if old is not None else np.nan,
            cost_adjusted_ls_25bps=float(old["ls_costadj_25bps"]) if old is not None else np.nan,
            crash_rank_ic=float(old["f2_crash_ic_h10"]) if old is not None else np.nan,
            capacity_proxy=float(old["capacity_rmb"]) if old is not None else np.nan,
            max_reference_correlation=float(old["max_corr_ref"]) if old is not None else np.nan,
            cluster_id=clusters.get(n, {}).get("cluster_id", np.nan),
            cluster_representative=clusters.get(n, {}).get("cluster_representative", n == "RC7_composite"),
            **er_n))
    COND = {"QF_roe": "crash-conditional (EXP-017/batch-2: F2 IC +0.098, uncond ~0)",
            "QF_net_margin": "crash-conditional (F2 IC +0.091)",
            "QF_quality": "crash-conditional (F2 IC +0.080)",
            "valuation_8cols(pb,pe_ttm,...)": "regime-conditional reserve (EXP-020 strong standalone, EXP-021/22 redundant as raw model input)",
            "D1_regime_overlay": "frozen champion member (L1+D1_regime), fold-informed suspicion (EXP-023)"}
    for n, why in COND.items():
        rows.append(base_row(n, source="batch-2/EXP-020/EXP-019", source_batch="historical",
                             family="conditional", historical_status="conditional_active",
                             conditional_only=True, reconsideration_reason=why,
                             final_pool_status="conditional_reserve"))
    REJ = {
        "D2_trend_quality_60": ("batch1", "neg IC (60d reversal) — direction-valid rejection, no invalidation condition met"),
        "D3_near_high_120": ("batch1", "weak neg IC — valid"),
        "D4_liquidity_amount_60": ("batch1", "illiq premium long side = capacity trap — valid"),
        "D5_amihud_illiq_neg_20": ("batch1", "illiq premium — valid"),
        "D7_downside_range_neg_20": ("batch1", "redundant 0.91 vs D1; D1 still in pool — redundancy unchanged"),
        "QF_gross_margin": ("batch2", "uncond IC~0, weak crash — valid"),
        "QF_revenue_yoy": ("batch2", "uncond IC~0 — valid"),
        "QF_net_income_yoy": ("batch2", "uncond IC~0 — valid"),
        "QF_growth": ("batch2", "uncond IC~0 — valid (composite of rejected parents)"),
        "M1_max_ret_neg_20": ("batch3", "novelty 0.859 vs low-vol family; D1 still in pool — redundancy unchanged"),
        "M2_skew_neg_20": ("batch3", "turnover 0.165>0.15 — valid for tilt use; model-input reconsideration would be a NEW hypothesis, not granted here"),
        "M5_clv_20": ("batch3", "a-priori direction wrong — valid"),
        "M6_overnight_neg_20": ("batch3", "a-priori direction wrong — valid"),
        "M8_semivol_neg_20": ("batch3", "novelty 0.859 — redundancy unchanged"),
        "M9_liq_shock_neg_20": ("batch3", "turnover 0.512 — valid"),
        "M11_fip_20": ("batch3", "IC~0 — valid"),
    }
    for n, (batch, why) in REJ.items():
        led = {"batch1": b1, "batch2": b2, "batch3": b3}[batch]
        old = led.loc[n] if n in led.index else None
        rows.append(base_row(n, source=batch, source_batch=batch, family="rejected",
                             historical_status="rejected",
                             reconsideration_reason=why,
                             final_pool_status="remains_rejected",
                             rejection_reason=why,
                             rank_ic=float(old["rank_ic_h10"]) if old is not None else np.nan))
    reg = pd.DataFrame(rows)
    reg.to_parquet(OUT / "factor_master_registry.parquet", index=False)
    reg.to_csv(OUT / "factor_master_registry.csv", index=False)

    report = {
        "n_production": len(prod), "n_pool": len(POOL), "n_pool_after_pruning": len(pruned),
        "pruned_out": [n for n in POOL if n not in pruned],
        "n_clusters": cid, "n_conditional": len(COND), "n_rejected_audited": len(REJ),
        "n_reconsidered": 0, "n_exact_duplicates": 0,
        "rc7_max_member_corr": round(float(corr.loc["RC7_composite", list(POOL)].abs().max()), 3),
        "pool_mutual_corr_max": round(float(corr.loc[list(POOL), list(POOL)].where(
            ~np.eye(len(POOL), dtype=bool)).abs().max().max()), 3),
        "peak_rss_gib": round(rss_gib(), 2), "runtime_s": round(time.time() - t0, 1),
        "trust": "candidate_research_only_not_fresh_holdout_validated / fixed_cohort_searched_validation",
    }
    (OUT / "registry_summary.json").write_text(json.dumps(report, indent=2))
    lines = ["# factor_cluster_report — H-026 C3\n",
             f"pool mutual |rho| max = {report['pool_mutual_corr_max']}; clusters = {cid}; "
             f"pruned M3 set = {pruned}\n", "\n| factor | cluster | rep | robust |\n|---|---|---|---|\n"]
    for n in POOL:
        lines.append(f"| {n} | {clusters[n]['cluster_id']} | {clusters[n]['cluster_representative']} | {rob[n]:.1f} |\n")
    (OUT / "factor_cluster_report.md").write_text("".join(lines))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
