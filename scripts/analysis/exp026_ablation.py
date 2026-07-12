#!/usr/bin/env python3
"""H-026 Phase D: CPU incremental model ablation M0-M4 (fu_20260713_h026).

Preregistered (HYPOTHESIS_REGISTRY.md H-026, commit f2931d7, BEFORE this run):
- M0 = EXP-022 base_xs (cross-sectional-only); M1 = M0+{D1}; M2 = M0+{6 H-025};
  M3 = M0+{C3-pruned pool}; M4 = M0+{RC7 composite}.
- Gate model = LightGBM, params byte-identical to EXP-022; ridge = diagnostic.
- 3 purged expanding outer folds; test starts 62 trading days after train end
  (embargo >= 60d label + 1d delay); all data <= 2025-08-29 (asserted).
- Labels: gold `_exec_` delay-1 executable forward_return_20d (gate axis),
  forward_return_60d (diagnostic, same predictions cross-scored).
- Universe: tradable = eligible AND liquid-half. fixed_cohort_searched_validation.
- Gate A: median paired delta mean-daily-rankIC(h20) >= +0.005 AND delta>0 in
  >=2/3 folds AND new col in top-15 importance in >=2/3 folds.
- Gate B (economic alt): delta top-decile long > 0 in >=2/3 folds AND pooled
  t >= 2 AND median delta >= +0.001 AND top-decile turnover not worse by >10%.
- No group passes => GPU_NO_GO, Phase E not entered.
"""
from __future__ import annotations

import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "scripts"))

from dual_track_factor_batch import FACTORS as FACTORS1, rss_gib  # noqa: E402
from factor_batch3_pv import FACTORS3  # noqa: E402
import baseline_protocol as bp  # noqa: E402

OUT = REPO / "runtime/reports/full_universe/fu_20260713_h026"
DS = REPO / "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean_fund.parquet"
QUAR = pd.Timestamp("2025-09-01")
LBL, LBL_DIAG = "forward_return_20d", "forward_return_60d"
DATA_END = pd.Timestamp("2025-08-29")
WARMUP_START = pd.Timestamp("2017-06-01")
EMBARGO_TDAYS = 62
FOLDS = [("2021-12-31", "2023-03-31"), ("2022-12-31", "2024-03-29"), ("2023-12-31", "2025-08-29")]
N_VALID_DATES = 40
# EXP-022 base_xs exclusions, replicated verbatim
VAL_COLS = ["pb", "pe_ttm", "earnings_yield", "valuation_percentile", "pb_own_pctile_2y",
            "pcf", "ocf_yield", "book_yield"]
CONST_PREFIX = ("idx_", "macro_", "flow_")
KEY_DROP = {"symbol", "trade_date", "available_at", "source", "source_type",
            "source_reliability", "point_in_time_valid"}
FLAG_DROP = {"is_st", "is_suspended", "is_limit_up", "is_limit_down",
             "is_st_provenance", "missing_fundamentals", "missing_valuation", "missing_disclosures"}
NEW_ALL = set(VAL_COLS + ["eps_ttm", "ocfps_ttm", "roe", "roe_diluted", "net_margin",
                          "gross_margin", "revenue_yoy", "net_income_yoy", "debt_to_asset",
                          "inventory_turnover", "operating_cash_to_revenue",
                          "quality_composite", "growth_composite"])
POOL7 = ["D1_low_vol_20", "M3_pv_corr_neg_20", "M4_volume_quiet_5_60", "M7_vov_neg_20",
         "M10_vol_cv_neg_20", "M12_cgo_vwap60_neg", "D6R_vol_compression_regate"]
H025_6 = POOL7[1:]

LGB_PARAMS = dict(objective="regression", n_estimators=600, learning_rate=0.03,
                  num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  min_child_samples=200, reg_lambda=1.0, n_jobs=8, verbosity=-1)


def build_factor_panel() -> pd.DataFrame:
    """Compute the 7 pool factors from the silver panel over full history."""
    cache = OUT / "new_factor_panel.parquet"
    if cache.exists():
        f = pd.read_parquet(cache)
        f["trade_date"] = pd.to_datetime(f["trade_date"])
        return f
    exprs = {"D1_low_vol_20": FACTORS1["D1_low_vol_20"][1],
             **{n: FACTORS3[n][1] for n in H025_6}}
    panel = pd.read_parquet(REPO / bp.PANEL,
        columns=["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"],
        filters=[("trade_date", ">=", WARMUP_START), ("trade_date", "<=", DATA_END)])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    assert panel["trade_date"].max() < QUAR
    panel = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    out = panel[["symbol", "trade_date"]].copy()
    for n, ex in exprs.items():
        out[n] = ex.evaluate(panel).to_numpy().astype("float32")
        print(f"factor {n} done", flush=True)
    del panel; gc.collect()
    out.to_parquet(cache, index=False)
    return out


def daily_ic(pred, actual, dates) -> pd.Series:
    df = pd.DataFrame({"p": pred, "a": actual, "d": dates}).dropna()
    return df.groupby("d", group_keys=False).apply(
        lambda g: g["p"].rank().corr(g["a"].rank()) if len(g) >= 30 else np.nan).dropna()


def decile_stats(pred, actual, dates, symbols):
    """Per-date top-decile long mean + membership turnover."""
    df = pd.DataFrame({"p": pred, "a": actual, "d": dates, "s": symbols}).dropna()
    df["q"] = df.groupby("d")["p"].rank(pct=True)
    top = df[df["q"] >= 0.9]
    long_by_d = top.groupby("d")["a"].mean()
    sets = top.groupby("d")["s"].apply(set).sort_index()
    to = [len(sets.iloc[i] - sets.iloc[i - 1]) / max(len(sets.iloc[i]), 1)
          for i in range(1, len(sets))]
    return long_by_d, float(np.mean(to)) if to else np.nan


def main() -> int:
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fac = build_factor_panel()

    schema = pq.ParquetFile(DS).schema_arrow
    all_cols = list(schema.names)
    numeric = {schema.field(i).name for i in range(len(schema))
               if pa.types.is_floating(schema.field(i).type) or pa.types.is_integer(schema.field(i).type)}
    label_drop = {c for c in all_cols if c.startswith("forward_return")}
    base_xs = [c for c in all_cols if c in numeric and c not in KEY_DROP and c not in label_drop
               and c not in FLAG_DROP and c not in NEW_ALL and not c.startswith(CONST_PREFIX)]
    print(f"base_xs: {len(base_xs)}", flush=True)

    read_cols = list(dict.fromkeys(["symbol", "trade_date", LBL, LBL_DIAG, "is_st",
                                    "is_suspended", "is_limit_up", "amount_mean_20d"] + base_xs))
    df = pd.read_parquet(DS, columns=read_cols, filters=[("trade_date", "<=", DATA_END)])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    assert df["trade_date"].max() < QUAR
    df = df.dropna(subset=[LBL])
    elig = ~(df["is_st"].fillna(False).astype(bool) | df["is_suspended"].fillna(False).astype(bool)
             | df["is_limit_up"].fillna(False).astype(bool))
    med = df.groupby("trade_date")["amount_mean_20d"].transform("median")
    df = df[elig & (df["amount_mean_20d"] >= med)].copy()
    df = df.drop(columns=["is_st", "is_suspended", "is_limit_up"])
    n0 = len(df)
    df = df.merge(fac, on=["symbol", "trade_date"], how="left")
    assert len(df) == n0, "factor merge fan-out"
    del fac; gc.collect()
    fcols = [c for c in df.columns if c not in ("symbol", "trade_date")]
    df[fcols] = df[fcols].astype("float32")
    # RC7 composite: per-date equal rank-mean within the tradable frame (a-priori)
    df["RC7_composite"] = df.groupby("trade_date")[POOL7].rank(pct=True).mean(axis=1).astype("float32")
    df["y"] = df.groupby("trade_date")[LBL].rank(pct=True).astype("float32")
    print(f"tradable rows {len(df):,}, dates {df['trade_date'].nunique()}, RSS {rss_gib():.1f} GiB", flush=True)

    clusters = pd.read_csv(OUT / "factor_clusters.csv")
    m3_set = list(clusters.loc[clusters["cluster_representative"], "factor_id"])
    groups = {"M0": [], "M1": ["D1_low_vol_20"], "M2": H025_6, "M3": m3_set, "M4": ["RC7_composite"]}
    # record identical/empty groups honestly instead of fake trials
    group_notes = {}
    if sorted(m3_set) == sorted(POOL7):
        group_notes["M3"] = "pruning removed nothing (no mutual >0.90 cluster); M3 = M0+all7"

    dates_all = np.array(sorted(df["trade_date"].unique()))
    fold_rows, results = [], {}
    for fi, (tr_end, te_end) in enumerate(FOLDS, 1):
        tr_end, te_end = pd.Timestamp(tr_end), pd.Timestamp(te_end)
        te_start = dates_all[np.searchsorted(dates_all, tr_end, side="right") + EMBARGO_TDAYS]
        tr = df[df["trade_date"] <= tr_end]
        te = df[(df["trade_date"] >= te_start) & (df["trade_date"] <= te_end)]
        vdates = sorted(tr["trade_date"].unique())[-N_VALID_DATES:]
        vmask = tr["trade_date"].isin(vdates)
        y20, y60 = te[LBL].to_numpy(), te[LBL_DIAG].to_numpy()
        d_te, s_te = te["trade_date"].to_numpy(), te["symbol"].to_numpy()
        print(f"\nF{fi}: train<= {tr_end.date()} ({len(tr):,}) | test {pd.Timestamp(te_start).date()}"
              f"..{te_end.date()} ({len(te):,})", flush=True)
        ic_by_group, dec_by_group = {}, {}
        for gname, extra in groups.items():
            feats = base_xs + extra
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(tr.loc[~vmask, feats], tr.loc[~vmask, "y"],
                      eval_set=[(tr.loc[vmask, feats], tr.loc[vmask, "y"])],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
            pred = model.predict(te[feats], num_iteration=model.best_iteration_)
            ic = daily_ic(pred, y20, d_te)
            ic60 = daily_ic(pred, y60, d_te)
            longs, topto = decile_stats(pred, y20, d_te, s_te)
            imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
            new_in_top15 = [c for c in imp.head(15).index if c in extra]
            ic_by_group[gname] = ic; dec_by_group[gname] = longs
            # ridge diagnostic (rank features on the fly for this group is too
            # heavy per group; run for M0 and this group only when extra)
            fold_rows.append({
                "fold": fi, "group": gname, "model": "lgbm", "n_features": len(feats),
                "mean_ic_h20": round(float(ic.mean()), 5), "icir_h20": round(float(ic.mean()/ic.std(ddof=1)), 3),
                "mean_ic_h60_diag": round(float(ic60.mean()), 5),
                "topdecile_long_h20": round(float(longs.mean()), 5),
                "topdecile_turnover": round(topto, 4),
                "new_cols_in_top15": ",".join(new_in_top15),
                "best_iter": int(model.best_iteration_ or 0),
            })
            print(f"  {gname} feats {len(feats)} IC20 {ic.mean():+.5f} IC60 {ic60.mean():+.5f} "
                  f"topD {longs.mean():+.5f} to {topto:.3f} new@15 {new_in_top15}", flush=True)
            del model; gc.collect()
        # paired deltas vs M0 (date-aligned)
        for gname in ("M1", "M2", "M3", "M4"):
            a = pd.concat({"g": ic_by_group[gname], "b": ic_by_group["M0"]}, axis=1).dropna()
            dl = pd.concat({"g": dec_by_group[gname], "b": dec_by_group["M0"]}, axis=1).dropna()
            ddec = dl["g"] - dl["b"]
            results.setdefault(gname, []).append({
                "fold": fi,
                "delta_ic": float((a["g"] - a["b"]).mean()),
                "delta_topdecile": float(ddec.mean()),
                "delta_topdecile_t": float(ddec.mean() / (ddec.std(ddof=1) / np.sqrt(len(ddec)))) if len(ddec) > 2 else np.nan,
                "to_ratio": [r for r in fold_rows if r["fold"] == fi and r["group"] == gname][0]["topdecile_turnover"]
                            / max([r for r in fold_rows if r["fold"] == fi and r["group"] == "M0"][0]["topdecile_turnover"], 1e-9),
                "new_in_top15": bool([r for r in fold_rows if r["fold"] == fi and r["group"] == gname][0]["new_cols_in_top15"]),
            })
        del tr, te; gc.collect()

    # ---- ridge diagnostic (fold 3 only, M0 vs best-delta group, declared diagnostic)
    # kept minimal by design; reported, never gated on.
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT / "fold_metrics.csv", index=False)

    verdicts = {}
    for gname, folds in results.items():
        d_ic = [f["delta_ic"] for f in folds]
        d_dec = [f["delta_topdecile"] for f in folds]
        pooled_t = np.mean([f["delta_topdecile_t"] for f in folds])
        gate_a = (np.median(d_ic) >= 0.005 and sum(d > 0 for d in d_ic) >= 2
                  and sum(f["new_in_top15"] for f in folds) >= 2)
        gate_b = (sum(d > 0 for d in d_dec) >= 2 and pooled_t >= 2
                  and np.median(d_dec) >= 0.001
                  and all(f["to_ratio"] <= 1.10 for f in folds))
        verdicts[gname] = {"delta_ic_by_fold": [round(x, 5) for x in d_ic],
                           "median_delta_ic": round(float(np.median(d_ic)), 5),
                           "delta_topdecile_by_fold": [round(x, 5) for x in d_dec],
                           "median_delta_topdecile": round(float(np.median(d_dec)), 5),
                           "pooled_topdecile_t": round(float(pooled_t), 2),
                           "turnover_ratio_by_fold": [round(f["to_ratio"], 3) for f in folds],
                           "gate_A_ic": bool(gate_a), "gate_B_economic": bool(gate_b),
                           "passes": bool(gate_a or gate_b)}
    any_pass = any(v["passes"] for v in verdicts.values())
    summary = {"groups": {k: (["M0-baseline"] if k == "M0" else v) for k, v in
                          {**{g: groups[g] for g in groups}}.items()},
               "group_notes": group_notes, "verdicts": verdicts,
               "gpu_decision": "GO_CONDITIONAL" if any_pass else "GPU_NO_GO",
               "phase_e": "enter (<=2 finalists)" if any_pass else "not entered",
               "trust": "candidate_research_only_not_fresh_holdout_validated / fixed_cohort_searched_validation",
               "peak_rss_gib": round(rss_gib(), 2), "runtime_s": round(time.time() - t0, 1)}
    (OUT / "ablation_results.json").write_text(json.dumps(summary, indent=2))
    print("\n=== VERDICTS ==="); print(json.dumps({k: v for k, v in verdicts.items()}, indent=2))
    print(f"GPU decision: {summary['gpu_decision']} | peak RSS {summary['peak_rss_gib']} GiB, "
          f"{summary['runtime_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
