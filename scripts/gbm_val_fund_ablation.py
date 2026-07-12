#!/usr/bin/env python3
"""EXP-021 / H-021: LightGBM cross-sectional ablation — does valuation/fundamental
add INCREMENTAL OOS rank-IC over the existing technical/macro feature set?

Pre-registered (HYPOTHESIS_REGISTRY.md H-021, frozen commit before run, N=4):
  A base           original plus7clean numeric features (alpha/gtja/macro/idx/flow)
  B base + val     + pb/pe_ttm/earnings_yield/valuation_percentile/pb_own_pctile_2y/pcf/ocf_yield/book_yield
  C base + fund    + roe/.../quality_composite/growth_composite
  D base + val+fund (full)

Strict pre-quarantine time split, 60-td embargo, fresh-holdout untouched.
Target = per-date cross-sectional percentile rank of forward_return_60d.
Metric = OOS mean daily Spearman rank-IC vs forward_return_60d + ICIR/t +
feature importance (are the new cols in the gain top-15?).
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean_fund.parquet"
OUT = REPO / "runtime/reports/v89_closed_loop/gbm_val_fund_ablation"
QUARANTINE = pd.Timestamp("2025-09-01")
LABEL = "forward_return_60d"

VAL_COLS = ["pb", "pe_ttm", "earnings_yield", "valuation_percentile", "pb_own_pctile_2y",
            "pcf", "ocf_yield", "book_yield"]
FUND_COLS = ["roe", "roe_diluted", "net_margin", "gross_margin", "revenue_yoy",
             "net_income_yoy", "debt_to_asset", "inventory_turnover",
             "operating_cash_to_revenue", "quality_composite", "growth_composite"]
NEW_COLS = set(VAL_COLS + FUND_COLS + ["eps_ttm", "ocfps_ttm",
                                       "missing_fundamentals", "missing_valuation"])
KEY_DROP = {"symbol", "trade_date", "available_at", "source", "source_type",
            "source_reliability", "point_in_time_valid"}
FLAG_DROP = {"is_st", "is_suspended", "is_limit_up", "is_limit_down",
             "is_st_provenance", "missing_fundamentals", "missing_valuation",
             "missing_disclosures"}

TRAIN_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-04-01")   # ~60 td embargo after train_end
TEST_END = pd.Timestamp("2025-08-29")     # pre-quarantine
N_VALID_DATES = 40                        # tail of train for early stopping


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def daily_rank_ic(pred: np.ndarray, actual: np.ndarray, dates: np.ndarray) -> pd.Series:
    df = pd.DataFrame({"p": pred, "a": actual, "d": dates}).dropna()
    def _ic(g):
        if len(g) < 30:
            return np.nan
        return g["p"].rank().corr(g["a"].rank())
    return df.groupby("d", group_keys=False).apply(_ic).dropna()


def main() -> int:
    import lightgbm as lgb
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    import pyarrow as pa
    all_cols = list(pq.ParquetFile(DS).schema_arrow.names)
    label_drop = {c for c in all_cols if c.startswith("forward_return")}
    # numeric feature universe = everything except keys / labels / flags
    schema = pq.ParquetFile(DS).schema_arrow
    numeric = {schema.field(i).name for i in range(len(schema))
               if pa.types.is_floating(schema.field(i).type) or pa.types.is_integer(schema.field(i).type)}
    feats_all = [c for c in all_cols if c in numeric and c not in KEY_DROP
                 and c not in label_drop and c not in FLAG_DROP]
    base = [c for c in feats_all if c not in NEW_COLS]
    val = [c for c in VAL_COLS if c in all_cols]
    fund = [c for c in FUND_COLS if c in all_cols]
    variants = {"A_base": base, "B_base_val": base + val,
                "C_base_fund": base + fund, "D_full": base + val + fund}
    print(f"base features: {len(base)} | val: {len(val)} | fund: {len(fund)}", flush=True)

    read_cols = ["symbol", "trade_date", LABEL] + sorted(set(base + val + fund))
    df = pd.read_parquet(DS, columns=read_cols,
                         filters=[("trade_date", "<=", TEST_END)])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    assert df["trade_date"].max() < QUARANTINE, "quarantine breach"
    df = df.dropna(subset=[LABEL])
    # downcast to float32 to bound RAM
    fcols = [c for c in df.columns if c not in ("symbol", "trade_date")]
    df[fcols] = df[fcols].astype("float32")
    # per-date percentile-rank target (cross-sectional)
    df["y"] = df.groupby("trade_date")[LABEL].rank(pct=True).astype("float32")

    tr = df[df["trade_date"] <= TRAIN_END]
    te = df[(df["trade_date"] >= TEST_START) & (df["trade_date"] <= TEST_END)]
    valid_dates = sorted(tr["trade_date"].unique())[-N_VALID_DATES:]
    vmask = tr["trade_date"].isin(valid_dates)
    print(f"train rows {len(tr):,} (fit {int((~vmask).sum()):,} / valid {int(vmask.sum()):,}) "
          f"| test rows {len(te):,} dates {te['trade_date'].nunique()}", flush=True)

    params = dict(objective="regression", n_estimators=600, learning_rate=0.03,
                  num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  min_child_samples=200, reg_lambda=1.0, n_jobs=-1, verbosity=-1)

    results = {}
    y_actual = te[LABEL].to_numpy()
    dates_te = te["trade_date"].to_numpy()
    for name, feats in variants.items():
        Xtr, ytr = tr.loc[~vmask, feats], tr.loc[~vmask, "y"]
        Xva, yva = tr.loc[vmask, feats], tr.loc[vmask, "y"]
        model = lgb.LGBMRegressor(**params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
        pred = model.predict(te[feats])
        ic = daily_rank_ic(pred, y_actual, dates_te)
        m, sd = float(ic.mean()), float(ic.std(ddof=1))
        icir = m / sd if sd > 0 else float("nan")
        t = m / (sd / np.sqrt(len(ic))) if sd > 0 else float("nan")
        imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
        top15 = list(imp.head(15).index)
        new_in_top15 = [c for c in top15 if c in NEW_COLS]
        results[name] = {"n_features": len(feats), "best_iter": int(model.best_iteration_ or params["n_estimators"]),
                         "oos_mean_ic": round(m, 5), "oos_icir": round(icir, 4),
                         "oos_t": round(t, 2), "n_test_dates": int(len(ic)),
                         "new_cols_in_top15": new_in_top15,
                         "top15_importance": top15}
        print(f"{name:14s} feats {len(feats):3d} OOS meanIC {m:+.5f} ICIR {icir:+.3f} "
              f"t {t:+.2f} | new in top15: {new_in_top15}", flush=True)

    base_ic = results["A_base"]["oos_mean_ic"]
    verdict = {}
    for name in ("B_base_val", "C_base_fund", "D_full"):
        delta = results[name]["oos_mean_ic"] - base_ic
        verdict[name] = {"delta_ic_vs_base": round(delta, 5),
                         "passes_+0.005": delta >= 0.005,
                         "new_cols_in_top15": bool(results[name]["new_cols_in_top15"])}
    gpu_go = (verdict["B_base_val"]["passes_+0.005"] or verdict["D_full"]["passes_+0.005"]) \
        and (verdict["B_base_val"]["new_cols_in_top15"] or verdict["D_full"]["new_cols_in_top15"])
    summary = {"base_oos_ic": base_ic, "results": results, "verdict": verdict,
               "gpu_h022_go": gpu_go, "peak_rss_gib": round(rss_gib(), 2),
               "runtime_sec": round(time.time() - t0, 1),
               "split": {"train_end": str(TRAIN_END.date()), "test": [str(TEST_START.date()), str(TEST_END.date())]}}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))
    print(f"GPU (H-022) go: {gpu_go}")
    print(f"peak RSS {summary['peak_rss_gib']} GiB, {summary['runtime_sec']}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
