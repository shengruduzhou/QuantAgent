#!/usr/bin/env python3
"""EXP-022 / H-022: valuation incremental IC on a TRADABLE universe with a
cross-sectional-only base — controls H-021's two confounds (per-date-constant
gating features + untradable-microcap phantom breadth).

Pre-registered (HYPOTHESIS_REGISTRY.md H-022, N=2):
  base_xs        cross-sectional features only (alpha*/gtja*/per-stock px-vol),
                 per-date-CONSTANT idx_*/macro_*/flow_* DROPPED
  base_xs + val  + 8 valuation columns

Tradable universe (per date): eligible (~is_st & ~is_suspended & ~is_limit_up)
AND amount_mean_20d >= that date's cross-sectional median (liquid half).
Split/params identical to H-021. Metrics: OOS rank-IC + top-decile long 60d
return + long-short decile spread (capacity-aware proxy).
"""
from __future__ import annotations

import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean_fund.parquet"
OUT = REPO / "runtime/reports/v89_closed_loop/gbm_val_tradable_ablation"
QUARANTINE = pd.Timestamp("2025-09-01")
LABEL = "forward_return_60d"
VAL_COLS = ["pb", "pe_ttm", "earnings_yield", "valuation_percentile", "pb_own_pctile_2y",
            "pcf", "ocf_yield", "book_yield"]
# per-date-constant prefixes to EXCLUDE from a cross-sectional base
CONST_PREFIX = ("idx_", "macro_", "flow_")
KEY_DROP = {"symbol", "trade_date", "available_at", "source", "source_type",
            "source_reliability", "point_in_time_valid"}
FLAG_DROP = {"is_st", "is_suspended", "is_limit_up", "is_limit_down",
             "is_st_provenance", "missing_fundamentals", "missing_valuation", "missing_disclosures"}
NEW_ALL = set(VAL_COLS + ["pcf", "ocf_yield", "book_yield", "eps_ttm", "ocfps_ttm",
                          "roe", "roe_diluted", "net_margin", "gross_margin", "revenue_yoy",
                          "net_income_yoy", "debt_to_asset", "inventory_turnover",
                          "operating_cash_to_revenue", "quality_composite", "growth_composite"])
TRAIN_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-04-01")
TEST_END = pd.Timestamp("2025-08-29")
N_VALID_DATES = 40


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def daily_rank_ic(pred, actual, dates) -> pd.Series:
    df = pd.DataFrame({"p": pred, "a": actual, "d": dates}).dropna()
    return df.groupby("d", group_keys=False).apply(
        lambda g: g["p"].rank().corr(g["a"].rank()) if len(g) >= 30 else np.nan).dropna()


def decile_returns(pred, actual, dates):
    """Per date: top-decile long mean 60d return + long-short decile spread."""
    df = pd.DataFrame({"p": pred, "a": actual, "d": dates}).dropna()
    def _day(g):
        if len(g) < 50:
            return pd.Series({"long": np.nan, "ls": np.nan})
        q = g["p"].rank(pct=True)
        top = g.loc[q >= 0.9, "a"].mean()
        bot = g.loc[q <= 0.1, "a"].mean()
        return pd.Series({"long": top, "ls": top - bot})
    d = df.groupby("d", group_keys=False).apply(_day).dropna()
    return d


def main() -> int:
    import lightgbm as lgb
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    schema = pq.ParquetFile(DS).schema_arrow
    all_cols = list(schema.names)
    numeric = {schema.field(i).name for i in range(len(schema))
               if pa.types.is_floating(schema.field(i).type) or pa.types.is_integer(schema.field(i).type)}
    label_drop = {c for c in all_cols if c.startswith("forward_return")}
    base_xs = [c for c in all_cols if c in numeric and c not in KEY_DROP and c not in label_drop
               and c not in FLAG_DROP and c not in NEW_ALL
               and not c.startswith(CONST_PREFIX)]
    val = [c for c in VAL_COLS if c in all_cols]
    print(f"base_xs (cross-sectional only): {len(base_xs)} | val: {len(val)}", flush=True)

    read_cols = ["symbol", "trade_date", LABEL, "is_st", "is_suspended", "is_limit_up",
                 "amount_mean_20d"] + base_xs + val
    read_cols = list(dict.fromkeys(read_cols))
    df = pd.read_parquet(DS, columns=read_cols, filters=[("trade_date", "<=", TEST_END)])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    assert df["trade_date"].max() < QUARANTINE
    df = df.dropna(subset=[LABEL])
    # tradable filter: eligible + liquid half
    elig = ~(df["is_st"].fillna(False).astype(bool) | df["is_suspended"].fillna(False).astype(bool)
             | df["is_limit_up"].fillna(False).astype(bool))
    med = df.groupby("trade_date")["amount_mean_20d"].transform("median")
    liquid = df["amount_mean_20d"] >= med
    df = df[elig & liquid].copy()
    df = df.drop(columns=["is_st", "is_suspended", "is_limit_up"])
    print(f"tradable rows: {len(df):,} ({df['trade_date'].nunique()} dates)", flush=True)
    fcols = [c for c in df.columns if c not in ("symbol", "trade_date")]
    df[fcols] = df[fcols].astype("float32")
    df["y"] = df.groupby("trade_date")[LABEL].rank(pct=True).astype("float32")

    tr = df[df["trade_date"] <= TRAIN_END]
    te = df[(df["trade_date"] >= TEST_START) & (df["trade_date"] <= TEST_END)].copy()
    valid_dates = sorted(tr["trade_date"].unique())[-N_VALID_DATES:]
    vmask = tr["trade_date"].isin(valid_dates)
    y_actual = te[LABEL].to_numpy(); dates_te = te["trade_date"].to_numpy()
    print(f"train {len(tr):,} (valid {int(vmask.sum()):,}) | test {len(te):,} "
          f"dates {te['trade_date'].nunique()}", flush=True)

    params = dict(objective="regression", n_estimators=600, learning_rate=0.03,
                  num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  min_child_samples=200, reg_lambda=1.0, n_jobs=8, verbosity=-1)
    variants = {"A_base_xs": base_xs, "B_base_xs_val": base_xs + val}
    results = {}
    for name, feats in variants.items():
        model = lgb.LGBMRegressor(**params)
        model.fit(tr.loc[~vmask, feats], tr.loc[~vmask, "y"],
                  eval_set=[(tr.loc[vmask, feats], tr.loc[vmask, "y"])],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
        pred = model.predict(te[feats])
        ic = daily_rank_ic(pred, y_actual, dates_te)
        dec = decile_returns(pred, y_actual, dates_te)
        m, sd = float(ic.mean()), float(ic.std(ddof=1))
        lm, lsd = float(dec["long"].mean()), float(dec["long"].std(ddof=1))
        lsm, lssd = float(dec["ls"].mean()), float(dec["ls"].std(ddof=1))
        n = len(dec)
        imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
        results[name] = {
            "n_features": len(feats), "oos_mean_ic": round(m, 5),
            "oos_icir": round(m / sd, 4) if sd else None, "oos_t": round(m / (sd/np.sqrt(len(ic))), 2),
            "topdecile_long_60d": round(lm, 5), "topdecile_long_t": round(lm/(lsd/np.sqrt(n)), 2),
            "longshort_decile_60d": round(lsm, 5), "longshort_t": round(lsm/(lssd/np.sqrt(n)), 2),
            "val_cols_in_top15": [c for c in imp.head(15).index if c in NEW_ALL],
            "top15": list(imp.head(15).index)}
        print(f"{name:16s} feats {len(feats):3d} IC {m:+.5f} ICIR {results[name]['oos_icir']:+.3f} "
              f"| topDlong {lm:+.4f}(t{results[name]['topdecile_long_t']:+.1f}) "
              f"LS {lsm:+.4f}(t{results[name]['longshort_t']:+.1f}) | val@top15 {results[name]['val_cols_in_top15']}",
              flush=True)
        del model; gc.collect()

    dic = results["B_base_xs_val"]["oos_mean_ic"] - results["A_base_xs"]["oos_mean_ic"]
    dlong = results["B_base_xs_val"]["topdecile_long_60d"] - results["A_base_xs"]["topdecile_long_60d"]
    verdict = {"delta_ic": round(dic, 5), "delta_topdecile_long": round(dlong, 5),
               "passes_ic_+0.005": dic >= 0.005,
               "val_in_top15": bool(results["B_base_xs_val"]["val_cols_in_top15"]),
               "gpu_h023_go": dic >= 0.005 or (dlong > 0 and results["B_base_xs_val"]["topdecile_long_t"] > 2
                                               and dlong > 0.001)}
    summary = {"universe": "eligible + liquid-half (amount_mean_20d>=daily median)",
               "results": results, "verdict": verdict,
               "peak_rss_gib": round(rss_gib(), 2), "runtime_sec": round(time.time() - t0, 1)}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    print("\n=== VERDICT ==="); print(json.dumps(verdict, indent=2))
    print(f"peak RSS {summary['peak_rss_gib']} GiB, {summary['runtime_sec']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
