#!/usr/bin/env python3
"""H-027 Stage 1 (CPU): B0 baseline + cross-fitted linear residual sleeves +
learning-to-rank tree candidates (fu_20260713_h027).

Preregistered in HYPOTHESIS_REGISTRY.md H-027 (commit 4ca88e2) BEFORE this run.
Candidates here: B0 (baseline, OOF persisted), L1/L2 x lambda {0.1,0.2,0.3},
X0/X1 (XGB rank:ndcg), C1 (CatBoost LambdaMart), LG1 (LGBM rank_xendcg).
Per-candidate failures are caught, categorized and recorded — never silently
substituted. All data <= 2025-08-29 (asserted).
"""
from __future__ import annotations

import gc
import json
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "scripts"))

from dual_track_factor_batch import rss_gib  # noqa: E402
from exp026_ablation import (  # noqa: E402
    DS, QUAR, LBL, DATA_END, EMBARGO_TDAYS, FOLDS, POOL7, N_VALID_DATES,
    KEY_DROP, FLAG_DROP, NEW_ALL, CONST_PREFIX, LGB_PARAMS, build_factor_panel,
)

OUT = REPO / "runtime/reports/full_universe/fu_20260713_h027"
OOF = OUT / "oof"
LAMBDAS = (0.10, 0.20, 0.30)
XGB_PARAMS = dict(objective="rank:ndcg", tree_method="hist", learning_rate=0.05,
                  max_depth=8, min_child_weight=200, subsample=0.8,
                  colsample_bytree=0.7, n_estimators=600,
                  lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=8,
                  eval_metric="ndcg@100", early_stopping_rounds=40, n_jobs=16,
                  verbosity=0)
CB_PARAMS = dict(loss_function="LambdaMart", iterations=600, learning_rate=0.05,
                 depth=8, eval_metric="NDCG:top=100", thread_count=16,
                 verbose=False, allow_writing_files=False, random_seed=0)


def load_frame():
    """EXP-026-identical frame: base_xs raw + 7 pool factors + RC7 + labels."""
    fac = build_factor_panel()
    schema = pq.ParquetFile(DS).schema_arrow
    all_cols = list(schema.names)
    numeric = {schema.field(i).name for i in range(len(schema))
               if pa.types.is_floating(schema.field(i).type) or pa.types.is_integer(schema.field(i).type)}
    label_drop = {c for c in all_cols if c.startswith("forward_return")}
    base_xs = [c for c in all_cols if c in numeric and c not in KEY_DROP and c not in label_drop
               and c not in FLAG_DROP and c not in NEW_ALL and not c.startswith(CONST_PREFIX)]
    read_cols = list(dict.fromkeys(["symbol", "trade_date", LBL, "is_st", "is_suspended",
                                    "is_limit_up", "amount_mean_20d"] + base_xs))
    df = pd.read_parquet(DS, columns=read_cols, filters=[("trade_date", "<=", DATA_END)])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    assert df["trade_date"].max() < QUAR, "quarantine breach"
    df = df.dropna(subset=[LBL])
    elig = ~(df["is_st"].fillna(False).astype(bool) | df["is_suspended"].fillna(False).astype(bool)
             | df["is_limit_up"].fillna(False).astype(bool))
    med = df.groupby("trade_date")["amount_mean_20d"].transform("median")
    df = df[elig & (df["amount_mean_20d"] >= med)].copy()
    df = df.drop(columns=["is_st", "is_suspended", "is_limit_up"])
    n0 = len(df)
    df = df.merge(fac, on=["symbol", "trade_date"], how="left")
    assert len(df) == n0
    del fac; gc.collect()
    fcols = [c for c in df.columns if c not in ("symbol", "trade_date")]
    df[fcols] = df[fcols].astype("float32")
    df["RC7_composite"] = df.groupby("trade_date")[POOL7].rank(pct=True).mean(axis=1).astype("float32")
    df["y"] = df.groupby("trade_date")[LBL].rank(pct=True).astype("float32")
    df["grade"] = np.minimum((df["y"] * 5).astype("int32"), 4)  # 5 daily quantile grades 0..4
    df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return df, base_xs


def fold_masks(df):
    dates_all = np.array(sorted(df["trade_date"].unique()))
    out = []
    for fi, (tr_end, te_end) in enumerate(FOLDS, 1):
        tr_end, te_end = pd.Timestamp(tr_end), pd.Timestamp(te_end)
        te_start = dates_all[np.searchsorted(dates_all, tr_end, side="right") + EMBARGO_TDAYS]
        trm = df["trade_date"] <= tr_end
        vdates = sorted(df.loc[trm, "trade_date"].unique())[-N_VALID_DATES:]
        vm = trm & df["trade_date"].isin(vdates)
        tem = (df["trade_date"] >= te_start) & (df["trade_date"] <= te_end)
        out.append((fi, trm, vm, tem))
    return out


def daily_ic(pred, actual, dates):
    d = pd.DataFrame({"p": pred, "a": actual, "d": dates}).dropna()
    return d.groupby("d", group_keys=False)[["p", "a"]].apply(
        lambda g: g["p"].rank().corr(g["a"].rank()) if len(g) >= 30 else np.nan).dropna()


def decile_stats(pred, actual, dates, symbols):
    d = pd.DataFrame({"p": pred, "a": actual, "d": dates, "s": symbols}).dropna()
    d["q"] = d.groupby("d")["p"].rank(pct=True)
    top = d[d["q"] >= 0.9]
    long_mean = float(top.groupby("d")["a"].mean().mean())
    sets = top.groupby("d")["s"].apply(set).sort_index()
    to = [len(sets.iloc[i] - sets.iloc[i - 1]) / max(len(sets.iloc[i]), 1) for i in range(1, len(sets))]
    return long_mean, float(np.mean(to)) if to else np.nan


def record(store, cand, fi, pred, te, note=""):
    y20 = te[LBL].to_numpy()
    ic = daily_ic(pred, y20, te["trade_date"].to_numpy())
    tl, to = decile_stats(pred, y20, te["trade_date"].to_numpy(), te["symbol"].to_numpy())
    store["ics"].append(pd.DataFrame({"candidate": cand, "fold": fi,
                                      "d": ic.index, "ic": ic.values}))
    store["rows"].append({"candidate": cand, "fold": fi, "mean_ic": round(float(ic.mean()), 5),
                          "topdecile_long_h20": round(tl, 5), "topdecile_turnover": round(to, 4),
                          "note": note})
    store["oof"].append(pd.DataFrame({"candidate": cand, "fold": fi,
                                      "trade_date": te["trade_date"].to_numpy(),
                                      "symbol": te["symbol"].to_numpy(),
                                      "pred": np.asarray(pred, dtype="float32")}))
    print(f"  {cand:10s} F{fi} IC {ic.mean():+.5f} topD {tl:+.5f} to {to:.3f} {note}", flush=True)


def main() -> int:
    import lightgbm as lgb
    from sklearn.linear_model import ElasticNet, Ridge
    import xgboost as xgbm
    from catboost import CatBoostRanker, Pool as CBPool
    t0 = time.time()
    OOF.mkdir(parents=True, exist_ok=True)
    df, base_xs = load_frame()
    m3_feats = base_xs + POOL7
    pool_rank = df.groupby("trade_date")[POOL7].rank(pct=True).fillna(0.5).astype("float32")
    pool_rank.columns = [f"rk_{c}" for c in POOL7]
    df = pd.concat([df, pool_rank], axis=1)
    rk7 = [f"rk_{c}" for c in POOL7]
    del pool_rank; gc.collect()
    print(f"frame {len(df):,} rows, RSS {rss_gib():.1f} GiB, {time.time()-t0:.0f}s", flush=True)

    store = {"ics": [], "rows": [], "oof": []}
    failures = {}
    for fi, trm, vm, tem in fold_masks(df):
        tr, te = df[trm], df[tem]
        trn = df[trm & ~vm]; val = df[vm]
        print(f"\nF{fi}: train {len(tr):,} test {len(te):,}", flush=True)

        # ---- B0 baseline (EXP-026 params, OOF persisted this time)
        b0 = lgb.LGBMRegressor(**LGB_PARAMS)
        b0.fit(trn[base_xs], trn["y"], eval_set=[(val[base_xs], val["y"])],
               callbacks=[lgb.early_stopping(40, verbose=False)])
        b0_pred = b0.predict(te[base_xs], num_iteration=b0.best_iteration_)
        record(store, "B0", fi, b0_pred, te)
        b0_rank_te = pd.Series(b0_pred).groupby(te["trade_date"].to_numpy()).rank(pct=True).to_numpy()

        # ---- cross-fitted baseline inside outer-train (3 chronological blocks)
        try:
            tr_dates = np.array(sorted(tr["trade_date"].unique()))
            blocks = np.array_split(tr_dates, 3)
            pred_cf = pd.Series(np.nan, index=tr.index, dtype="float64")
            for b in range(3):
                bm = tr["trade_date"].isin(blocks[b])
                om = ~bm
                otr = tr[om]
                ovd = sorted(otr["trade_date"].unique())[-N_VALID_DATES:]
                ovm = otr["trade_date"].isin(ovd)
                m = lgb.LGBMRegressor(**LGB_PARAMS)
                m.fit(otr.loc[~ovm, base_xs], otr.loc[~ovm, "y"],
                      eval_set=[(otr.loc[ovm, base_xs], otr.loc[ovm, "y"])],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
                pred_cf.loc[tr.index[bm]] = m.predict(tr.loc[bm, base_xs],
                                                      num_iteration=m.best_iteration_)
                del m; gc.collect()
            cf_rank = pred_cf.groupby(tr["trade_date"].to_numpy()).rank(pct=True)
            resid = (tr["y"].to_numpy() - cf_rank.to_numpy()).astype("float32")
            for cand, mk in (("L1", lambda: Ridge(alpha=10.0)),
                             ("L2", lambda: ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=2000))):
                rm = mk()
                rm.fit(tr[rk7], resid)
                rp_te = rm.predict(te[rk7])
                for lam in LAMBDAS:
                    record(store, f"{cand}_{lam:.2f}", fi, b0_rank_te + lam * rp_te, te,
                           note=f"resid_ic={daily_ic(rp_te, te[LBL].to_numpy(), te['trade_date'].to_numpy()).mean():+.4f}")
        except Exception:
            failures[f"residual_F{fi}"] = traceback.format_exc()[-500:]
            print(f"  RESIDUAL FAILED F{fi}", flush=True)

        # ---- ranking trees
        qid_tr, qid_v = trn["trade_date"].factorize()[0], val["trade_date"].factorize()[0]
        for cand, feats in (("X0", base_xs), ("X1", m3_feats)):
            try:
                xr = xgbm.XGBRanker(**XGB_PARAMS)
                xr.fit(trn[feats], trn["grade"], qid=qid_tr,
                       eval_set=[(val[feats], val["grade"])], eval_qid=[qid_v],
                       verbose=False)
                record(store, cand, fi, xr.predict(te[feats]), te)
                del xr; gc.collect()
            except Exception:
                failures[f"{cand}_F{fi}"] = traceback.format_exc()[-500:]
                print(f"  {cand} FAILED F{fi}", flush=True)
        try:
            cb = CatBoostRanker(**CB_PARAMS)
            cb.fit(CBPool(trn[m3_feats], trn["grade"], group_id=qid_tr),
                   eval_set=CBPool(val[m3_feats], val["grade"], group_id=qid_v),
                   early_stopping_rounds=40)
            record(store, "C1", fi, cb.predict(CBPool(te[m3_feats],
                   group_id=te["trade_date"].factorize()[0])), te)
            del cb; gc.collect()
        except Exception:
            failures[f"C1_F{fi}"] = traceback.format_exc()[-500:]
            print(f"  C1 FAILED F{fi}", flush=True)
        try:
            lg1 = lgb.LGBMRanker(objective="rank_xendcg", n_estimators=600, learning_rate=0.03,
                                 num_leaves=63, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.7, min_child_samples=200, reg_lambda=1.0,
                                 n_jobs=16, verbosity=-1)
            gtr = trn.groupby("trade_date", sort=True).size().to_numpy()
            gv = val.groupby("trade_date", sort=True).size().to_numpy()
            lg1.fit(trn[base_xs], trn["grade"], group=gtr,
                    eval_set=[(val[base_xs], val["grade"])], eval_group=[gv],
                    eval_at=[100], callbacks=[lgb.early_stopping(40, verbose=False)])
            record(store, "LG1", fi, lg1.predict(te[base_xs],
                   num_iteration=lg1.best_iteration_), te)
            del lg1; gc.collect()
        except Exception:
            failures[f"LG1_F{fi}"] = traceback.format_exc()[-500:]
            print(f"  LG1 FAILED F{fi}", flush=True)
        del b0, tr, te, trn, val; gc.collect()

    pd.concat(store["ics"]).to_parquet(OUT / "daily_ic_stage1.parquet", index=False)
    pd.concat(store["oof"]).to_parquet(OOF / "stage1_oof.parquet", index=False)
    pd.DataFrame(store["rows"]).to_csv(OUT / "stage1_fold_metrics.csv", index=False)
    (OUT / "stage1_failures.json").write_text(json.dumps(failures, indent=2))
    print(f"\nstage1 done: {len(store['rows'])} candidate-folds, failures={len(failures)}, "
          f"peak RSS {rss_gib():.2f} GiB, {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
