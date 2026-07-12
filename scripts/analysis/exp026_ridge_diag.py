#!/usr/bin/env python3
"""H-026 ridge diagnostic (preregistered report-only; never gates).

Ridge alpha=10 on per-date rank-pct features (fillna 0.5 = cross-sectional
median, constant => no fold leakage), every-2nd-date training subsample,
same folds/groups/labels as exp026_ablation. Checks that the LightGBM
GPU_NO_GO is not a nonlinear-model artifact.
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

from exp026_ablation import (  # noqa: E402
    DS, OUT, QUAR, LBL, DATA_END, EMBARGO_TDAYS, FOLDS, POOL7, H025_6,
    KEY_DROP, FLAG_DROP, NEW_ALL, CONST_PREFIX, build_factor_panel, daily_ic,
)
from dual_track_factor_batch import rss_gib  # noqa: E402


def main() -> int:
    from sklearn.linear_model import Ridge
    t0 = time.time()
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
    assert df["trade_date"].max() < QUAR
    df = df.dropna(subset=[LBL])
    elig = ~(df["is_st"].fillna(False).astype(bool) | df["is_suspended"].fillna(False).astype(bool)
             | df["is_limit_up"].fillna(False).astype(bool))
    med = df.groupby("trade_date")["amount_mean_20d"].transform("median")
    df = df[elig & (df["amount_mean_20d"] >= med)].copy()
    df = df.drop(columns=["is_st", "is_suspended", "is_limit_up"])
    df = df.merge(fac, on=["symbol", "trade_date"], how="left")
    del fac; gc.collect()
    df["RC7_composite"] = df.groupby("trade_date")[POOL7].rank(pct=True).mean(axis=1).astype("float32")
    feats_all = base_xs + POOL7 + ["RC7_composite"]
    # per-date rank-pct (cross-sectional op, no temporal leakage); constant-fill 0.5
    df[feats_all] = df.groupby("trade_date")[feats_all].rank(pct=True).fillna(0.5).astype("float32")
    df["y"] = df.groupby("trade_date")[LBL].rank(pct=True).astype("float32")
    print(f"frame ready {len(df):,} rows, RSS {rss_gib():.1f} GiB, {time.time()-t0:.0f}s", flush=True)

    groups = {"M0": [], "M1": ["D1_low_vol_20"], "M2": H025_6, "M3": POOL7, "M4": ["RC7_composite"]}
    dates_all = np.array(sorted(df["trade_date"].unique()))
    out = {}
    for fi, (tr_end, te_end) in enumerate(FOLDS, 1):
        tr_end, te_end = pd.Timestamp(tr_end), pd.Timestamp(te_end)
        te_start = dates_all[np.searchsorted(dates_all, tr_end, side="right") + EMBARGO_TDAYS]
        tr_dates = [d for d in dates_all if d <= tr_end][::2]  # every-2nd-date subsample
        tr = df[df["trade_date"].isin(tr_dates)]
        te = df[(df["trade_date"] >= te_start) & (df["trade_date"] <= te_end)]
        ics = {}
        for gname, extra in groups.items():
            feats = base_xs + extra
            m = Ridge(alpha=10.0)
            m.fit(tr[feats], tr["y"])
            ic = daily_ic(m.predict(te[feats]), te[LBL].to_numpy(), te["trade_date"].to_numpy())
            ics[gname] = ic
            print(f"F{fi} {gname} ridge IC20 {ic.mean():+.5f}", flush=True)
        for g in ("M1", "M2", "M3", "M4"):
            a = pd.concat({"g": ics[g], "b": ics["M0"]}, axis=1).dropna()
            out.setdefault(g, []).append(round(float((a["g"] - a["b"]).mean()), 5))
        del tr, te; gc.collect()
    summary = {"delta_ic_by_fold_vs_M0": out,
               "median_delta_ic": {g: round(float(np.median(v)), 5) for g, v in out.items()},
               "role": "diagnostic only (preregistered), never gates",
               "peak_rss_gib": round(rss_gib(), 2), "runtime_s": round(time.time() - t0, 1)}
    (OUT / "ridge_diagnostic.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
