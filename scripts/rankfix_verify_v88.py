# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): v8.8 rank-corruption forensics: mission complete, kept as evidence chain.
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""Verify the rank-fixed v88 dataset.

1. Reproducibility: recompute the 22 patched columns with the diag's method
   (trailing ~900-day window, full panel universe, current code — a different
   frame than the patch's 2018+ full-history pass, so this is not circular)
   and check per-day cross-sectional spearman >= 0.99 on several dates.
2. Integrity: non-patched columns must be identical to the original v88;
   patched columns' null rates must stay close to the originals.
"""
import sys
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
from scipy.stats import spearmanr

sys.path.insert(0, "src")
from quantagent.factors.alpha101 import compute_alpha101

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
OLD = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v88.parquet"
NEW = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v88_rankfix.parquet"

BAD = [1, 3, 13, 15, 16, 18, 20, 27, 29, 37, 40, 42, 44, 45, 50, 55, 61, 62, 65, 73, 94, 95]
NAMES = [f"alpha{n:03d}" for n in BAD]
TEST_DATES = ["2026-05-13", "2025-06-04", "2024-06-03", "2022-06-01", "2020-06-01", "2019-01-02"]
UNTOUCHED_SAMPLE = ["alpha002", "alpha014", "alpha026", "alpha038", "alpha101",
                    "gtja001", "gtja016", "close", "volume", "is_limit_up"]

raw_cols = ["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"]
panel = ds.dataset(PANEL).to_table(columns=raw_cols).to_pandas()
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel["symbol"] = panel["symbol"].astype(str)

new_ds = ds.dataset(NEW)
old_ds = ds.dataset(OLD)

print("=== 1. reproducibility of patched columns (trailing-window recompute) ===")
worst = 1.0
for dt in TEST_DATES:
    ts = pd.Timestamp(dt)
    lo = ts - pd.Timedelta(days=1350)  # ~900 trading days
    work = panel[(panel["trade_date"] >= lo) & (panel["trade_date"] <= ts)]
    wide = compute_alpha101(work, names=NAMES, wide=True)
    sl = wide[wide["trade_date"] == ts].copy()
    sl["symbol"] = sl["symbol"].astype(str)
    gold = new_ds.to_table(columns=["trade_date", "symbol"] + NAMES,
                           filter=pc.field("trade_date") == ts).to_pandas()
    gold["symbol"] = gold["symbol"].astype(str)
    m = gold.merge(sl, on=["trade_date", "symbol"], suffixes=("_g", "_r"))
    rhos = {}
    for name in NAMES:
        a, b = m[f"{name}_g"].astype(float), m[f"{name}_r"].astype(float)
        ok = a.notna() & b.notna()
        rho = spearmanr(a[ok], b[ok])[0] if ok.sum() > 10 else np.nan
        rhos[name] = rho
        if np.isfinite(rho):
            worst = min(worst, rho)
    fails = {k: round(v, 4) for k, v in rhos.items() if not (np.isfinite(v) and v >= 0.99)}
    print(f"{dt}: merged {len(m)}, min spearman {np.nanmin(list(rhos.values())):.6f}"
          + (f"  FAILS: {fails}" if fails else "  all >= 0.99"), flush=True)
print(f"worst spearman across all dates/columns: {worst:.6f}")

print("\n=== 2. untouched columns identical to original v88 ===")
ts = pd.Timestamp("2024-06-03")
o = old_ds.to_table(columns=["trade_date", "symbol"] + UNTOUCHED_SAMPLE,
                    filter=pc.field("trade_date") == ts).to_pandas()
n = new_ds.to_table(columns=["trade_date", "symbol"] + UNTOUCHED_SAMPLE,
                    filter=pc.field("trade_date") == ts).to_pandas()
m = o.merge(n, on=["trade_date", "symbol"], suffixes=("_o", "_n"))
assert len(m) == len(o) == len(n), "row mismatch"
for c in UNTOUCHED_SAMPLE:
    a = m[f"{c}_o"].to_numpy()
    b = m[f"{c}_n"].to_numpy()
    if a.dtype.kind in "fc":
        same = np.array_equal(a, b, equal_nan=True)
    else:
        same = np.array_equal(a, b)
    print(f"  {c:12s} identical: {same}")

print("\n=== 3. null-rate drift on patched columns (old vs new, whole file) ===")
for name in NAMES:
    co = old_ds.to_table(columns=[name]).to_pandas()[name]
    cn = new_ds.to_table(columns=[name]).to_pandas()[name]
    ro, rn = co.isna().mean(), cn.isna().mean()
    flag = "" if abs(rn - ro) < 0.01 else "  <-- CHECK"
    print(f"  {name}: old {ro:.4%}  new {rn:.4%}{flag}")
print("\nDONE")
