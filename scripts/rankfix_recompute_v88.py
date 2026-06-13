"""Recompute the 22 batch-rank-tainted alpha101 columns full-universe and patch v88.

Root cause (proven by /tmp/verify_batch_rank.py): the 2026-05-21 materialize run
used --batch-symbols 300, so every cross-sectional rank() was computed inside
alphabetically-contiguous 300-symbol batches instead of the full universe.
The 22 columns where rank enters nonlinearly (corr/cov of ranks, rank ratios,
rank<rank gates) drifted materially; single-outer-rank columns survived.

This recomputes those 22 columns with the current (correct, full-universe)
code over the original build window (>= 2018-01-02, all 3872 symbols, same
NaN-edge convention) and writes a patched copy of the v88 gold dataset.
"""
import json
import time
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import sys

sys.path.insert(0, "src")
from quantagent.factors.alpha101 import compute_alpha101

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
GOLD = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v88.parquet"
OUT = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v88_rankfix.parquet"
SIDE = "runtime/data/v7/gold/training_dataset/alpha101_rankfix_22cols.parquet"

BAD = [1, 3, 13, 15, 16, 18, 20, 27, 29, 37, 40, 42, 44, 45, 50, 55, 61, 62, 65, 73, 94, 95]
NAMES = [f"alpha{n:03d}" for n in BAD]

t0 = time.time()
raw_cols = ["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"]
panel = ds.dataset(PANEL).to_table(columns=raw_cols).to_pandas()
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel["symbol"] = panel["symbol"].astype(str)
work = panel[panel["trade_date"] >= "2018-01-02"].copy()
del panel
print(f"[{time.time()-t0:.0f}s] frame: {len(work)} rows, {work['symbol'].nunique()} symbols", flush=True)
assert len(work) == 7386272, f"frame rows {len(work)} != original build 7386272"

wide = compute_alpha101(work, names=NAMES, wide=True)
del work
print(f"[{time.time()-t0:.0f}s] recompute done: {wide.shape}", flush=True)
wide["trade_date"] = pd.to_datetime(wide["trade_date"])
wide["symbol"] = wide["symbol"].astype(str)
wide.to_parquet(SIDE, index=False)
print(f"[{time.time()-t0:.0f}s] side table written: {SIDE}", flush=True)

# ---- patch v88 ----
gold = pd.read_parquet(GOLD)
print(f"[{time.time()-t0:.0f}s] gold loaded: {gold.shape}", flush=True)
gold_td = pd.to_datetime(gold["trade_date"])
gold_sym = gold["symbol"].astype(str)

key_gold = pd.MultiIndex.from_arrays([gold_sym, gold_td])
wide_idx = wide.set_index(["symbol", "trade_date"])
missing = ~key_gold.isin(wide_idx.index)
assert missing.sum() == 0, f"{missing.sum()} gold rows missing from recompute"

aligned = wide_idx.reindex(key_gold)
del wide, wide_idx
for name in NAMES:
    old_dtype = gold[name].dtype
    gold[name] = aligned[name].to_numpy(dtype="float32" if old_dtype == np.float32 else float)
del aligned
print(f"[{time.time()-t0:.0f}s] columns replaced", flush=True)

gold.to_parquet(OUT, index=False)
print(f"[{time.time()-t0:.0f}s] patched dataset written: {OUT}", flush=True)

meta = {
    "source": GOLD,
    "patched_columns": NAMES,
    "reason": "original 2026-05-21 materialize-alpha181-v7 run used --batch-symbols 300; "
              "cross-sectional rank() was batch-local (300 alphabetically-contiguous symbols) "
              "instead of full-universe. Recomputed with full-universe single-pass "
              "compute_alpha101 (HEAD), frame >= 2018-01-02, all 3872 panel symbols.",
    "patched_at": pd.Timestamp.now().isoformat(),
}
with open(OUT.replace(".parquet", "_schema.json"), "w") as fh:
    json.dump(meta, fh, indent=2)
print(f"[{time.time()-t0:.0f}s] DONE", flush=True)
