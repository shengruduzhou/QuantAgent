"""Cheap positive proof: does gold match a batch-local (300-symbol) computation?"""
import sys
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
import pyarrow.compute as pc
from scipy.stats import spearmanr

sys.path.insert(0, "src")
from quantagent.factors.alpha101 import compute_alpha101

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
GOLD = "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v88.parquet"
BAD = [3, 13, 15, 16, 27, 42, 50, 61, 62, 65, 95]
NAMES = [f"alpha{n:03d}" for n in BAD]
TEST_DATE = pd.Timestamp("2026-05-13")

raw_cols = ["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"]
panel = ds.dataset(PANEL).to_table(columns=raw_cols).to_pandas()
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel["symbol"] = panel["symbol"].astype(str)

build_syms = sorted(panel.loc[panel["trade_date"] >= "2018-01-02", "symbol"].unique())
batch0 = set(build_syms[:300])
work = panel[(panel["trade_date"] >= "2024-06-01") & (panel["symbol"].isin(batch0))]
print(f"batch0: {len(batch0)} syms, work {len(work)} rows")

wide = compute_alpha101(work, names=NAMES, wide=True)
sl = wide[wide["trade_date"] == TEST_DATE].copy()
sl["symbol"] = sl["symbol"].astype(str)

gold = ds.dataset(GOLD).to_table(
    columns=["trade_date", "symbol"] + NAMES,
    filter=pc.field("trade_date") == TEST_DATE,
).to_pandas()
gold["symbol"] = gold["symbol"].astype(str)
gold = gold[gold["symbol"].isin(batch0)]

m = gold.merge(sl, on=["trade_date", "symbol"], suffixes=("_g", "_r"))
print(f"merged {len(m)} rows @ {TEST_DATE.date()}")
for name in NAMES:
    a, b = m[f"{name}_g"].astype(float), m[f"{name}_r"].astype(float)
    ok = a.notna() & b.notna()
    rho = spearmanr(a[ok], b[ok])[0] if ok.sum() > 10 else np.nan
    mad = (a[ok] - b[ok]).abs().median()
    print(f"  {name}  spearman {rho:.6f}  median|diff| {mad:.6g}  n {ok.sum()}")
