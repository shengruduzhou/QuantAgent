# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): v8.8 rank-corruption forensics: mission complete, kept as evidence chain.
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""Is alpha045's <0.99 a residual-corruption signal or intrinsic numeric chaos?
Two full-universe recomputes with current code, SAME date, different frame starts.
If they disagree with each other at the same ~0.97 level, 0.97 is the factor's
reproducibility ceiling (2-point rolling corr -> +/-1-dense ranks amplify
float accumulator noise), not corruption."""
import sys
import pandas as pd
import numpy as np
import pyarrow.dataset as ds
from scipy.stats import spearmanr

sys.path.insert(0, "src")
from quantagent.factors.alpha101 import compute_alpha101

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
TEST = pd.Timestamp("2026-05-13")
panel = ds.dataset(PANEL).to_table(
    columns=["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"]
).to_pandas()
panel["trade_date"] = pd.to_datetime(panel["trade_date"])
panel["symbol"] = panel["symbol"].astype(str)

outs = {}
for label, lo in (("startA_2022", TEST - pd.Timedelta(days=1350)),
                  ("startB_2021", TEST - pd.Timedelta(days=1800))):
    work = panel[(panel["trade_date"] >= lo) & (panel["trade_date"] <= TEST)]
    w = compute_alpha101(work, names=["alpha045"], wide=True)
    sl = w[w["trade_date"] == TEST][["symbol", "alpha045"]].copy()
    sl["symbol"] = sl["symbol"].astype(str)
    outs[label] = sl.set_index("symbol")["alpha045"]

a, b = outs["startA_2022"].align(outs["startB_2021"], join="inner")
ok = a.notna() & b.notna()
rho = spearmanr(a[ok], b[ok])[0]
exact = (a[ok] == b[ok]).mean()
print(f"alpha045 self-consistency (same code, same date, different frame start):")
print(f"  spearman {rho:.6f}   exact-equal share {exact:.4f}   n {ok.sum()}")
