#!/usr/bin/env python3
"""H-020 -> H-021 bridge: cheap CPU rank-IC pre-check of the new valuation/
fundamental features vs forward returns on the PRE-QUARANTINE training window.

Purpose: decide whether the GPU retrain (H-021) is worth it BEFORE spending GPU.
Not a selection step (no config chosen); a signal-existence probe. Quarantine
guard: only trade_date < 2025-09-01 rows are used; fresh holdout never touched.

Reports per feature: mean cross-sectional Spearman IC, ICIR, |mean|/se t-stat,
and coverage, vs forward_return_20d and forward_return_60d (long-horizon labels,
matching where these features feed the LONG sleeve). Raw (un-neutralized) IC —
a first-pass existence check; sign is informative (cheap valuation -> negative
pe/pb IC, positive earnings_yield IC).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus7clean_fund.parquet"
QUARANTINE = pd.Timestamp("2025-09-01")
FEATURES = ["pb", "pe_ttm", "earnings_yield", "valuation_percentile", "pb_own_pctile_2y",
            "roe", "net_margin", "gross_margin", "revenue_yoy", "net_income_yoy",
            "debt_to_asset", "inventory_turnover", "quality_composite", "growth_composite"]
LABELS = ["forward_return_20d", "forward_return_60d"]


def daily_ic(df: pd.DataFrame, feat: str, label: str) -> pd.Series:
    def _ic(g: pd.DataFrame) -> float:
        s = g[[feat, label]].dropna()
        if len(s) < 30:
            return np.nan
        return s[feat].rank().corr(s[label].rank())
    return df.groupby("trade_date", group_keys=False).apply(_ic)


def main() -> int:
    cols = ["symbol", "trade_date"] + FEATURES + LABELS
    df = pd.read_parquet(DS, columns=cols)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[df["trade_date"] < QUARANTINE]
    assert df["trade_date"].max() < QUARANTINE, "quarantine breach"
    print(f"rows (pre-quarantine): {len(df):,}  dates: {df['trade_date'].nunique()}  "
          f"range {df['trade_date'].min().date()}..{df['trade_date'].max().date()}\n")
    print(f"{'feature':22s} {'label':20s} {'meanIC':>8s} {'ICIR':>7s} {'t':>7s} {'cov':>6s}")
    rows = []
    for label in LABELS:
        for feat in FEATURES:
            ic = daily_ic(df, feat, label).dropna()
            if ic.empty:
                continue
            m, sd = ic.mean(), ic.std(ddof=1)
            icir = m / sd if sd > 0 else np.nan
            t = m / (sd / np.sqrt(len(ic))) if sd > 0 else np.nan
            cov = df[feat].notna().mean()
            rows.append((feat, label, m, icir, t, cov, len(ic)))
            print(f"{feat:22s} {label:20s} {m:+8.4f} {icir:+7.3f} {t:+7.2f} {cov:6.1%}")
    out = pd.DataFrame(rows, columns=["feature", "label", "mean_ic", "icir", "t", "coverage", "n_dates"])
    op = REPO / "runtime/reports/v89_closed_loop/val_fund_ic_precheck.csv"
    op.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(op, index=False)
    print(f"\nwrote {op}")
    # headline: strongest |IC| features
    strong = out.reindex(out["mean_ic"].abs().sort_values(ascending=False).index).head(6)
    print("\nstrongest |mean IC|:")
    for _, r in strong.iterrows():
        print(f"  {r['feature']:22s} {r['label']:20s} IC {r['mean_ic']:+.4f} (t {r['t']:+.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
