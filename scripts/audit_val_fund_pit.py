#!/usr/bin/env python3
"""H-020 PIT leakage audit for the val_fund daily block (independent of the
builder's own asserts). Runs the G-PIT gates from
VALUATION_FUNDAMENTAL_INTEGRATION_PLAN.md and prints a coverage-by-year table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BLOCK = REPO / "runtime/data/v7/silver/valuation/val_fund_features.parquet"
METRICS = REPO / "runtime/data/v7/silver/fundamentals/metrics_panel.parquet"
QUARANTINE = pd.Timestamp("2025-09-01")
FRESH = pd.Timestamp("2026-05-19")


def main() -> int:
    b = pd.read_parquet(BLOCK)
    b["trade_date"] = pd.to_datetime(b["trade_date"])
    b["symbol"] = b["symbol"].astype(str)
    m = pd.read_parquet(METRICS, columns=["symbol", "period_end", "available_at", "roe", "bps"])
    m["available_at"] = pd.to_datetime(m["available_at"])
    m["period_end"] = pd.to_datetime(m["period_end"])
    m["symbol"] = m["symbol"].astype(str)

    fails = []

    # G-PIT-3 independent recompute for a sample of rows: the roe visible at
    # trade_date must equal the latest metrics_panel roe with available_at <= td.
    rng = np.random.default_rng(0)
    samp = b[b["roe"].notna()].sample(min(4000, int(b["roe"].notna().sum())), random_state=1)
    # deterministic as-of: latest fiscal period wins on equal available_at (matches builder)
    ms = m.dropna(subset=["available_at"]).sort_values(["available_at", "period_end"])
    ok = 0
    checked = 0
    for sym, grp in samp.groupby("symbol"):
        mm = ms[ms["symbol"] == sym]
        if mm.empty:
            continue
        for _, r in grp.iterrows():
            visible = mm[mm["available_at"] <= r["trade_date"]]
            if visible.empty:
                continue
            exp = visible.iloc[-1]["roe"]
            checked += 1
            if pd.isna(exp) and pd.isna(r["roe"]):
                ok += 1
            elif not pd.isna(exp) and abs(float(exp) - float(r["roe"])) < 1e-6:
                ok += 1
    print(f"G-PIT-3 as-of roe recompute: {ok}/{checked} match "
          f"({ok/checked:.2%})" if checked else "G-PIT-3: no rows")
    if checked and ok / checked < 0.999:
        fails.append(f"G-PIT-3 as-of mismatch {ok}/{checked}")

    # G-PIT-4 cross-sectional percentile is within-date: pick a date, recompute
    d = b[b["valuation_percentile"].notna()]["trade_date"].mode().iloc[0]
    day = b[b["trade_date"] == d]  # FULL day's rows (builder ranks over the full cross-section)
    if len(day) > 50:
        ey = day["earnings_yield"].rank(pct=True)
        by = day["book_yield"].rank(pct=True)
        recomputed = ((ey + by) / 2.0).to_numpy()
        got = day["valuation_percentile"].to_numpy()
        both = ~(np.isnan(recomputed) | np.isnan(got))
        maxdiff = float(np.abs(recomputed[both] - got[both]).max()) if both.any() else 0.0
        print(f"G-PIT-4 within-date percentile recompute on {d.date()} "
              f"(n={len(day)}, compared={both.sum()}): max|diff|={maxdiff:.2e}")
        if maxdiff > 1e-9:
            fails.append(f"G-PIT-4 percentile not within-date (diff {maxdiff})")

    # value sanity
    pb = b["pb"].dropna()
    pe = b["pe_ttm"].dropna()
    print(f"pb: n={len(pb):,} min={pb.min():.3f} med={pb.median():.2f} p99={pb.quantile(.99):.1f} neg={(pb<0).mean():.2%}")
    print(f"pe_ttm: n={len(pe):,} min={pe.min():.3f} med={pe.median():.1f} p99={pe.quantile(.99):.0f} neg={(pe<0).mean():.2%}")
    if (pb < 0).any() or (pe < 0).any():
        fails.append("negative pb/pe leaked (should be NaN)")

    # coverage by year (recent years must be high for pb/roe)
    b["yr"] = b["trade_date"].dt.year
    cov = b.groupby("yr").agg(
        rows=("pb", "size"),
        pb=("pb", lambda s: s.notna().mean()),
        roe=("roe", lambda s: s.notna().mean()),
        pe=("pe_ttm", lambda s: s.notna().mean()),
        valp=("valuation_percentile", lambda s: s.notna().mean()),
    )
    print("\ncoverage by year:")
    print(cov.to_string(float_format=lambda x: f"{x:.1%}" if x < 1.5 else f"{x:.0f}"))

    # quarantine/fresh sanity: block may span all dates (it's built from the
    # training keys which end pre-quarantine) — report the max date, do not read perf
    print(f"\nblock max trade_date: {b['trade_date'].max().date()} "
          f"(training keys are pre-quarantine {QUARANTINE.date()})")
    if b["trade_date"].max() >= FRESH:
        fails.append("block contains fresh-holdout dates (should not from plus7clean keys)")

    print("\n=== G-PIT AUDIT:", "ALL PASS" if not fails else f"FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
