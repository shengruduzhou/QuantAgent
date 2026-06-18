#!/usr/bin/env python3
"""Benchmark + equivalence harness for ``compute_alpha101``.

Times the original reference implementation against the vectorized fast path
(serial and with factor-level ``workers`` parallelism) on a slice of the silver
market panel, and reports the max abs difference so any numerical regression is
caught alongside the speedup.

Examples
--------
    # quick check on 60 symbols since 2022 (includes the slow reference baseline)
    AI_quant_venv/bin/python3 scripts/benchmark_alpha101.py --symbols 60 --start 2022-01-01

    # full panel, fast path only, 12 workers (skip the multi-hour reference)
    AI_quant_venv/bin/python3 scripts/benchmark_alpha101.py --symbols 0 --workers 12 --no-reference
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
COLS = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]


def _load(symbols: int, start: str | None) -> pd.DataFrame:
    df = pd.read_parquet(PANEL, columns=COLS)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["symbol"] = df["symbol"].astype(str)
    if start:
        df = df[df["trade_date"] >= pd.Timestamp(start)]
    if symbols and symbols > 0:
        keep = sorted(df["symbol"].unique())[:symbols]
        df = df[df["symbol"].isin(keep)]
    return df.reset_index(drop=True)


def _max_abs_diff(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, int]:
    cols = [c for c in a.columns if c.startswith("alpha")]
    worst = 0.0
    nan_mismatch = 0
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        nx, ny = np.isnan(x), np.isnan(y)
        nan_mismatch += int((nx != ny).sum())
        m = ~(nx | ny)
        d = x[m] - y[m]
        d = d[np.isfinite(d)]
        if d.size:
            worst = max(worst, float(np.abs(d).max()))
    return worst, nan_mismatch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", type=int, default=60, help="cap symbols (0 = all)")
    ap.add_argument("--start", default="2022-01-01", help="min trade_date (empty for all)")
    ap.add_argument("--workers", type=int, default=8, help="parallel workers for the fast path")
    ap.add_argument("--no-reference", action="store_true", help="skip the slow reference baseline")
    args = ap.parse_args()

    import quantagent.factors.alpha101 as alpha101
    from quantagent.factors.alpha101 import compute_alpha101

    df = _load(args.symbols, args.start or None)
    print(f"panel slice: {len(df):,} rows, {df['symbol'].nunique()} symbols, "
          f"{df['trade_date'].min().date()}..{df['trade_date'].max().date()}", flush=True)

    ref = None
    if not args.no_reference:
        alpha101._REFERENCE_HELPERS = True
        try:
            t = time.time()
            ref = compute_alpha101(df, wide=True)
            t_ref = time.time() - t
        finally:
            alpha101._REFERENCE_HELPERS = False
        print(f"reference (orig)      : {t_ref:8.1f}s", flush=True)

    t = time.time()
    fast = compute_alpha101(df, wide=True)
    t_fast = time.time() - t
    line = f"fast serial           : {t_fast:8.1f}s"
    if ref is not None:
        line += f"   ({t_ref / t_fast:6.1f}x)"
    print(line, flush=True)

    if args.workers and args.workers > 1:
        t = time.time()
        par = compute_alpha101(df, wide=True, workers=args.workers)
        t_par = time.time() - t
        line = f"fast workers={args.workers:<3d}      : {t_par:8.1f}s"
        if ref is not None:
            line += f"   ({t_ref / t_par:6.1f}x)"
        print(line, flush=True)
        d, nm = _max_abs_diff(fast, par)
        print(f"parallel vs serial    : max|diff| {d:.2e}, nan-mismatch {nm}", flush=True)

    if ref is not None:
        d, nm = _max_abs_diff(ref, fast)
        verdict = "PASS" if (d < 1e-9 and nm == 0) else "FAIL"
        print(f"fast vs reference     : max|diff| {d:.2e}, nan-mismatch {nm}  -> {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
