#!/usr/bin/env python3
"""Evaluate 东财 DAILY fund-flow as a cross-sectional alpha factor.

Builds size-normalized fund-flow factors and measures their predictive power for
tradable forward returns (rank-IC + IR), with PIT-honest tradability filters
(drop ST / suspended / next-day limit-locked names, per honest-baseline-truth).

Inputs:
  runtime/data/v7/silver/fundflow_daily/fundflow_daily.parquet  (fetcher output)
  runtime/data/v7/silver/market_panel/market_panel.parquet
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

FF = "runtime/data/v7/silver/fundflow_daily/fundflow_daily.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"


def _ic(df: pd.DataFrame, fcol: str, rcol: str) -> tuple[float, float, int]:
    """Daily cross-sectional rank-IC mean, IR, and #days (>=20 names/day)."""
    daily = []
    for _, g in df.groupby("trade_date"):
        g = g.dropna(subset=[fcol, rcol])
        if len(g) >= 20 and g[fcol].nunique() > 5:
            daily.append(g[fcol].rank().corr(g[rcol].rank()))
    s = pd.Series(daily, dtype=float).dropna()
    if s.empty:
        return float("nan"), float("nan"), 0
    return float(s.mean()), float(s.mean() / s.std()) if s.std() > 0 else float("nan"), int(len(s))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fundflow", default=FF)
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--horizons", default="1,5", help="forward-return horizons (days)")
    args = ap.parse_args()

    ff = pd.read_parquet(args.fundflow)
    ff["symbol"] = ff["symbol"].astype(str)
    ff["trade_date"] = pd.to_datetime(ff["date"], errors="coerce").dt.normalize()
    panel = pd.read_parquet(args.panel, columns=["symbol", "trade_date", "close", "amount",
                                                 "is_st", "is_suspended", "is_limit_up", "is_limit_down"])
    panel["symbol"] = panel["symbol"].astype(str)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.normalize()
    panel = panel[panel["trade_date"] >= ff["trade_date"].min() - pd.Timedelta(days=5)]

    df = panel.merge(ff[["symbol", "trade_date", "main_net", "super_net", "large_net"]],
                     on=["symbol", "trade_date"], how="inner")
    df = df.sort_values(["symbol", "trade_date"])
    amt = pd.to_numeric(df["amount"], errors="coerce").replace(0.0, np.nan)
    # size-normalized factors (net inflow as a fraction of daily turnover)
    df["f_main_ratio"] = df["main_net"] / amt
    df["f_super_ratio"] = df["super_net"] / amt
    df["f_smart_ratio"] = (df["super_net"] + df["large_net"]) / amt
    g = df.groupby("symbol", sort=False)
    df["f_main_5d"] = g["f_main_ratio"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    df["f_main_20d"] = g["f_main_ratio"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["f_smart_5d"] = g["f_smart_ratio"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    # forward returns (close-to-close), tradability-filtered
    df["fwd_ret_1"] = g["close"].transform(lambda s: s.shift(-1) / s - 1.0)
    df["fwd_ret_5"] = g["close"].transform(lambda s: s.shift(-5) / s - 1.0)
    # PIT-honest: drop entries that can't be acted on next day
    tradable = (~df["is_st"].fillna(False).astype(bool)) & (~df["is_suspended"].fillna(False).astype(bool)) \
        & (~df["is_limit_up"].fillna(False).astype(bool))
    dft = df[tradable].copy()

    factors = ["f_main_ratio", "f_super_ratio", "f_smart_ratio", "f_main_5d", "f_main_20d", "f_smart_5d"]
    horizons = [int(h) for h in args.horizons.split(",")]
    out = {"coverage": {"rows": int(len(df)), "tradable_rows": int(len(dft)),
                        "symbols": int(df["symbol"].nunique()), "days": int(df["trade_date"].nunique()),
                        "date_range": [str(df["trade_date"].min().date()), str(df["trade_date"].max().date())]},
           "rank_IC": {}}
    for h in horizons:
        rcol = f"fwd_ret_{h}"
        out["rank_IC"][f"h{h}"] = {}
        for f in factors:
            ic, ir, n = _ic(dft, f, rcol)
            out["rank_IC"][f"h{h}"][f] = {"IC": round(ic, 4), "IR": round(ir, 3), "days": n}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
