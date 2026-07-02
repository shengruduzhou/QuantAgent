#!/usr/bin/env python3
"""Stage 7 (new signal type): do PIT-safe FUNDAMENTAL factors have robust
cross-regime selection edge — where daily price-volume factors did not?

Builds cross-sectional value/quality/growth factors from the PIT fundamentals
panel (available_at = real announce_date + 1d, no look-ahead), merges them onto
the price panel with a backward as-of join, and screens each factor through the
multi-window regime-stability test (long-only top-k, after-cost, vs the
same-universe tradable basket, per non-overlapping window). Ranks factors by
consistency of excess (% positive windows + excess IR), not single-window CAGR.

Fast vectorised screen via the policy_search engine; any winner must then be
re-confirmed through the strict simulator (baseline_protocol).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.portfolio.policy_search import (
    PolicyConfig, backtest_policy, prepare_working_frame, universe_benchmark,
)

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
FUND = "runtime/data/v7/silver/fundamentals/metrics_panel.parquet"


def _pit_merge(panel: pd.DataFrame, fund: pd.DataFrame, fcols: list[str]) -> pd.DataFrame:
    """Backward as-of: each (symbol, trade_date) gets the latest fundamentals
    whose available_at <= trade_date (PIT-safe, no look-ahead)."""
    f = fund[["symbol", "available_at", *fcols]].copy()
    f["available_at"] = pd.to_datetime(f["available_at"], errors="coerce")
    f = f.dropna(subset=["available_at"]).sort_values("available_at")
    p = panel.sort_values("trade_date")
    merged = pd.merge_asof(p, f, left_on="trade_date", right_on="available_at",
                           by="symbol", direction="backward")
    return merged


def _factors(df: pd.DataFrame) -> dict[str, pd.Series]:
    close = pd.to_numeric(df["close"], errors="coerce")
    return {
        "earnings_yield": pd.to_numeric(df["eps_basic"], errors="coerce") / close,     # value
        "book_to_price": pd.to_numeric(df["bps"], errors="coerce") / close,            # value
        "roe": pd.to_numeric(df["roe"], errors="coerce"),                              # quality
        "net_margin": pd.to_numeric(df["net_margin"], errors="coerce"),               # quality
        "gross_margin": pd.to_numeric(df["gross_margin"], errors="coerce"),           # quality
        "low_leverage": -pd.to_numeric(df["debt_to_asset_ratio"], errors="coerce"),   # quality
        "earnings_growth": pd.to_numeric(df["net_income_yoy"], errors="coerce"),       # change
        "revenue_growth": pd.to_numeric(df["revenue_yoy"], errors="coerce"),           # change
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2018-01-02")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--rebalance-days", type=int, default=20)
    ap.add_argument("--window-days", type=int, default=120)
    ap.add_argument("--output-dir", default="runtime/stage7_fundamental_stability")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "close", "amount",
                                            "is_st", "is_suspended", "is_limit_up"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce")
    panel = panel[panel["trade_date"] >= args.start].reset_index(drop=True)
    fund = pd.read_parquet(FUND)
    fcols = ["eps_basic", "bps", "roe", "net_margin", "gross_margin",
             "debt_to_asset_ratio", "net_income_yoy", "revenue_yoy"]
    merged = _pit_merge(panel, fund, fcols)
    factors = _factors(merged)
    sector = pd.read_parquet(SECTOR)

    dates = sorted(panel["trade_date"].dropna().unique())
    windows = [(dates[i], dates[min(i + args.window_days, len(dates)) - 1])
               for i in range(0, len(dates), args.window_days)]
    windows = [(s, e) for (s, e) in windows
               if pd.Index(dates).get_indexer([e])[0] - pd.Index(dates).get_indexer([s])[0] >= 40]

    rows = []
    for name, fac in factors.items():
        preds = merged[["symbol", "trade_date"]].copy()
        preds["alpha_5d"] = fac.to_numpy()
        preds["alpha_1d"] = preds["alpha_5d"]; preds["alpha_20d"] = preds["alpha_5d"]
        work = prepare_working_frame(preds, panel, sector)
        cfg = PolicyConfig(horizon=5, top_k=args.top_k, rebalance_days=args.rebalance_days,
                           side="long_only", transform="csrank", neutralize="none", liquidity_filter="ex_bottom_30pct")
        excesses = []
        for (ws, we) in windows:
            wk = work[(work["trade_date"] >= ws) & (work["trade_date"] <= we)]
            if wk["alpha_5d"].notna().sum() < 100:
                continue
            res = backtest_policy(wk, cfg)
            bm = universe_benchmark(wk)
            ex = res.metrics["cagr"] - bm["cagr"]
            if np.isfinite(ex):
                excesses.append(ex)
        ex = np.array(excesses, dtype=float)
        n = len(ex)
        if n < 4:
            continue
        mean_ex = float(np.mean(ex)); med_ex = float(np.median(ex))
        std_ex = float(np.std(ex, ddof=1)); ir = mean_ex / std_ex if std_ex > 1e-9 else float("nan")
        rows.append({"factor": name, "n_windows": n,
                     "pct_pos_excess": round(float((ex > 0).mean()) * 100, 1),
                     "mean_excess": round(mean_ex, 4), "median_excess": round(med_ex, 4),
                     "excess_IR": round(ir, 3), "worst": round(float(np.min(ex)), 4),
                     "best": round(float(np.max(ex)), 4)})
        print(f"  {name:16} n={n} %pos {rows[-1]['pct_pos_excess']:>5}  median_ex {med_ex:+.2%}  IR {ir:+.2f}", flush=True)

    lb = pd.DataFrame(rows).sort_values(["pct_pos_excess", "excess_IR"], ascending=False)
    lb.to_csv(out / "fundamental_factor_stability.csv", index=False)
    print("\n=== FUNDAMENTAL FACTOR STABILITY (sorted) ===")
    print(lb.to_string(index=False))
    robust = lb[(lb["pct_pos_excess"] >= 60) & (lb["excess_IR"] >= 0.5) & (lb["median_excess"] > 0)]
    verdict = (f"ROBUST candidates: {robust['factor'].tolist()}" if not robust.empty
               else "NO fundamental factor shows robust cross-regime excess (≥60% windows + IR≥0.5 + median>0)")
    (out / "summary.json").write_text(json.dumps({"verdict": verdict, "ranking": rows,
        "config": {"top_k": args.top_k, "rebalance_days": args.rebalance_days, "window_days": args.window_days,
                   "start": args.start, "n_windows": len(windows)}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
