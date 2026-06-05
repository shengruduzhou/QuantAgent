#!/usr/bin/env python3
"""A/B test: does the LLM-side evidence overlay improve the raw factor alpha?

Baseline  = raw factor ``alpha_score`` (top-K equal weight).
Overlay   = cross-sectional blend of alpha + deterministic evidence signals
            (policy / sentiment / fundamental / sector_resonance / dip
            − old_dealer) drawn from the core30 dataset — the lookahead-safe,
            historical stand-in for the live LLM hybrid pool.

Both go through the SAME strict A-share backtest; we also compute the
equal-weight all-A benchmark to report excess. This is the "LLM+factor
positive resonance" check requested for Phase 2.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8

PRED = "runtime/reports/v8/deep/v8_full_v3_20260602_051048/short_5d/predictions.parquet"
CORE = "runtime/data/v7/gold/training_dataset/training_dataset_core30.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"

EVID = ["core_policy_score", "core_sentiment_score", "fundamental_quality_score",
        "sector_resonance_score", "dip_buy_flow_score", "old_dealer_risk_score"]
# factor-dominant blend; evidence adds a tilt, old_dealer subtracts.
W = {"alpha": 0.55, "fundamental_quality_score": 0.15, "sector_resonance_score": 0.12,
     "core_policy_score": 0.10, "core_sentiment_score": 0.05, "dip_buy_flow_score": 0.05,
     "old_dealer_risk_score": -0.12}


def _zscore(g: pd.Series) -> pd.Series:
    s = g.std()
    return (g - g.mean()) / s if s and s > 1e-12 else g * 0.0


def _target_weights(df: pd.DataFrame, score_col: str, top_k: int) -> pd.DataFrame:
    d = df[["trade_date", "symbol", score_col]].copy()
    d = d.sort_values(["trade_date", score_col], ascending=[True, False])
    d["rank"] = d.groupby("trade_date").cumcount()
    d["w"] = (d["rank"] < top_k).astype(float) / float(top_k)
    return d.pivot_table(index="trade_date", columns="symbol", values="w", fill_value=0.0)


def _run(name: str, tw: pd.DataFrame, panel: pd.DataFrame, sector: pd.DataFrame) -> dict:
    res = run_strict_backtest_v8(tw, panel, sector_map=sector,
                                 config=AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=8.0))
    m = res.metrics
    return {"name": name, "total_return": m.total_return, "annualized_return": m.annualized_return,
            "sharpe": m.sharpe, "max_drawdown": m.max_drawdown}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--start", default="2024-08-01")
    args = ap.parse_args()

    preds = pd.read_parquet(PRED)
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    preds = preds[preds["trade_date"] >= pd.Timestamp(args.start)]
    core = pd.read_parquet(CORE, columns=["trade_date", "symbol", *EVID])
    core["trade_date"] = pd.to_datetime(core["trade_date"])
    df = preds.merge(core, on=["trade_date", "symbol"], how="inner")
    print(f"joined rows: {len(df)} | dates {df['trade_date'].min().date()}..{df['trade_date'].max().date()}")

    # cross-sectional z-scores per day
    df["z_alpha"] = df.groupby("trade_date")["alpha_score"].transform(_zscore)
    overlay = W["alpha"] * df["z_alpha"]
    for c in EVID:
        z = df.groupby("trade_date")[c].transform(_zscore)
        overlay = overlay + W.get(c, 0.0) * z
    df["overlay_score"] = overlay

    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "high", "low",
                                            "close", "volume", "amount", "available_at"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[panel["trade_date"] >= pd.Timestamp(args.start) - pd.Timedelta(days=5)]
    sector = pd.read_parquet(SECTOR)

    results = []
    results.append(_run("baseline_factor", _target_weights(df, "alpha_score", args.top_k), panel, sector))
    results.append(_run("overlay_llmside", _target_weights(df, "overlay_score", args.top_k), panel, sector))

    # equal-weight all-A benchmark on the same trade dates
    dates = sorted(df["trade_date"].unique())
    px = panel[panel["trade_date"].isin(dates)].pivot_table(index="trade_date", columns="symbol", values="close")
    daily_ret = px.pct_change().mean(axis=1).dropna()
    bench_total = float((1 + daily_ret).prod() - 1)
    yrs = max(1e-9, (px.index.max() - px.index.min()).days / 365.25)
    bench_ann = float((1 + bench_total) ** (1 / yrs) - 1)

    print(f"\n{'strategy':18} {'tot_ret':>9} {'ann_ret':>9} {'sharpe':>7} {'max_dd':>8} {'excess_ann':>10}")
    for r in results:
        print(f"{r['name']:18} {r['total_return']:>9.4f} {r['annualized_return']:>9.4f} "
              f"{r['sharpe']:>7.2f} {r['max_drawdown']:>8.4f} {r['annualized_return']-bench_ann:>10.4f}")
    print(f"{'eqw_all_A_bench':18} {bench_total:>9.4f} {bench_ann:>9.4f} {'-':>7} {'-':>8} {0.0:>10.4f}")
    delta = results[1]["annualized_return"] - results[0]["annualized_return"]
    print(f"\nRESONANCE: overlay − baseline annualized = {delta:+.4f} "
          f"({'POSITIVE (LLM-side helps)' if delta > 0 else 'NEGATIVE (no lift)'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
