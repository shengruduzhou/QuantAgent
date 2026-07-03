#!/usr/bin/env python3
# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): one-shot overlay split analysis.
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""Regime-split A/B: where does the LLM-side overlay achieve positive resonance?

Computes lookahead-safe daily portfolio returns (weight at t, return t->t+1)
for baseline factor vs evidence-overlay vs equal-weight all-A, labels each day
bull/bear/sideways from the benchmark trailing return, and reports annualized
excess (over eqw all-A) per regime. Light (no execution sim) so the relative
overlay-vs-baseline comparison per regime is what matters.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

PRED_DEFAULT = "runtime/reports/v8/deep/v8_full_v3_20260602_051048/short_5d/predictions.parquet"
CORE = "runtime/data/v7/gold/training_dataset/training_dataset_core30.parquet"
PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"

EVID = ["core_policy_score", "core_sentiment_score", "fundamental_quality_score",
        "sector_resonance_score", "dip_buy_flow_score", "old_dealer_risk_score"]
W = {"alpha": 0.55, "fundamental_quality_score": 0.15, "sector_resonance_score": 0.12,
     "core_policy_score": 0.10, "core_sentiment_score": 0.05, "dip_buy_flow_score": 0.05,
     "old_dealer_risk_score": -0.12}
ANN = 244  # A-share trading days/yr


def _z(g):
    s = g.std()
    return (g - g.mean()) / s if s and s > 1e-12 else g * 0.0


def _port_ret(df, score, fwd, top_k):
    """Daily equal-weight return of the top-K by `score`, using fwd return."""
    d = df[["trade_date", "symbol", score]].merge(fwd, on=["trade_date", "symbol"], how="left")
    d = d.dropna(subset=[score, "fwd_ret"])
    d = d.sort_values(["trade_date", score], ascending=[True, False])
    d["rank"] = d.groupby("trade_date").cumcount()
    top = d[d["rank"] < top_k]
    return top.groupby("trade_date")["fwd_ret"].mean()


def _ann(daily):
    if len(daily) == 0:
        return 0.0
    return float((1 + daily.mean()) ** ANN - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--start", default="2024-08-01")
    ap.add_argument("--bull", type=float, default=0.05, help="60d benchmark cumret > this => bull")
    ap.add_argument("--bear", type=float, default=-0.05)
    ap.add_argument("--pred", default=PRED_DEFAULT)
    args = ap.parse_args()

    preds = pd.read_parquet(args.pred); preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    preds = preds[preds["trade_date"] >= pd.Timestamp(args.start)]
    core = pd.read_parquet(CORE, columns=["trade_date", "symbol", *EVID]); core["trade_date"] = pd.to_datetime(core["trade_date"])
    df = preds.merge(core, on=["trade_date", "symbol"], how="inner")
    df["overlay_score"] = W["alpha"] * df.groupby("trade_date")["alpha_score"].transform(_z)
    for c in EVID:
        df["overlay_score"] += W.get(c, 0.0) * df.groupby("trade_date")[c].transform(_z)

    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "close"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[panel["trade_date"] >= pd.Timestamp(args.start) - pd.Timedelta(days=10)].sort_values(["symbol", "trade_date"])
    panel["fwd_ret"] = panel.groupby("symbol")["close"].shift(-1) / panel["close"] - 1.0
    fwd = panel[["trade_date", "symbol", "fwd_ret"]]

    base = _port_ret(df, "alpha_score", fwd, args.top_k)
    over = _port_ret(df, "overlay_score", fwd, args.top_k)
    # equal-weight all-A benchmark daily return
    bench = panel.groupby("trade_date")["fwd_ret"].mean()
    bench = bench.loc[bench.index.isin(base.index)]

    # regime label from benchmark 60d trailing cumulative return
    bench_cum = (1 + bench).cumprod()
    trail = bench_cum / bench_cum.shift(60) - 1.0
    regime = pd.Series(np.where(trail > args.bull, "bull", np.where(trail < args.bear, "bear", "sideways")), index=bench.index)

    print(f"days: {len(base)} | regime counts: {regime.value_counts().to_dict()}\n")
    print(f"{'regime':10} {'n':>4} {'base_ann':>9} {'over_ann':>9} {'bench_ann':>9} "
          f"{'base_exc':>9} {'over_exc':>9} {'over-base':>10}")
    for reg in ["bull", "sideways", "bear", "ALL"]:
        mask = regime == reg if reg != "ALL" else pd.Series(True, index=regime.index)
        idx = regime.index[mask]
        b, o, m = _ann(base.loc[idx]), _ann(over.loc[idx]), _ann(bench.loc[idx])
        print(f"{reg:10} {int(mask.sum()):>4} {b:>9.4f} {o:>9.4f} {m:>9.4f} "
              f"{b-m:>9.4f} {o-m:>9.4f} {o-b:>+10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
