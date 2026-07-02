#!/usr/bin/env python3
"""Measure whether the board-aware limit-flag fix moves the trusted deliverable.

Re-runs the honest variant-C (flags ON, t+1 fill, eligible ranking) twice on the
same v8.9 winner predictions:
  (1) ORIGINAL panel flags (flat-10% limit approximation)
  (2) BOARD-FIXED flags (from market_panel_boardfix.parquet sidecar)

Non-destructive: reads only; writes nothing to silver. Prints both metric sets.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8
from baseline_protocol import _target_weights  # noqa: E402

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
BOARDFIX = "runtime/data/v7/silver/market_panel/market_panel_boardfix.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"


def run_variant_c(preds, panel, sector, top_k, slippage):
    flags = panel[["symbol", "trade_date", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]]
    p = preds.merge(flags, on=["symbol", "trade_date"], how="left")
    trade_dates = sorted(panel["trade_date"].unique())
    tw = _target_weights(p, "alpha_score", top_k, eligible_only=True, delay_days=1, trade_dates=trade_dates)
    res = run_strict_backtest_v8(tw, panel, sector_map=sector,
                                 config=AShareExecutionSimulationConfig(initial_cash=1_000_000.0, slippage_bps=slippage))
    m = res.metrics
    calmar = (m.annualized_return / abs(m.max_drawdown)) if m.max_drawdown else float("nan")
    return dict(ann=m.annualized_return, maxDD=m.max_drawdown, sharpe=m.sharpe, calmar=calmar, total=m.total_return)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="runtime/reports/v89_closed_loop/ensemble_search_plus7/winner_predictions.parquet")
    ap.add_argument("--score-column", default="composite_score")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--start", default="2024-08-28")
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    args = ap.parse_args()

    preds = pd.read_parquet(args.predictions).rename(columns={args.score_column: "alpha_score"})
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    preds = preds[preds["trade_date"] >= pd.Timestamp(args.start)]

    cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
            "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    panel = pd.read_parquet(PANEL, columns=cols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    lo = preds["trade_date"].min() - pd.Timedelta(days=10)
    panel = panel[panel["trade_date"] >= lo]
    sector = pd.read_parquet(SECTOR)

    # Board-fixed panel: swap is_limit_up/down from the sidecar.
    bf = pd.read_parquet(BOARDFIX, columns=["symbol", "trade_date", "is_limit_up", "is_limit_down"])
    bf["trade_date"] = pd.to_datetime(bf["trade_date"])
    panel_fix = panel.drop(columns=["is_limit_up", "is_limit_down"]).merge(
        bf, on=["symbol", "trade_date"], how="left")
    panel_fix["is_limit_up"] = panel_fix["is_limit_up"].fillna(False).astype(bool)
    panel_fix["is_limit_down"] = panel_fix["is_limit_down"].fillna(False).astype(bool)

    orig = run_variant_c(preds, panel, sector, args.top_k, args.slippage_bps)
    fixed = run_variant_c(preds, panel_fix, sector, args.top_k, args.slippage_bps)

    print(f"{'variant-C':<22}{'ann':>9}{'maxDD':>9}{'calmar':>9}{'sharpe':>8}{'total':>9}")
    for name, m in [("orig flat-10% flags", orig), ("board-fixed flags", fixed)]:
        print(f"{name:<22}{m['ann']:>8.2%}{m['maxDD']:>8.2%}{m['calmar']:>9.2f}{m['sharpe']:>8.2f}{m['total']:>8.2%}")
    d_ann = fixed["ann"] - orig["ann"]; d_dd = fixed["maxDD"] - orig["maxDD"]
    print(f"\nΔ board-fix: ann {d_ann:+.2%}  maxDD {d_dd:+.2%}  calmar {fixed['calmar']-orig['calmar']:+.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
