#!/usr/bin/env python3
"""How much of the frictionless equal-weight all-A benchmark is actually capturable?

The project's stated target is excess vs the *paper* equal-weight all-A index
(close-to-close mean of every listed name). That index holds suspended names
through gaps, earns sealed limit-up moves, includes ST shells, and pays no
costs. This script runs an *executable* equal-weight all-A portfolio through
the SAME strict simulator used for strategies (monthly rebalance, tradability
flags, costs, lot sizes) and reports the implementability gap.

Use the result to contextualise strategy excess: a strategy can only be
fairly judged against what a real benchmark replica could earn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
ANN = 244


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2024-08-28")
    ap.add_argument("--end", default=None)
    ap.add_argument("--rebalance", default="ME", help="pandas offset alias for rebalance dates (ME = month end)")
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    ap.add_argument("--output", default="runtime/reports/v8/baseline_protocol/executable_benchmark.json")
    args = ap.parse_args()

    cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
            "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    panel = pd.read_parquet(PANEL, columns=cols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[panel["trade_date"] >= pd.Timestamp(args.start) - pd.Timedelta(days=10)]
    if args.end:
        panel = panel[panel["trade_date"] <= pd.Timestamp(args.end)]

    dates = pd.DatetimeIndex(sorted(panel.loc[panel["trade_date"] >= args.start, "trade_date"].unique()))
    # paper benchmark
    px = panel[panel["trade_date"].isin(dates)].pivot_table(index="trade_date", columns="symbol", values="close")
    paper_daily = px.pct_change(fill_method=None).mean(axis=1).dropna()
    paper_total = float((1 + paper_daily).prod() - 1)
    paper_ann = float((1 + paper_total) ** (ANN / max(1, len(paper_daily))) - 1)

    # executable replica: month-end equal weights over names buyable that day
    reb_dates = [d for i, d in enumerate(dates) if i == 0 or d.month != dates[i - 1].month]
    rows = {}
    for d in reb_dates:
        day = panel[panel["trade_date"] == d]
        ok = day[
            ~day["is_suspended"].fillna(False).astype(bool)
            & ~day["is_st"].fillna(False).astype(bool)
            & ~day["is_limit_up"].fillna(False).astype(bool)
            & day["close"].gt(0)
        ]["symbol"].astype(str)
        if len(ok):
            rows[d] = pd.Series(1.0 / len(ok), index=ok.values)
    tw = pd.DataFrame(rows).T.fillna(0.0)
    tw.index.name = "trade_date"
    res = run_strict_backtest_v8(
        tw.sort_index(), panel,
        config=AShareExecutionSimulationConfig(initial_cash=10_000_000.0, slippage_bps=args.slippage_bps),
    )
    m = res.metrics
    out = {
        "window": f"{dates.min().date()}..{dates.max().date()}",
        "paper_eqw_all_A": {"total": round(paper_total, 4), "ann": round(paper_ann, 4)},
        "executable_eqw_all_A": {"total": round(m.total_return, 4), "ann": round(m.annualized_return, 4),
                                 "sharpe": round(m.sharpe, 3), "maxDD": round(m.max_drawdown, 4)},
        "implementability_gap_ann": round(paper_ann - m.annualized_return, 4),
        "rebalance_dates": len(reb_dates),
        "slippage_bps": args.slippage_bps,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
