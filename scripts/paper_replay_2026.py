#!/usr/bin/env python3
# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): one-shot replay.
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""2026 paper-account replay: v8.8 ensemble + hold-band + strict A-share sim.

The headline deliverable of the 模拟仓 milestone: replay 2026-01-02..latest
predictions through the SAME components the live loop will use —

  v8.8 judgment-routed ensemble scores
  -> eligibility filter (ST / suspended / limit-up sealed at signal close)
  -> hold-band target weights (enter rank<=30, exit rank>150, hold 50, t+1)
  -> strict simulator (costs, lots, participation caps, tradability flags)

Hold-band state is warmed up from --warmup-start so the book entering
January already reflects persistent holdings (no cold-start artifact);
NAV accounting starts at --start.

Outputs (under --output-dir):
  nav.csv, holdings_daily.csv, summary.json, report.md
with excess vs BOTH the paper equal-weight all-A index and the executable
equal-weight replica convention (paper minus the measured 12.7pp/yr gap is
NOT used — the replica is re-simulated on the same window for honesty).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
from quantagent.backtest.strict_v8 import run_strict_backtest_v8
from quantagent.portfolio.hold_band import HoldBandConfig, build_hold_band_weights, turnover_stats

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
SECTOR = "runtime/data/v7/silver/sector_map/sector_map.parquet"
PREDS = "runtime/reports/v8/deep/v88_judgment_20260611_2015/ensemble_composite.parquet"
ANN = 244


def _executable_benchmark(panel: pd.DataFrame, start: pd.Timestamp, slippage_bps: float) -> dict:
    dates = pd.DatetimeIndex(sorted(panel.loc[panel["trade_date"] >= start, "trade_date"].unique()))
    reb = [d for i, d in enumerate(dates) if i == 0 or d.month != dates[i - 1].month]
    rows = {}
    for d in reb:
        day = panel[panel["trade_date"] == d]
        ok = day[~day["is_suspended"].fillna(False).astype(bool)
                 & ~day["is_st"].fillna(False).astype(bool)
                 & ~day["is_limit_up"].fillna(False).astype(bool)
                 & day["close"].gt(0)]["symbol"].astype(str)
        if len(ok):
            rows[d] = pd.Series(1.0 / len(ok), index=ok.values)
    tw = pd.DataFrame(rows).T.fillna(0.0)
    tw.index.name = "trade_date"
    res = run_strict_backtest_v8(tw.sort_index(), panel,
                                 config=AShareExecutionSimulationConfig(initial_cash=10_000_000.0,
                                                                        slippage_bps=slippage_bps))
    m = res.metrics
    return {"ann": float(m.annualized_return), "total": float(m.total_return),
            "sharpe": float(m.sharpe), "maxDD": float(m.max_drawdown)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", default=PREDS)
    ap.add_argument("--score-column", default="composite_score")
    ap.add_argument("--start", default="2026-01-02")
    ap.add_argument("--warmup-start", default="2025-10-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--n-hold", type=int, default=50)
    ap.add_argument("--entry-rank", type=int, default=30)
    ap.add_argument("--exit-rank", type=int, default=150)
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    ap.add_argument("--initial-cash", type=float, default=1_000_000.0)
    ap.add_argument("--output-dir", default="runtime/paper/replay_2026")
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    warmup = pd.Timestamp(args.warmup_start)

    preds = pd.read_parquet(args.predictions)
    preds["trade_date"] = pd.to_datetime(preds["trade_date"])
    if args.score_column != "alpha_score":
        preds = preds.rename(columns={args.score_column: "alpha_score"})
    preds = preds[preds["trade_date"] >= warmup]
    if args.end:
        preds = preds[preds["trade_date"] <= pd.Timestamp(args.end)]
    if preds.empty:
        raise SystemExit("no predictions in window")

    panel_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount",
                  "available_at", "is_suspended", "is_st", "is_limit_up", "is_limit_down"]
    panel = pd.read_parquet(PANEL, columns=panel_cols)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[panel["trade_date"] >= warmup - pd.Timedelta(days=10)]
    if args.end:
        panel = panel[panel["trade_date"] <= pd.Timestamp(args.end) + pd.Timedelta(days=5)]
    sector = pd.read_parquet(SECTOR)
    trade_dates = sorted(panel["trade_date"].unique())

    flags = panel[["symbol", "trade_date", "is_st", "is_suspended", "is_limit_up"]]
    preds = preds.merge(flags, on=["symbol", "trade_date"], how="left")

    cfg = HoldBandConfig(n_hold=args.n_hold, entry_rank=args.entry_rank,
                         exit_rank=args.exit_rank, delay_days=1)
    tw_full = build_hold_band_weights(preds, config=cfg, trade_dates=trade_dates)
    tw = tw_full[tw_full.index >= start]
    if tw.empty:
        raise SystemExit("hold-band produced no weights after start date")

    sim_panel = panel[panel["trade_date"] >= start - pd.Timedelta(days=5)]
    res = run_strict_backtest_v8(
        tw, sim_panel, sector_map=sector,
        config=AShareExecutionSimulationConfig(initial_cash=args.initial_cash,
                                               slippage_bps=args.slippage_bps),
    )
    m = res.metrics

    # paper benchmark on the same dates
    px = sim_panel[sim_panel["trade_date"].isin(tw.index)].pivot_table(
        index="trade_date", columns="symbol", values="close")
    bench_daily = px.pct_change(fill_method=None).mean(axis=1).dropna()
    n_days = max(1, len(bench_daily))
    bench_total = float((1 + bench_daily).prod() - 1)
    bench_ann = float((1 + bench_total) ** (ANN / n_days) - 1)
    exec_bench = _executable_benchmark(sim_panel, start, args.slippage_bps)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.nav.rename("nav").to_csv(out_dir / "nav.csv")
    holdings = tw.stack()
    holdings = holdings[holdings > 0].rename("weight").reset_index()
    holdings.columns = ["trade_date", "symbol", "weight"]
    holdings.to_csv(out_dir / "holdings_daily.csv", index=False)

    tstats = turnover_stats(tw)
    summary = {
        "window": f"{tw.index.min().date()}..{tw.index.max().date()}",
        "config": {"n_hold": args.n_hold, "entry_rank": args.entry_rank, "exit_rank": args.exit_rank,
                   "delay_days": 1, "slippage_bps": args.slippage_bps,
                   "predictions": args.predictions, "warmup_start": args.warmup_start},
        "paper_account": {"total_return": round(m.total_return, 4),
                          "annualized": round(m.annualized_return, 4),
                          "sharpe": round(m.sharpe, 3), "maxDD": round(m.max_drawdown, 4),
                          "mean_daily_turnover": round(tstats["mean_daily_turnover"], 4)},
        "paper_eqw_all_A": {"total_return": round(bench_total, 4), "annualized": round(bench_ann, 4)},
        "executable_eqw_all_A": {k: round(v, 4) for k, v in exec_bench.items()},
        "excess_vs_paper_bench_ann": round(m.annualized_return - bench_ann, 4),
        "excess_vs_executable_bench_ann": round(m.annualized_return - exec_bench["ann"], 4),
        "n_unique_names_held": int(holdings["symbol"].nunique()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 2026 模拟仓回放报告（research/backtest only — not financial advice）",
        "",
        f"- 窗口: {summary['window']}（hold-band 自 {args.warmup_start} 暖机）",
        f"- 配置: v8.8 judgment ensemble + 持有带(进{args.entry_rank}/出{args.exit_rank}/持{args.n_hold}) + t+1 执行 + {args.slippage_bps}bps",
        "",
        "| 指标 | 模拟仓 | 纸面等权全A | 可执行等权全A |",
        "|---|---|---|---|",
        f"| 区间收益 | {m.total_return:+.2%} | {bench_total:+.2%} | {exec_bench['total']:+.2%} |",
        f"| 年化 | {m.annualized_return:+.2%} | {bench_ann:+.2%} | {exec_bench['ann']:+.2%} |",
        f"| 年化超额 | — | {m.annualized_return - bench_ann:+.2%} | {m.annualized_return - exec_bench['ann']:+.2%} |",
        f"| Sharpe | {m.sharpe:.2f} | — | {exec_bench['sharpe']:.2f} |",
        f"| maxDD | {m.max_drawdown:.2%} | — | {exec_bench['maxDD']:.2%} |",
        f"| 日均换手(单边) | {tstats['mean_daily_turnover']:.2%} | — | — |",
        "",
        f"持仓覆盖 {summary['n_unique_names_held']} 只；持仓明细 holdings_daily.csv；净值 nav.csv。",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
