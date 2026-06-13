#!/usr/bin/env python3
"""做T overlay backtest on the paper-account holdings (real 1-minute bars).

For every (trade_date, held symbol) in the paper replay, runs the CAUSAL
intraday 做T FSM (`intraday_dot_strategy.simulate_dot_day`) on that day's
cached 1-minute bars and aggregates the incremental return the overlay
would have added on top of the base book.

T+1 legality: the FSM buys a dip then sells the same notional later the
same day — legal because the SOLD shares come from the pre-existing base
position (做T fraction must stay <= base weight, enforced by --dot-fraction).

Regime gating (user spec: 牛市/震荡适合做T): the overlay only trades on
days whose benchmark-60d-trail regime is bull or sideways; bear days idle.

Costs per executed round trip (both legs):
  2x commission (2.5bp) + sell stamp tax (5bp) + 2x slippage (--slippage-bps).

Outputs <out>/dot_overlay_{trades.csv, daily.csv, summary.json}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.execution.intraday_dot_strategy import DotParams, simulate_dot_day

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
MINUTE_DIR = Path("runtime/data/v7/silver/minute_bars")
ANN = 244


def _regime_by_date(panel: pd.DataFrame) -> pd.Series:
    px = panel.pivot_table(index="trade_date", columns="symbol", values="close")
    bench = px.pct_change(fill_method=None).mean(axis=1).dropna()
    cum = (1 + bench).cumprod().shift(1).bfill()
    trail = cum / cum.shift(60) - 1.0
    return pd.Series(np.where(trail > 0.05, "bull", np.where(trail < -0.05, "bear", "sideways")),
                     index=bench.index)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdings-csv", default="runtime/paper/replay_2026/holdings_daily.csv")
    ap.add_argument("--output-dir", default="runtime/paper/replay_2026")
    ap.add_argument("--dot-fraction", type=float, default=0.3, help="fraction of each position cycled per 做T")
    ap.add_argument("--target-pct", type=float, default=0.015)
    ap.add_argument("--stop-pct", type=float, default=0.012)
    ap.add_argument("--dip-buffer", type=float, default=0.002)
    ap.add_argument("--slippage-bps", type=float, default=8.0)
    ap.add_argument("--commission-bps", type=float, default=2.5)
    ap.add_argument("--stamp-bps", type=float, default=5.0)
    ap.add_argument("--regimes", default="bull,sideways", help="regimes where 做T is active")
    ap.add_argument("--max-pairs", type=int, default=0, help="cap (date,symbol) pairs for smoke (0=all)")
    args = ap.parse_args()

    cost_rt = (2 * args.commission_bps + args.stamp_bps + 2 * args.slippage_bps) / 1e4
    active_regimes = {r.strip() for r in args.regimes.split(",") if r.strip()}
    params = DotParams(target_pct=args.target_pct, stop_pct=args.stop_pct, dip_buffer=args.dip_buffer)

    holdings = pd.read_csv(args.holdings_csv)
    holdings["trade_date"] = pd.to_datetime(holdings["trade_date"])
    start = holdings["trade_date"].min() - pd.Timedelta(days=120)
    panel = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "close"])
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel[panel["trade_date"] >= start]
    regime = _regime_by_date(panel)

    rows: list[dict] = []
    symbols = sorted(holdings["symbol"].astype(str).unique())
    n_pairs = 0
    missing_minutes = 0
    for sym in symbols:
        path = MINUTE_DIR / f"{sym}.parquet"
        if not path.exists():
            missing_minutes += holdings[holdings["symbol"] == sym].shape[0]
            continue
        bars = pd.read_parquet(path)
        bars["trade_time"] = pd.to_datetime(bars["trade_time"])
        bars["d"] = bars["trade_time"].dt.normalize()
        bars_by_day = dict(tuple(bars.groupby("d")))
        for _, h in holdings[holdings["symbol"] == sym].iterrows():
            d = h["trade_date"]
            rg = regime.get(d, "sideways")
            if rg not in active_regimes:
                continue
            day_bars = bars_by_day.get(d)
            if day_bars is None or len(day_bars) < 30:
                missing_minutes += 1
                continue
            db = day_bars.copy()
            db["trade_date"] = str(d.date())
            result = simulate_dot_day(db, params, symbol=sym)
            if result.state == "waiting_no_entry" or result.ret is None:
                rows.append({"trade_date": d, "symbol": sym, "weight": h["weight"], "regime": rg,
                             "state": result.state, "gross_ret": 0.0, "net_ret": 0.0, "executed": False})
                continue
            net = float(result.ret) - cost_rt
            rows.append({"trade_date": d, "symbol": sym, "weight": h["weight"], "regime": rg,
                         "state": result.state, "gross_ret": float(result.ret),
                         "net_ret": net, "executed": True})
            n_pairs += 1
            if args.max_pairs and n_pairs >= args.max_pairs:
                break
        if args.max_pairs and n_pairs >= args.max_pairs:
            break

    trades = pd.DataFrame(rows)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if trades.empty:
        raise SystemExit(f"no 做T evaluations possible (missing minute bars for {missing_minutes} pairs?)")
    trades.to_csv(out_dir / "dot_overlay_trades.csv", index=False)

    executed = trades[trades["executed"]]
    daily = (trades.assign(contrib=trades["weight"] * args.dot_fraction * trades["net_ret"])
             .groupby("trade_date")["contrib"].sum().rename("dot_daily_uplift"))
    daily.to_csv(out_dir / "dot_overlay_daily.csv")
    n_days = max(1, daily.shape[0])
    ann_uplift = float((1 + daily).prod() ** (ANN / n_days) - 1)

    by_regime = {}
    for rg, g in executed.groupby("regime"):
        by_regime[rg] = {
            "attempts": int(len(g)),
            "hit_rate": round(float((g["state"] == "closed_profit").mean()), 3),
            "avg_gross": round(float(g["gross_ret"].mean()), 5),
            "avg_net": round(float(g["net_ret"].mean()), 5),
        }
    summary = {
        "pairs_evaluated": int(len(trades)),
        "executed_legs": int(len(executed)),
        "entry_rate": round(float(trades["executed"].mean()), 3),
        "hit_rate": round(float((executed["state"] == "closed_profit").mean()), 3) if len(executed) else None,
        "avg_net_per_leg": round(float(executed["net_ret"].mean()), 5) if len(executed) else None,
        "cost_per_roundtrip": round(cost_rt, 5),
        "dot_fraction": args.dot_fraction,
        "daily_uplift_mean_bps": round(float(daily.mean()) * 1e4, 2),
        "annualized_uplift": round(ann_uplift, 4),
        "days_covered": int(n_days),
        "missing_minute_pairs": int(missing_minutes),
        "by_regime": by_regime,
        "params": {"target_pct": args.target_pct, "stop_pct": args.stop_pct,
                   "dip_buffer": args.dip_buffer, "regimes": sorted(active_regimes)},
    }
    (out_dir / "dot_overlay_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
