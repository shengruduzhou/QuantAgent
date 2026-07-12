#!/usr/bin/env python3
# DEPRECATED (2026-07-04, DEAD_CODE_AUDIT.md / PRUNE_PLAN.md P-C): 做T on 1-min OHLCV: no realizable edge (stage3b/4 REJECT).
# Zero references found in scripts/src/tests/docs/systemd (dependency scan 2026-07-03).
# Scheduled for removal after 2026-10-01 if still unused. Do not build on this.
"""分时做T 决策 CLI —— 对持仓池逐票产出严格 JSON 做T建议（引擎+决策层）。

两种模式：
  replay   离线回放缓存分钟线的某一天（验证/复盘），每票每分钟跑因果引擎，
           在指定时间点产出决策；汇总信号分布。
  live     用 TickFlow 拉当日分钟线，对 forward 持仓池逐票产出"此刻"决策。

研究/参考用途；不下真实订单。卖出严格受 sellable_qty(昨仓) 约束。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from quantagent.execution.intraday_dot_engine import compute_intraday_state
from quantagent.execution.intraday_dot_decision import DecisionConfig, Position, decide

PANEL = "runtime/data/v7/silver/market_panel/market_panel.parquet"
MINUTE_DIR = Path("runtime/data/v7/silver/minute_bars")


def _prev_close_from_minutes(sym: str, as_of: pd.Timestamp) -> float | None:
    """昨收 = 分钟缓存中 as_of 之前最后一个交易日的最后一根收盘（永不陈旧）。"""
    p = MINUTE_DIR / f"{sym}.parquet"
    if not p.exists():
        return None
    b = pd.read_parquet(p, columns=["trade_time", "close"])
    b["trade_time"] = pd.to_datetime(b["trade_time"])
    b["d"] = b["trade_time"].dt.normalize()
    prev = b[b["d"] < as_of]
    if prev.empty:
        return None
    last_day = prev["d"].max()
    return float(prev[prev["d"] == last_day]["close"].iloc[-1])


def _holdings(book: str) -> pd.DataFrame:
    f = Path(f"runtime/paper/forward/{book}/targets_latest.csv")
    if not f.exists():
        f = Path("runtime/paper/replay_2026/holdings_daily.csv")
    h = pd.read_csv(f)
    h["symbol"] = h["symbol"].astype(str)
    return h


def replay(args) -> int:
    book = _holdings(args.book)
    symbols = sorted(book["symbol"].unique())[: args.max_symbols or None]
    decisions, sig_counts = [], {"低A": 0, "低S": 0, "高A": 0, "高S": 0}
    D = pd.Timestamp(args.date)
    for sym in symbols:
        p = MINUTE_DIR / f"{sym}.parquet"
        if not p.exists():
            continue
        pc = _prev_close_from_minutes(sym, D)
        if pc is None or pc <= 0:
            continue
        bars = pd.read_parquet(p, columns=["trade_time", "open", "high", "low", "close", "volume", "amount"])
        bars["trade_time"] = pd.to_datetime(bars["trade_time"])
        db = bars[bars["trade_time"].dt.normalize() == D].sort_values("trade_time")
        if len(db) < 30:
            continue
        for i in range(20, len(db), 5):
            st = compute_intraday_state(db.iloc[: i + 1], pre_close=pc)
            if st:
                if st.low_signal:
                    sig_counts[st.low_signal] += 1
                if st.high_signal:
                    sig_counts[st.high_signal] += 1
        upto = db[db["trade_time"].dt.strftime("%H:%M:%S") <= args.at]
        if upto.empty:
            continue
        st = compute_intraday_state(upto, pre_close=pc)
        w = float(book[book["symbol"] == sym]["weight"].iloc[0]) if "weight" in book.columns else 0.0
        held = int(w * args.capital / max(pc, 1e-6) // 100 * 100)
        dec = decide(st, Position(total_qty=held, sellable_qty=held),
                     symbol=sym, current_time=f"{args.date} {args.at}", pre_close=pc,
                     limit_up=round(pc * 1.1, 2), limit_down=round(pc * 0.9, 2),
                     cash=args.capital * 0.2)
        if dec["action"] not in ("HOLD", "WAIT"):
            decisions.append(dec)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"dot_decisions_{args.date}.json").write_text(
        json.dumps({"date": args.date, "at": args.at, "signal_counts": sig_counts,
                    "actionable": decisions}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"date": args.date, "symbols": len(symbols),
                      "signal_counts": sig_counts, "n_actionable": len(decisions)},
                     ensure_ascii=False, indent=2))
    for d in decisions[:10]:
        print(f"  {d['symbol']} {d['action']} qty={d['qty']} @{d['limit_price']} conf={d['confidence']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay", help="offline replay one cached day")
    r.add_argument("--date", required=True)
    r.add_argument("--at", default="10:30:00", help="decision time of day (HH:MM:SS)")
    r.add_argument("--book", default="A_default")
    r.add_argument("--capital", type=float, default=1_000_000.0)
    r.add_argument("--max-symbols", type=int, default=0)
    r.add_argument("--output-dir", default="runtime/reports/daily")
    r.set_defaults(func=replay)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
