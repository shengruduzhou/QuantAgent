#!/usr/bin/env python3
"""Produce a NO_TRADE-default intraday Do-T plan for the held book.

Integration entry point: given the held book (forward targets), a trading day's
1-minute bars, and trained EV models, emit one do-T intent per held name.  The
default is NO_TRADE; an actionable SELL_HIGH / BUY_LOW / BUY_BACK / SELL_AFTER_BUY
intent is produced only when a calibrated positive-EV, T+1-legal round trip
clears the dynamic cost / probability / risk gates.  Only carried (sellable)
shares are ever touched -- today's buys are never resold the same session.

Replaces the old parametric ``dot_watchlist`` in ``forward_book_update.py`` with
a cost-sensitive, model-driven, T+1-correct overlay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quantagent.research.intraday_dot_ev_backtest import EVBacktestConfig, plan_book_dot
from quantagent.training.do_t_models import load_models


def _load_book(book_csv: str) -> pd.DataFrame:
    df = pd.read_csv(book_csv)
    if "symbol" not in df.columns:
        raise SystemExit(f"book csv {book_csv} has no 'symbol' column")
    keep = ["symbol"] + [c for c in ("weight", "shares") if c in df.columns]
    df = df[keep].copy()
    df["symbol"] = df["symbol"].astype(str)
    if "weight" in df.columns:
        df = df[pd.to_numeric(df["weight"], errors="coerce").fillna(0.0) > 0]
    return df.drop_duplicates("symbol").reset_index(drop=True)


def _load_day_bars(minute_dir: str, symbols: list[str], date: pd.Timestamp) -> dict[str, pd.DataFrame]:
    mdir = Path(minute_dir)
    out: dict[str, pd.DataFrame] = {}
    cols = ["symbol", "trade_time", "open", "high", "low", "close", "volume", "amount"]
    for sym in symbols:
        p = mdir / f"{sym}.parquet"
        if not p.exists():
            continue
        b = pd.read_parquet(p, columns=cols)
        b["trade_time"] = pd.to_datetime(b["trade_time"], errors="coerce")
        b = b[b["trade_time"].dt.normalize() == date]
        if not b.empty:
            out[sym] = b.sort_values("trade_time").reset_index(drop=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book-csv", default="runtime/paper/forward/A_default/targets_latest.csv")
    ap.add_argument("--models", default="runtime/reports/intraday_dot_ev_full/do_t_models.joblib")
    ap.add_argument("--minute-dir", default="runtime/data/v7/silver/minute_bars")
    ap.add_argument("--date", required=True, help="trading day YYYY-MM-DD for the minute bars")
    ap.add_argument("--as-of-minute", type=int, default=-1, help="-1 = latest bar (full day replay/EOD)")
    ap.add_argument("--order-notional-yuan", type=float, default=100_000.0)
    ap.add_argument("--maker-only", action="store_true")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    book = _load_book(args.book_csv)
    date = pd.Timestamp(args.date).normalize()
    bars = _load_day_bars(args.minute_dir, book["symbol"].tolist(), date)
    models = load_models(args.models)
    cfg = EVBacktestConfig(order_notional_yuan=args.order_notional_yuan)
    if args.maker_only:
        cfg = EVBacktestConfig(order_notional_yuan=args.order_notional_yuan,
                               slippage_bps=2.0, spread_bps=2.0, commission_rate=0.0001)

    plan = plan_book_dot(
        held=book, minute_bars_by_symbol=bars, models=models, cfg=cfg,
        as_of_minute=None if args.as_of_minute < 0 else args.as_of_minute,
    )
    actionable = plan[plan["action"] != "NO_TRADE"] if not plan.empty else plan
    summary = {
        "date": str(date.date()),
        "held_names": int(len(book)),
        "names_with_bars": int(len(bars)),
        "no_trade": int((plan["action"] == "NO_TRADE").sum()) if not plan.empty else 0,
        "actionable": int(len(actionable)),
        "actions": actionable["action"].value_counts().to_dict() if not actionable.empty else {},
    }
    out = args.output or f"runtime/paper/forward/dot_ev_plan_{date.date()}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out, index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"plan -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
