#!/usr/bin/env python3
"""Forward collector: 东财 push2 per-minute individual fund flow for held names.

The blocked TickFlow depth has a free, accessible alternative: 东财's push2
``fflow/kline`` (klt=1) returns the full trading day's per-minute *cumulative*
net inflow split into 主力 / 超大单 / 大单 / 中单 / 小单 (元).  One ``lmt=1000``
request returns all 240 minutes, so a single after-close call per held name
(~50/day) stays well under the ban thresholds (>200 req/min triggers a ban).

This is current-day only (no history), so an intraday order-flow do-T model must
be trained on FORWARD-collected data.  Run this once after 15:00 CST each trading
day to accumulate:

    runtime/data/v7/silver/fundflow_minute/{YYYY-MM-DD}.parquet

Columns: symbol, trade_date, trade_time, main_net, super_net, large_net,
mid_net, small_net (cumulative 元), + amount-free derived features are built at
training time by quantagent.execution.intraday_features.merge_fundflow_features.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path("runtime/data/v7/silver/fundflow_minute")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
URL = "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get"


def _code(symbol: str) -> str:
    return str(symbol).split(".")[0]


def _secid(code: str) -> str:
    return f"1.{code}" if code.startswith(("6", "5", "11")) else f"0.{code}"


def fetch_minute_fundflow(code: str, *, lmt: int = 1000, timeout: float = 10.0) -> pd.DataFrame:
    params = {"secid": _secid(code), "klt": 1, "lmt": lmt,
              "fields1": "f1,f2,f3,f7", "fields2": "f51,f52,f53,f54,f55,f56,f57"}
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/",
               "Origin": "https://quote.eastmoney.com"}
    try:
        r = requests.get(URL, params=params, headers=headers, timeout=timeout)
        klines = r.json().get("data", {}).get("klines", [])
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    rows = []
    for line in klines:
        p = line.split(",")
        if len(p) >= 6:
            rows.append({"trade_time": p[0], "main_net": float(p[1]), "small_net": float(p[2]),
                         "mid_net": float(p[3]), "large_net": float(p[4]), "super_net": float(p[5])})
    return pd.DataFrame(rows)


def _load_symbols(args) -> list[str]:
    if args.book_csv and Path(args.book_csv).exists():
        df = pd.read_csv(args.book_csv)
        if "symbol" in df.columns:
            return sorted(df["symbol"].astype(str).unique())
    if args.symbols_file and Path(args.symbols_file).exists():
        txt = Path(args.symbols_file).read_text(encoding="utf-8")
        return [t.strip() for t in txt.replace(",", "\n").splitlines() if t.strip()]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book-csv", default="runtime/paper/forward/A_default/targets_latest.csv")
    ap.add_argument("--symbols-file", default="")
    ap.add_argument("--sleep", type=float, default=0.30, help="seconds between symbols (rate-limit safety)")
    ap.add_argument("--lmt", type=int, default=1000)
    args = ap.parse_args()

    symbols = _load_symbols(args)
    if not symbols:
        raise SystemExit("no symbols (provide --book-csv or --symbols-file)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    ok = 0
    for sym in symbols:
        df = fetch_minute_fundflow(_code(sym), lmt=args.lmt)
        if not df.empty:
            df["symbol"] = sym
            df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
            df["trade_date"] = df["trade_time"].dt.normalize()
            frames.append(df)
            ok += 1
        time.sleep(args.sleep)
    if not frames:
        print("no fund-flow rows collected (market closed / empty / banned?)")
        return 0
    out = pd.concat(frames, ignore_index=True)
    day = out["trade_date"].dropna().max()
    day_str = pd.Timestamp(day).strftime("%Y-%m-%d") if pd.notna(day) else pd.Timestamp.now().strftime("%Y-%m-%d")
    path = OUT_DIR / f"{day_str}.parquet"
    if path.exists():
        prev = pd.read_parquet(path)
        out = pd.concat([prev, out], ignore_index=True).drop_duplicates(["symbol", "trade_time"], keep="last")
    out.to_parquet(path, index=False)
    print(f"collected {ok}/{len(symbols)} names, {len(out)} minute rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
