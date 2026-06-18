#!/usr/bin/env python3
"""Fetch 东财 push2his DAILY individual fund flow (120-day history) for a universe.

Unlike the minute fund flow (current-day only), the daily ``fflow/daykline``
endpoint serves ~120 trading days of history → this IS backtestable today as a
cross-sectional daily factor (主力/超大单/大单 net inflow, 元).

Output: runtime/data/v7/silver/fundflow_daily/fundflow_daily.parquet
        columns: symbol, date, main_net, super_net, large_net, mid_net, small_net
Incremental: merges with any existing file (dedup on symbol+date).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path("runtime/data/v7/silver/fundflow_daily/fundflow_daily.parquet")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
URL = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"


def _code(sym: str) -> str:
    return str(sym).split(".")[0]


def _secid(code: str) -> str:
    return f"1.{code}" if code.startswith(("6", "5", "11")) else f"0.{code}"


def fetch_daily(code: str, *, lmt: int = 120, timeout: float = 15.0, retries: int = 2) -> pd.DataFrame:
    params = {"secid": _secid(code), "fields1": "f1,f2,f3,f7",
              "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65", "lmt": str(lmt)}
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/",
               "Origin": "https://quote.eastmoney.com"}
    for attempt in range(retries + 1):
        try:
            r = requests.get(URL, params=params, headers=headers, timeout=timeout)
            klines = r.json().get("data", {}).get("klines", [])
            if klines:
                rows = []
                for line in klines:
                    p = line.split(",")
                    if len(p) >= 6:
                        def f(x):
                            return float(x) if x not in ("-", "") else 0.0
                        rows.append({"date": p[0], "main_net": f(p[1]), "small_net": f(p[2]),
                                     "mid_net": f(p[3]), "large_net": f(p[4]), "super_net": f(p[5])})
                return pd.DataFrame(rows)
        except Exception:  # noqa: BLE001
            pass
        if attempt < retries:
            time.sleep(2.0 * (attempt + 1))
    return pd.DataFrame()


def _load_symbols(args) -> list[str]:
    if args.symbols_file and Path(args.symbols_file).exists():
        txt = Path(args.symbols_file).read_text(encoding="utf-8")
        syms = [t.strip() for t in txt.replace(",", "\n").splitlines() if t.strip()]
    elif args.book_csv and Path(args.book_csv).exists():
        syms = sorted(pd.read_csv(args.book_csv)["symbol"].astype(str).unique())
    else:
        syms = []
    if args.max_symbols:
        syms = syms[: args.max_symbols]
    return syms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols-file", default="runtime/tmp/minute_cache_symbols.txt")
    ap.add_argument("--book-csv", default="")
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.4)
    args = ap.parse_args()

    symbols = _load_symbols(args)
    if not symbols:
        raise SystemExit("no symbols")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames, ok, empty = [], 0, 0
    for i, sym in enumerate(symbols):
        df = fetch_daily(_code(sym))
        if df.empty:
            empty += 1
        else:
            df["symbol"] = sym
            frames.append(df)
            ok += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(symbols)} (ok={ok} empty={empty})", flush=True)
        time.sleep(args.sleep)
    if not frames:
        print(f"no rows collected (ok=0 empty={empty}); push2his likely blocked — retry later/other network")
        return 1
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        out = pd.concat([prev, out], ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")
    out.to_parquet(OUT, index=False)
    print(f"collected {ok}/{len(symbols)} symbols, {len(out)} rows, "
          f"{out['date'].min()}..{out['date'].max()} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
