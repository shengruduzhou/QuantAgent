#!/usr/bin/env python3
"""Incremental 1-minute bar history fetcher (tickflow paid API).

Depth probe 2026-06-12: `tf.klines.get(symbol, period="1m", count<=2000,
end_time=epoch_ms)` serves ~1 year of minute history (2025-06 reachable,
2024-06 not). This script pages backward per symbol and maintains a local
parquet cache so 做T / auction / microstructure backtests are reproducible
offline:

    runtime/data/v7/silver/minute_bars/{symbol}.parquet

Re-runs are incremental: only missing head/tail ranges are fetched.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("runtime/data/v7/silver/minute_bars")
PAGE = 2000  # max bars per call observed working


def _client():
    from dotenv import load_dotenv

    load_dotenv(".env", override=False)
    import tickflow

    return tickflow.TickFlow(api_key=os.environ["TICKFLOW_API_KEY"],
                             base_url=os.environ.get("TICKFLOW_API_ENDPOINT") or None)


def fetch_symbol(tf, symbol: str, start: pd.Timestamp, end: pd.Timestamp,
                 *, sleep_s: float = 0.25, max_calls: int = 60) -> pd.DataFrame:
    """Page backward from ``end`` until ``start`` (or history exhausted)."""
    frames: list[pd.DataFrame] = []
    cursor = end
    for _ in range(max_calls):
        try:
            k = tf.klines.get(symbol, period="1m", count=PAGE,
                              end_time=int(cursor.timestamp() * 1000), as_dataframe=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{symbol}] fetch err at {cursor}: {type(exc).__name__}: {str(exc)[:80]}")
            time.sleep(2.0)
            continue
        if k is None or len(k) == 0:
            break
        k = k.copy()
        k["trade_time"] = pd.to_datetime(k["trade_time"])
        frames.append(k)
        oldest = k["trade_time"].min()
        if oldest <= start or len(k) < PAGE:
            break
        cursor = oldest - pd.Timedelta(minutes=1)
        time.sleep(sleep_s)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["trade_time"])
    out = out[(out["trade_time"] >= start) & (out["trade_time"] <= end)]
    out["symbol"] = symbol
    return out.sort_values("trade_time").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols-file", help="text/csv file with one symbol per line (or comma-separated)")
    ap.add_argument("--symbols", default="", help="comma-separated symbols")
    ap.add_argument("--holdings-csv", help="holdings_daily.csv from paper replay (symbol column)")
    ap.add_argument("--start", default="2025-06-12")
    ap.add_argument("--end", default=None, help="default: now")
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="cap number of symbols (0 = all)")
    args = ap.parse_args()

    symbols: list[str] = []
    if args.symbols:
        symbols += [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.symbols_file:
        text = Path(args.symbols_file).read_text(encoding="utf-8")
        for token in text.replace(",", "\n").splitlines():
            if token.strip():
                symbols.append(token.strip())
    if args.holdings_csv:
        h = pd.read_csv(args.holdings_csv)
        symbols += sorted(h["symbol"].astype(str).unique())
    symbols = list(dict.fromkeys(symbols))
    if args.limit:
        symbols = symbols[: args.limit]
    if not symbols:
        raise SystemExit("no symbols given")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tf = _client()

    done = skipped = failed = 0
    for i, sym in enumerate(symbols):
        path = CACHE_DIR / f"{sym}.parquet"
        have_min = have_max = None
        existing = None
        if path.exists():
            existing = pd.read_parquet(path)
            if len(existing):
                have_min, have_max = existing["trade_time"].min(), existing["trade_time"].max()
        # determine missing ranges (head extension + tail extension)
        ranges = []
        if existing is None or have_min is None:
            ranges = [(start, end)]
        else:
            if start < have_min - pd.Timedelta(hours=1):
                ranges.append((start, have_min - pd.Timedelta(minutes=1)))
            if end > have_max + pd.Timedelta(hours=1):
                ranges.append((have_max + pd.Timedelta(minutes=1), end))
        if not ranges:
            skipped += 1
            continue
        pieces = [existing] if existing is not None and len(existing) else []
        ok = True
        for lo, hi in ranges:
            df = fetch_symbol(tf, sym, lo, hi, sleep_s=args.sleep)
            if len(df) == 0 and existing is None:
                ok = False
            if len(df):
                pieces.append(df)
        if not pieces:
            failed += 1
            print(f"[{i+1}/{len(symbols)}] {sym}: NO DATA")
            continue
        merged = pd.concat(pieces, ignore_index=True).drop_duplicates(subset=["trade_time"])
        merged = merged.sort_values("trade_time").reset_index(drop=True)
        merged.to_parquet(path, index=False)
        done += 1
        print(f"[{i+1}/{len(symbols)}] {sym}: {len(merged)} bars "
              f"({merged['trade_time'].min()} .. {merged['trade_time'].max()}) {'OK' if ok else 'PARTIAL'}")
    print(json.dumps({"symbols": len(symbols), "fetched": done, "cached_already": skipped,
                      "no_data": failed, "cache_dir": str(CACHE_DIR)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
