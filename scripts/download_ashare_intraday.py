#!/usr/bin/env python3
"""Download intraday (minute-bar) data for the full China A-share universe.

WHAT IS AND IS NOT AVAILABLE
----------------------------
AKShare provides **no true Level-2 order book data**. That is a property of the
upstream sources, not of this script, and no amount of endpoint-shopping changes
it. Measured 2026-08-16 from this host:

  stock_zh_a_minute      (Sina)      WORKS  minute OHLCV+amount, all periods
  stock_zh_a_hist_min_em (EastMoney) FAILS  same push2his block as the daily feed
  stock_zh_a_tick_tx_js  (Tencent)   WORKS  ~4k rows, CURRENT TRADING DAY ONLY
  stock_intraday_sina    (Sina)      WORKS  ticks >=400 shares, recent days only
  stock_bid_ask_em       (EastMoney) L1 five-level SNAPSHOT, not a series

So the deepest reproducible intraday history here is Sina minute bars, and the
closest thing to order flow is a 3-second-aggregated tick print for one day --
which is not tick-by-tick and not order book. Anything claiming L2 from these
sources would be fabricating it.

THE 1970-BAR CAP
----------------
Sina returns at most ~1970 bars per symbol regardless of period, so the period
IS the history-depth decision:

    period=1   ~5 trading days
    period=5   ~2 months     (measured: first bar 2026-06-17)
    period=15  ~6 months
    period=30  ~1 year
    period=60  ~2 years      (measured: first bar 2024-08-02)

Downloading several periods gives overlapping resolutions rather than one long
series. Pick per use: 60 for regime/holding-period work, 5 or 15 for execution.

Resumable, one parquet shard per (symbol, period). Failures are recorded, never
dropped -- a partial universe that does not name its gaps is a survivorship trap.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

#: Sina's minute endpoint takes no timeout argument; without this a half-open
#: socket blocks a worker forever. The daily downloader hung ~2h50m that way.
SOCKET_TIMEOUT_SECONDS = 30.0
DEFAULT_OUT = Path("data/raw/ashare_intraday")
DEFAULT_WORKERS = 6

NUMERIC = ("open", "high", "low", "close", "volume", "amount")


def fetch_universe(ak) -> pd.DataFrame:
    """All listed A-shares. Sina's spot snapshot is the only universe endpoint
    that responds here; it lists LIVE names only, so this inherits survivorship
    bias exactly as the daily download does."""
    spot = ak.stock_zh_a_spot()
    frame = pd.DataFrame({"prefixed": spot["代码"].astype(str),
                          "name": spot["名称"].astype(str)})
    return frame.drop_duplicates("prefixed").reset_index(drop=True)


def fetch_one(ak, prefixed: str, period: str, retries: int = 3):
    last = ""
    for attempt in range(retries):
        try:
            frame = ak.stock_zh_a_minute(symbol=prefixed, period=period, adjust="")
            if frame is None or frame.empty:
                return None, "empty"
            return frame, ""
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            last = f"{type(exc).__name__}: {str(exc)[:80]}"
            time.sleep(1.5 * (attempt + 1))
    return None, last


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="60", choices=["1", "5", "15", "30", "60"])
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="")
    args = parser.parse_args(argv)

    socket.setdefaulttimeout(SOCKET_TIMEOUT_SECONDS)
    import akshare as ak

    shard_dir = args.out / f"period={args.period}min"
    shard_dir.mkdir(parents=True, exist_ok=True)
    print(f"akshare {ak.__version__}  period={args.period}min  workers={args.workers}")

    universe = fetch_universe(ak)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        universe = universe[universe["prefixed"].isin(wanted)]
    if args.limit:
        universe = universe.head(args.limit)
    print(f"universe: {len(universe):,} symbols")

    pending = [r for r in universe.itertuples(index=False)
               if not (shard_dir / f"{r.prefixed}.parquet").exists()]
    print(f"already on disk: {len(universe) - len(pending):,}   to fetch: {len(pending):,}")
    if not pending:
        print("nothing to do")
        return 0

    failures: dict[str, str] = {}
    ok = rows_total = 0
    t0 = time.time()

    def work(row):
        frame, err = fetch_one(ak, row.prefixed, args.period)
        if frame is None:
            return row.prefixed, 0, err
        frame = frame.copy()
        # Sina returns every column as a string; leaving them so would make a
        # downstream `close > 0` compare lexicographically and silently pass.
        for col in NUMERIC:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.rename(columns={"day": "datetime"})
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame["symbol"] = row.prefixed
        frame["name"] = row.name
        frame["period_min"] = int(args.period)
        frame.to_parquet(shard_dir / f"{row.prefixed}.parquet", index=False)
        return row.prefixed, len(frame), ""

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, r): r.prefixed for r in pending}
        for done, future in enumerate(as_completed(futures), start=1):
            _, n, err = future.result()
            if err:
                failures[futures[future]] = err
            else:
                ok += 1
                rows_total += n
            if done % 200 == 0 or done == len(pending):
                el = time.time() - t0
                rate = done / el if el else 0.0
                eta = (len(pending) - done) / rate / 60 if rate else 0.0
                print(f"  [{done:>5}/{len(pending)}] ok={ok} fail={len(failures)} "
                      f"rows={rows_total:,} {rate:.2f} sym/s ETA {eta:.1f} min", flush=True)

    print(f"\nfetched {ok:,} symbols, {rows_total:,} rows in {(time.time()-t0)/60:.1f} min")
    if failures:
        path = args.out / f"failures_{args.period}min.json"
        path.write_text(json.dumps(failures, indent=2, ensure_ascii=False))
        print(f"FAILED {len(failures)} -> {path}  (re-run to retry only these)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
