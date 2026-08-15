#!/usr/bin/env python3
"""Download daily bars for the FULL China A-share universe via AKShare/Sina.

Why Sina rather than the configured primary: measured 2026-08-15 from this host,
``push2his.eastmoney.com`` (which backs ``stock_zh_a_hist``) refuses every
connection and burns ~262s per symbol before giving up, and Tencent's
``stock_zh_a_hist_tx`` publishes volume but NO CNY turnover at all. Sina's
``stock_zh_a_daily`` supplies volume in shares and amount in CNY, and its
``amount/(volume*close)`` reconciles to a median of 0.9986.

Throughput measured on the same host: 1 worker 0.61 sym/s, 4 workers 3.70,
8 workers 4.26, 12 workers 1.52 -- Sina rate-limits above ~8, so more workers is
slower, not faster. The default is deliberately below the knee.

The run is resumable: each symbol lands in its own parquet shard and existing
shards are skipped, so an interrupted run costs only the symbols in flight.
Failures are recorded per symbol rather than dropped -- a partial universe that
does not say which names are missing is a survivorship trap.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

#: ``stock_zh_a_daily`` accepts no timeout argument, so a half-open socket blocks
#: its worker forever. A first full-universe run hung for ~2h50m on its last 4
#: symbols with no progress and no error -- the retry loop never got to run,
#: because the underlying recv never returned. A process-wide default timeout is
#: the only lever that reaches it.
SOCKET_TIMEOUT_SECONDS = 30.0

DEFAULT_OUT = Path("data/raw/ashare_daily")
#: Below the measured rate-limit knee (8); see module docstring.
DEFAULT_WORKERS = 8


def fetch_universe(ak) -> pd.DataFrame:
    """Every listed A-share, from the one universe endpoint that responds.

    ``stock_info_a_code_name`` and the SSE/SZSE listings all fail from this host
    (query.sse.com.cn unreachable, SZSE connection reset). Sina's spot snapshot
    responds and carries all three exchanges.
    """
    spot = ak.stock_zh_a_spot()
    frame = pd.DataFrame({
        "prefixed": spot["代码"].astype(str),
        "name": spot["名称"].astype(str),
    })
    frame["code"] = frame["prefixed"].str[2:]
    frame["exchange"] = frame["prefixed"].str[:2].str.upper()
    # This is a CURRENT snapshot: it lists live names only. Any research built
    # from it inherits survivorship bias unless delisted names are added back
    # from a point-in-time master.
    return frame.drop_duplicates("prefixed").reset_index(drop=True)


def fetch_one(ak, prefixed: str, start: str, end: str, adjust: str, retries: int = 3):
    last = ""
    for attempt in range(retries):
        try:
            frame = ak.stock_zh_a_daily(
                symbol=prefixed, start_date=start, end_date=end, adjust=adjust
            )
            if frame is None or frame.empty:
                return None, "empty"
            return frame, ""
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            last = f"{type(exc).__name__}: {str(exc)[:80]}"
            time.sleep(1.5 * (attempt + 1))
    return None, last


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"))
    parser.add_argument("--adjust", default="", choices=["", "qfq", "hfq"])
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="0 = whole universe")
    parser.add_argument("--only", default="", help="comma-separated symbols to fetch")
    args = parser.parse_args(argv)

    socket.setdefaulttimeout(SOCKET_TIMEOUT_SECONDS)
    import akshare as ak

    shard_dir = args.out / f"adjust={args.adjust or 'none'}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(f"akshare {ak.__version__}  window {args.start}..{args.end}  "
          f"adjust={args.adjust or 'none'}  workers={args.workers}")
    universe = fetch_universe(ak)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        universe = universe[universe["prefixed"].isin(wanted)]
    if args.limit:
        universe = universe.head(args.limit)
    print(f"universe: {len(universe):,} symbols "
          f"({universe['exchange'].value_counts().to_dict()})")

    pending = [
        row for row in universe.itertuples(index=False)
        if not (shard_dir / f"{row.prefixed}.parquet").exists()
    ]
    done_already = len(universe) - len(pending)
    print(f"already on disk: {done_already:,}   to fetch: {len(pending):,}")
    if not pending:
        print("nothing to do")
        return 0

    failures: dict[str, str] = {}
    ok = rows_total = 0
    t0 = time.time()

    def work(row):
        frame, err = fetch_one(ak, row.prefixed, args.start, args.end, args.adjust)
        if frame is None:
            return row.prefixed, 0, err
        frame = frame.copy()
        frame["symbol"] = row.prefixed
        frame["name"] = row.name
        frame.to_parquet(shard_dir / f"{row.prefixed}.parquet", index=False)
        return row.prefixed, len(frame), ""

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, row): row.prefixed for row in pending}
        for done, future in enumerate(as_completed(futures), start=1):
            symbol, n, err = future.result()
            if err:
                failures[symbol] = err
            else:
                ok += 1
                rows_total += n
            if done % 200 == 0 or done == len(pending):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0.0
                eta = (len(pending) - done) / rate / 60 if rate else 0.0
                print(f"  [{done:>5}/{len(pending)}] ok={ok} fail={len(failures)} "
                      f"rows={rows_total:,} {rate:.2f} sym/s ETA {eta:.1f} min",
                      flush=True)

    elapsed = time.time() - t0
    print(f"\nfetched {ok:,} symbols, {rows_total:,} rows in {elapsed/60:.1f} min")
    if failures:
        fail_path = args.out / "failures.json"
        fail_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False))
        print(f"FAILED {len(failures)} symbols -> {fail_path}")
        print("  a universe missing names silently is a survivorship trap; "
              "re-run to retry only these")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
