#!/usr/bin/env python3
"""Probe every public AKShare endpoint and record what actually works here.

Motivation: which endpoints are reachable is a property of THIS host's network,
not of the library. EastMoney's push2his is blocked here while Sina and Tencent
respond, and that fact alone reshapes what a dataset can contain. Guessing which
of ~1100 functions are usable, or testing a hand-picked handful, produces a data
plan built on assumption. This measures all of them.

Each result records status, row count, columns, elapsed time and the exact error,
appended to JSONL as it goes so an interrupted run keeps everything already
learned. Arguments are auto-filled from parameter NAMES using per-family
conventions (Sina and Tencent want a `sh600519` style prefix, EastMoney wants a
bare `600519`), because a wrong-format symbol produces a failure that looks like
an outage and would poison the capability matrix.

A FAIL here means "not usable from this host today" -- it does NOT mean the
endpoint is broken upstream, and it must not be quoted as such.
"""

from __future__ import annotations

import argparse
import inspect
import json
import socket
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")

SOCKET_TIMEOUT = 25.0
OUT = Path("data/raw/akshare_probe")

#: Symbol formats differ by upstream, and the wrong one fails like an outage.
PREFIXED = ("sina", "_tx", "tick", "minute", "_daily", "zh_a_minute")
TODAY = "20260814"
RECENT_START, RECENT_END = "20260701", "20260814"

DEFAULTS: dict[str, object] = {
    "period": "daily",
    "adjust": "",
    "start_date": RECENT_START,
    "end_date": RECENT_END,
    "date": TODAY,
    "trade_date": TODAY,
    "start_year": "2024",
    "end_year": "2026",
    "start_month": "202601",
    "end_month": "202608",
    "year": "2025",
    "indicator": None,          # filled from the docstring/annotation when possible
    "market": "沪深A股",
    "flag": None,
    "page": "1",
    "num": "10",
}


def guess_symbol(name: str) -> str:
    return "sh600519" if any(k in name for k in PREFIXED) else "600519"


def build_kwargs(fn, name: str) -> dict | None:
    """Fill required params, or return None if we cannot honestly supply them."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    kwargs: dict[str, object] = {}
    for pname, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            continue  # optional: let the library's own default stand
        if pname in ("symbol", "code", "stock"):
            kwargs[pname] = guess_symbol(name)
        elif pname in DEFAULTS and DEFAULTS[pname] is not None:
            kwargs[pname] = DEFAULTS[pname]
        else:
            return None  # a required param we would only be guessing at
    return kwargs


def probe(name: str) -> dict:
    import akshare as ak

    fn = getattr(ak, name, None)
    record: dict[str, object] = {"name": name}
    if fn is None or not callable(fn):
        return {**record, "status": "missing"}
    kwargs = build_kwargs(fn, name)
    if kwargs is None:
        return {**record, "status": "skipped_needs_args"}

    record["kwargs"] = {k: str(v) for k, v in kwargs.items()}
    t0 = time.time()
    try:
        out = fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 - the error IS the finding
        return {**record, "status": "fail", "elapsed_s": round(time.time() - t0, 1),
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    elapsed = round(time.time() - t0, 1)
    try:
        rows = len(out)
        cols = [str(c) for c in getattr(out, "columns", [])][:40]
    except Exception:
        return {**record, "status": "ok_nonframe", "elapsed_s": elapsed,
                "type": type(out).__name__}
    if rows == 0:
        return {**record, "status": "empty", "elapsed_s": elapsed}
    return {**record, "status": "ok", "rows": int(rows), "cols": cols,
            "elapsed_s": elapsed}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prefix", default="", help="only probe names starting with this")
    args = parser.parse_args(argv)

    socket.setdefaulttimeout(SOCKET_TIMEOUT)
    import akshare as ak

    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / "probe_results.jsonl"

    done: set[str] = set()
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["name"])
            except Exception:
                pass

    names = sorted(n for n in dir(ak)
                   if not n.startswith("_") and callable(getattr(ak, n, None)))
    if args.prefix:
        names = [n for n in names if n.startswith(args.prefix)]
    pending = [n for n in names if n not in done]
    if args.limit:
        pending = pending[: args.limit]

    print(f"akshare {ak.__version__}: {len(names):,} public functions, "
          f"{len(done):,} already probed, {len(pending):,} to go", flush=True)

    counts: dict[str, int] = {}
    t0 = time.time()
    with result_path.open("a", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(probe, n): n for n in pending}
            for i, future in enumerate(as_completed(futures), start=1):
                try:
                    rec = future.result()
                except Exception as exc:  # noqa: BLE001
                    rec = {"name": futures[future], "status": "probe_error",
                           "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sink.flush()
                if i % 25 == 0 or i == len(pending):
                    el = time.time() - t0
                    rate = i / el if el else 0
                    print(f"  [{i:>4}/{len(pending)}] {rate:.2f} fn/s "
                          f"ETA {(len(pending)-i)/rate/60 if rate else 0:.1f} min  {counts}",
                          flush=True)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {result_path}")
    print(f"summary: {counts}")
    print("\nNOTE: 'fail' means not usable from THIS host today (network/auth/"
          "arg-shape), not that the endpoint is broken upstream.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
