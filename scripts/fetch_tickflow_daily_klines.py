#!/usr/bin/env python3
"""Fetch TickFlow daily K-lines into a QuantAgent-compatible parquet.

This is a narrow data-capture helper for PIT validation.  It downloads only
daily K-lines for the requested symbols/date range.  TickFlow's free client can
serve historical daily K without an API key; full-service/minute endpoints are
handled elsewhere and still require credentials.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from quantagent.data.providers.base import ProviderRequest
from quantagent.data.providers.tickflow_provider import TickflowProvider


def _parse_symbols(symbols: str | None, symbols_file: Path | None) -> list[str]:
    out: list[str] = []
    if symbols:
        out.extend(s.strip() for s in symbols.split(",") if s.strip())
    if symbols_file:
        text = symbols_file.read_text(encoding="utf-8")
        out.extend(s.strip() for s in text.replace("\n", ",").split(",") if s.strip())
    return list(dict.fromkeys(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default=None, help="comma-separated canonical symbols")
    ap.add_argument("--symbols-file", type=Path, default=None)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--batch-size", type=int, default=80)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    symbols = _parse_symbols(args.symbols, args.symbols_file)
    if not symbols:
        raise SystemExit("provide --symbols or --symbols-file")
    provider = TickflowProvider(allow_network=True, allow_free_daily=True)
    frames = []
    total_batches = math.ceil(len(symbols) / args.batch_size)
    for i in range(0, len(symbols), args.batch_size):
        batch = symbols[i:i + args.batch_size]
        result = provider.daily_ohlcv(ProviderRequest(args.start_date, args.end_date, tuple(batch)))
        if not result.frame.empty:
            frames.append(result.frame)
        print(json.dumps({"batch": i // args.batch_size + 1, "total_batches": total_batches, "symbols": len(batch), "rows": len(result.frame)}, ensure_ascii=False), flush=True)
    if not frames:
        raise SystemExit("tickflow returned no daily K-line rows")
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "trade_date"]).sort_values(["trade_date", "symbol"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(json.dumps({
        "output": str(args.output),
        "rows": int(len(out)),
        "symbols": int(out["symbol"].nunique()),
        "start": str(pd.to_datetime(out["trade_date"]).min().date()),
        "end": str(pd.to_datetime(out["trade_date"]).max().date()),
        "source": "tickflow",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
