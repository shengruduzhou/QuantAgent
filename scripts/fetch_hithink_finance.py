#!/usr/bin/env python3
"""Acquire HiThink Finance / Fuyao data into Runtime-safe files.

Modes
-----
daily
    Per-symbol daily history for bounded repair/small universes.  For full A-share
    history use ``dump`` instead of issuing thousands of HTTP requests.
dump
    Sign and immediately download one full-market Parquet dump.  The short-lived
    presigned URL is never printed or persisted.
capability
    Call one reviewed public REST capability from ``CAPABILITY_PATHS`` and write
    the returned ``data`` object as JSON.  This covers financials, valuations,
    indices, funds, hot lists, dragon-tiger and other public reference data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quantagent.data.providers.base import ProviderRequest
from quantagent.data.providers.hithink_finance_provider import (
    CAPABILITY_PATHS,
    HithinkFinanceProvider,
    MARKET_DUMP_PATHS,
)


def _symbols(value: str | None, path: Path | None) -> tuple[str, ...]:
    items: list[str] = []
    if value:
        items.extend(token.strip().upper() for token in value.split(",") if token.strip())
    if path:
        text = path.read_text(encoding="utf-8")
        items.extend(token.strip().upper() for token in text.replace("\n", ",").split(",") if token.strip())
    return tuple(dict.fromkeys(items))


def _json_object(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("--params-json must decode to an object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("daily", "dump", "capability"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--symbols-file", type=Path, default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--adjust", choices=("none", "forward"), default="none")
    parser.add_argument("--dump-kind", choices=tuple(MARKET_DUMP_PATHS), default="daily-k")
    parser.add_argument("--capability", choices=tuple(CAPABILITY_PATHS), default="prices_snapshot")
    parser.add_argument("--params-json", type=_json_object, default={})
    args = parser.parse_args()

    provider = HithinkFinanceProvider(allow_network=True)
    try:
        if args.mode == "daily":
            symbols = _symbols(args.symbols, args.symbols_file)
            if not symbols:
                raise SystemExit("daily mode requires --symbols or --symbols-file")
            if not args.start_date or not args.end_date:
                raise SystemExit("daily mode requires --start-date and --end-date")
            request = ProviderRequest(args.start_date, args.end_date, symbols)
            result = provider.adjusted_prices(request) if args.adjust == "forward" else provider.daily_ohlcv(request)
            if result.frame.empty:
                raise SystemExit("HiThink Finance returned no daily rows")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.frame.to_parquet(args.output, index=False)
            summary = {
                "mode": "daily",
                "output": str(args.output),
                "source": result.source,
                "adjust": args.adjust,
                "rows": int(len(result.frame)),
                "symbols": int(result.frame["symbol"].nunique()),
                "start": str(pd.to_datetime(result.frame["trade_date"]).min().date()),
                "end": str(pd.to_datetime(result.frame["trade_date"]).max().date()),
                "pointInTime": result.point_in_time,
            }
        elif args.mode == "dump":
            path = provider.download_market_dump(args.dump_kind, args.output)
            if not path.is_file() or path.stat().st_size <= 0:
                raise SystemExit("market dump download produced no file")
            summary = {
                "mode": "dump",
                "dumpKind": args.dump_kind,
                "output": str(path),
                "bytes": int(path.stat().st_size),
                "source": "hithink_finance",
                "presignedUrlPersisted": False,
            }
        else:
            params = dict(args.params_json)
            symbols = _symbols(args.symbols, args.symbols_file)
            if symbols and "thscodes" not in params and "thscode" not in params:
                params["thscodes"] = ",".join(symbols)
            payload = provider.capability(args.capability, **params)
            _write_json(args.output, payload)
            items = payload.get("item") if isinstance(payload, dict) else None
            summary = {
                "mode": "capability",
                "capability": args.capability,
                "output": str(args.output),
                "items": len(items) if isinstance(items, list) else None,
                "source": "hithink_finance",
            }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        provider.close()


if __name__ == "__main__":
    raise SystemExit(main())
