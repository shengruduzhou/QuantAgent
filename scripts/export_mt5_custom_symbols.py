#!/usr/bin/env python3
"""Export journalled A-share events into MT5 custom-symbol import bundles.

One-way by design: the journal feeds MT5, never the reverse. Each bundle
carries a manifest hashing both the source and the exported frame, so a
terminal-side row count can later be checked against a number recorded before
the data left this process.

    python scripts/export_mt5_custom_symbols.py \
        --trade-date 2026-07-24 --output runtime/data/mt5_custom_symbols
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from quantagent.data.microstructure import contracts as mc  # noqa: E402
from quantagent.data.microstructure.store import RawEventStore  # noqa: E402
from quantagent.mt5 import custom_symbol_bridge as bridge  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "runtime" / "data" / "u0" / "panel" / "daily_bars_raw.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", default="runtime/data/market_events")
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--output", default="runtime/data/mt5_custom_symbols")
    parser.add_argument("--symbols", default="", help="comma-separated subset")
    parser.add_argument("--bar-history-days", type=int, default=250)
    args = parser.parse_args()

    store = RawEventStore(args.journal)
    events = store.read(family=mc.FAMILY_TRADE, trade_date=args.trade_date)
    if events.empty:
        print(json.dumps({"error": "no journalled events for that date"}, indent=2))
        return 1

    wanted = {s.strip() for s in args.symbols.split(",") if s.strip()}
    symbols = sorted(set(events["symbol"]) & wanted) if wanted else sorted(set(events["symbol"]))

    panel = pd.read_parquet(
        PANEL, columns=["symbol", "trade_date", "open", "high", "low", "close", "volume"]
    )

    manifests = []
    for symbol in symbols:
        symbol_events = events[events["symbol"] == symbol]
        classes = sorted(set(symbol_events["data_class"].dropna().astype(str)))
        history = (
            panel[panel["symbol"] == symbol]
            .sort_values("trade_date")
            .tail(args.bar_history_days)
        )
        manifest = bridge.build_import_plan(
            symbol,
            events=symbol_events,
            bars=history,
            origin_data_class=classes[0] if len(classes) == 1 else mc.UNKNOWN_SEMANTICS,
            output_dir=args.output,
        )
        manifests.append(manifest.to_dict())

    summary = {
        "trade_date": args.trade_date,
        "symbols_exported": len(manifests),
        "total_ticks": sum(m["tick_rows"] for m in manifests),
        "total_bars": sum(m["bar_rows"] for m in manifests),
        "imported_data_class": mc.CUSTOM_SYMBOL_REPLAY,
        "distinct_warnings": sorted({w for m in manifests for w in m["warnings"]}),
        "bundles": [
            {"custom_symbol": m["custom_symbol"], "canonical": m["canonical_symbol"],
             "board": m["board"], "ticks": m["tick_rows"], "bars": m["bar_rows"],
             "source_hash": m["source_content_hash"]}
            for m in manifests
        ],
    }
    index = Path(args.output) / "export_index.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
