#!/usr/bin/env python3
"""Consolidate per-symbol A-share shards into one panel, and verify it.

Verification is not decoration here. The shards come from a live feed whose
sibling sources were measured shipping volume mislabelled as turnover, so the
panel is checked against economic identities before it is declared usable:

* ``amount ~= volume * close`` -- catches a lots/shares or CNY/lots unit slip,
  which is a 100x or 1e5x error that no schema check would notice.
* ``low <= open,close <= high`` -- catches row-level corruption.
* duplicate ``(symbol, trade_date)`` -- catches double-counted sessions.

Anything that fails is REPORTED, never silently dropped, and the summary states
plainly that this universe is a current-listing snapshot and therefore carries
survivorship bias until delisted names are merged in.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

SHARD_GLOB = "data/raw/ashare_daily/adjust=none/*.parquet"
OUT_PANEL = Path("data/raw/ashare_daily/panel_all.parquet")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", default=SHARD_GLOB)
    parser.add_argument("--out", type=Path, default=OUT_PANEL)
    args = parser.parse_args(argv)

    files = sorted(glob.glob(args.shards))
    print(f"consolidating {len(files):,} shards ...", flush=True)
    if not files:
        print("no shards found")
        return 1

    frames = []
    for i, path in enumerate(files, start=1):
        frames.append(pd.read_parquet(path))
        if i % 500 == 0:
            print(f"  read {i:,}/{len(files):,}", flush=True)

    panel = pd.concat(frames, ignore_index=True)
    del frames
    panel = panel.rename(columns={"date": "trade_date"})
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    print(f"\nrows={len(panel):,}  symbols={panel['symbol'].nunique():,}")
    print(f"dates {panel['trade_date'].min().date()} .. {panel['trade_date'].max().date()}")
    print(f"columns: {list(panel.columns)}")

    report: dict[str, object] = {
        "rows": int(len(panel)),
        "symbols": int(panel["symbol"].nunique()),
        "first_date": str(panel["trade_date"].min().date()),
        "last_date": str(panel["trade_date"].max().date()),
        # Stated, not buried: this is a live-listing snapshot.
        "survivorship_bias": (
            "PRESENT: universe is a current-listing snapshot; delisted names are "
            "absent and must be merged from a point-in-time master before any "
            "backtest result from this panel is believed"
        ),
    }

    print("\n--- economic identity checks ---")
    usable = panel.dropna(subset=["amount", "volume", "close"])
    usable = usable[(usable["volume"] > 0) & (usable["close"] > 0)]
    implied = usable["amount"] / (usable["volume"] * usable["close"])
    report["amount_identity_median"] = round(float(implied.median()), 6)
    off = int(((implied < 0.5) | (implied > 2.0)).sum())
    report["amount_identity_outlier_rows"] = off
    print(f"  amount/(volume*close) median = {implied.median():.6f}  (1.0 = consistent)")
    print(f"  rows outside [0.5, 2.0]      = {off:,} of {len(usable):,} "
          f"({off / max(len(usable), 1):.3%})")

    ohlc = panel.dropna(subset=["open", "high", "low", "close"])
    bad_ohlc = int((
        (ohlc["low"] > ohlc["open"]) | (ohlc["low"] > ohlc["close"])
        | (ohlc["high"] < ohlc["open"]) | (ohlc["high"] < ohlc["close"])
    ).sum())
    report["ohlc_violations"] = bad_ohlc
    print(f"  OHLC ordering violations     = {bad_ohlc:,}")

    dupes = int(panel.duplicated(["symbol", "trade_date"]).sum())
    report["duplicate_symbol_date_rows"] = dupes
    print(f"  duplicate (symbol, date)     = {dupes:,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out, index=False)
    size_mb = args.out.stat().st_size / 1e6
    report["panel_path"] = str(args.out)
    report["panel_size_mb"] = round(size_mb, 1)
    print(f"\nwrote {args.out}  ({size_mb:.1f} MB)")

    report_path = args.out.parent / "panel_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {report_path}")
    print("\nCONSOLIDATION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
