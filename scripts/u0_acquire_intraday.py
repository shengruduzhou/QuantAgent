#!/usr/bin/env python3
"""Intraday (minute) bar acquisition for a representative A-share cohort.

No provider available in this runtime serves deep minute history for the whole
universe: TickFlow's intraday entitlement is not on this subscription
(``PermissionError: 无日内分时查询权限``) and the public feeds keep only a rolling
window. What this command does is therefore explicit about its scope — it
acquires the vendor's full rolling window for a board-stratified cohort and
persists it with provenance, so the intraday layer is real, reconcilable against
the daily panel, and honest about depth.

Providers:
  ``tencent``   5-minute bars, ~320-bar rolling window, every board
  ``eastmoney`` 1-minute trend bars for the most recent sessions (IP-throttled)

Outputs (runtime/data/u0/intraday/):
  minute_bars.parquet, minute_ledger.csv, intraday_manifest.json

Usage:
  AI_quant_venv/bin/python3 scripts/u0_acquire_intraday.py --allow-network \\
      --per-board 40 --frequency 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quantagent.data.ashare.env import load_repo_env  # noqa: E402
from quantagent.data.ashare.http import HttpClient  # noqa: E402
from quantagent.data.ashare.sources import EastmoneySource, TencentSource  # noqa: E402

U0 = REPO / "runtime/data/u0"
OUT = U0 / "intraday"
MASTER = U0 / "security_master.parquet"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def select_cohort(master: pd.DataFrame, per_board: int) -> pd.DataFrame:
    """Board-stratified cohort: the largest listed names plus recent listings."""
    listed = master[master["status"] == "listed"].copy()
    listed["float_shares"] = pd.to_numeric(listed["float_shares"], errors="coerce")
    chunks = []
    for board, group in listed.groupby("board"):
        head = group.sort_values("float_shares", ascending=False).head(max(1, per_board - 5))
        recent = group.sort_values("listing_date", ascending=False).head(5)
        chunks.append(pd.concat([head, recent]).drop_duplicates("symbol"))
    return pd.concat(chunks, ignore_index=True) if chunks else listed.head(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--per-board", type=int, default=40)
    parser.add_argument("--frequency", type=int, default=5, choices=[1, 5, 15, 30, 60])
    parser.add_argument("--count", type=int, default=320)
    parser.add_argument("--providers", default="tencent,eastmoney")
    parser.add_argument("--max-minutes", type=float, default=45.0)
    args = parser.parse_args()
    if not args.allow_network:
        print("refusing to acquire: --allow-network was not confirmed")
        return 2
    load_repo_env()
    if not MASTER.exists():
        print("missing security master — run scripts/u0_security_master.py first")
        return 3

    OUT.mkdir(parents=True, exist_ok=True)
    master = pd.read_parquet(MASTER)
    cohort = select_cohort(master, args.per_board)
    client = HttpClient(timeout=20, max_attempts=3)
    tencent, eastmoney = TencentSource(client), EastmoneySource(client)
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    started = time.monotonic()
    deadline = started + args.max_minutes * 60
    frames, ledger = [], []
    for index, row in enumerate(cohort.itertuples()):
        if time.monotonic() > deadline:
            print(f"time budget reached after {index} symbols", flush=True)
            break
        for provider in providers:
            if provider == "tencent":
                result = tencent.minute_bars(row.symbol, args.frequency, args.count)
            elif provider == "eastmoney":
                result = eastmoney.minute_trends(row.symbol, days=1)
            else:
                continue
            ledger.append({"symbol": row.symbol, "board": row.board, "provider": provider,
                           "rows": result.rows, "retry_class": result.retry_class,
                           "detail": (result.error or "")[:160], "recorded_at": _now()})
            if result.rows:
                frames.append(result.frame.assign(serving_provider=provider))
                break
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(cohort)} · frames={len(frames)} "
                  f"{time.monotonic() - started:.0f}s", flush=True)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined["bar_time"] = pd.to_datetime(combined["bar_time"])
        combined = combined.drop_duplicates(["symbol", "bar_time", "frequency"])
        combined.to_parquet(OUT / "minute_bars.parquet", index=False)
    ledger_frame = pd.DataFrame(ledger)
    ledger_frame.to_csv(OUT / "minute_ledger.csv", index=False)

    sessions = 0
    if not combined.empty:
        sessions = int(combined.assign(d=combined["bar_time"].dt.normalize())
                       .groupby(["symbol", "d"]).ngroups)
    manifest = {
        "generated": _now(),
        "frequency_minutes": args.frequency,
        "cohort_symbols": int(len(cohort)),
        "symbols_with_bars": int(combined["symbol"].nunique()) if not combined.empty else 0,
        "rows": int(len(combined)),
        "symbol_sessions": sessions,
        "bar_time_range": [str(combined["bar_time"].min()), str(combined["bar_time"].max())]
        if not combined.empty else None,
        "by_board": cohort["board"].value_counts().to_dict(),
        "serving_providers": combined["serving_provider"].value_counts().to_dict()
        if not combined.empty else {},
        "retry_classes": ledger_frame["retry_class"].value_counts().to_dict()
        if len(ledger_frame) else {},
        "depth_limitation": ("public feeds serve a rolling intraday window only; TickFlow intraday "
                             "is UNAUTHORIZED on this subscription, so deep minute history for the "
                             "full universe has no lawful route in this runtime"),
        "volume_unit": "shares", "price_unit": "CNY", "timezone": "Asia/Shanghai",
    }
    (OUT / "intraday_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
