#!/usr/bin/env python3
"""Acquire A-share tick events, journal them immutably, then prove them.

One command runs the whole evidence chain for a symbol-day cohort:

    fetch -> canonical frame -> immutable journal -> integrity checks
          -> reconciliation against the verified U0 daily panel -> report

The cohort matters. A reconciliation over one liquid blue chip proves nothing
about coverage, so ``--cohort board-spread`` selects high- and low-turnover
names from every board that has a bar on the requested date, and the report
records exactly which securities were examined.

    python scripts/acquire_ashare_ticks.py \
        --trade-date 2026-07-24 --cohort board-spread \
        --output runtime/data/market_events
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from quantagent.data.ashare.http import HttpClient  # noqa: E402
from quantagent.data.microstructure import contracts as mc  # noqa: E402
from quantagent.data.microstructure import integrity, reconcile  # noqa: E402
from quantagent.data.microstructure.public_tick_sources import (  # noqa: E402
    TencentTickDetail,
)
from quantagent.data.microstructure.store import RawEventStore  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PANEL = REPO / "runtime" / "data" / "u0" / "panel" / "daily_bars_raw.parquet"
MASTER = REPO / "runtime" / "data" / "u0" / "security_master.parquet"


def select_cohort(trade_date: str, *, mode: str, limit_per_board: int = 2) -> pd.DataFrame:
    """Choose the symbols to acquire, and record why each was chosen."""
    master = pd.read_parquet(MASTER)
    panel = pd.read_parquet(
        PANEL, columns=["symbol", "trade_date", "close", "volume", "amount"]
    )
    day = panel[panel["trade_date"] == trade_date]
    merged = master.merge(day, on="symbol", how="inner")
    if merged.empty:
        raise SystemExit(f"no U0 panel rows for {trade_date}; nothing to reconcile against")

    if mode == "board-spread":
        picks: list[dict[str, object]] = []
        for board, group in merged.groupby("board"):
            ranked = group.dropna(subset=["amount"]).sort_values("amount")
            if ranked.empty:
                ranked = group
                picks.append({"symbol": ranked.iloc[0]["symbol"], "board": board,
                              "reason": "only security on this board with a bar; "
                                        "turnover not published"})
                continue
            picks.append({"symbol": ranked.iloc[-1]["symbol"], "board": board,
                          "reason": "highest turnover on board"})
            if len(ranked) > 1 and limit_per_board > 1:
                picks.append({"symbol": ranked.iloc[0]["symbol"], "board": board,
                              "reason": "lowest turnover on board (liquidity floor)"})
        st = merged[merged.get("current_st", False) == True]  # noqa: E712
        if len(st):
            worst = st.dropna(subset=["amount"]).sort_values("amount")
            if len(worst):
                picks.append({"symbol": worst.iloc[-1]["symbol"], "board": "ST",
                              "reason": "ST name: narrower price limit regime"})
        return pd.DataFrame(picks).drop_duplicates(subset=["symbol"])

    raise SystemExit(f"unknown cohort mode {mode!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--cohort", default="board-spread")
    parser.add_argument("--symbols", default="", help="explicit comma-separated override")
    parser.add_argument("--output", default="runtime/data/market_events")
    parser.add_argument("--report", default="runtime/data/market_events/_reports")
    parser.add_argument("--max-pages", type=int, default=200)
    args = parser.parse_args()

    if args.symbols:
        cohort = pd.DataFrame([
            {"symbol": s.strip(), "board": "explicit", "reason": "caller supplied"}
            for s in args.symbols.split(",") if s.strip()
        ])
    else:
        cohort = select_cohort(args.trade_date, mode=args.cohort)

    store = RawEventStore(args.output)
    client = HttpClient()
    source = TencentTickDetail(client)

    panel = pd.read_parquet(
        PANEL, columns=["symbol", "trade_date", "open", "high", "low",
                        "close", "volume", "amount"]
    )
    panel_day = panel[panel["trade_date"] == args.trade_date]

    acquired: list[pd.DataFrame] = []
    fetch_log: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []

    for record in cohort.to_dict("records"):
        symbol = str(record["symbol"])
        frame, outcome = source.fetch(symbol, args.trade_date, max_pages=args.max_pages)
        fetch_log.append({
            "symbol": symbol, "board": record.get("board"),
            "reason": record.get("reason"), "rows": len(frame),
            **outcome.summary(),
        })
        if frame.empty:
            continue
        written = store.append(
            frame, provider="tencent", family=mc.FAMILY_TRADE,
            data_class=source.data_class,
        )
        receipts.extend(r.to_dict() for r in written)
        acquired.append(frame)

    if not acquired:
        print(json.dumps({"acquired_symbols": 0, "fetch_log": fetch_log}, indent=2))
        return 1

    events = pd.concat(acquired, ignore_index=True)
    report = integrity.run_integrity_checks(
        events, family=mc.FAMILY_TRADE, data_class=source.data_class
    )
    reconciliation = reconcile.reconcile_days(events, panel_day)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trade_date": args.trade_date,
        "provider": "tencent",
        "data_class": source.data_class,
        "aggregation_seconds": source.AGGREGATION_SECONDS,
        "cohort": cohort.to_dict("records"),
        "fetch_log": fetch_log,
        "journal": {
            "root": args.output,
            "partitions_written": len(receipts),
            "rows_written": int(sum(r["rows"] for r in receipts)),
            "receipts": receipts,
        },
        "integrity": report.to_dict(),
        "reconciliation": reconciliation,
    }

    report_dir = Path(args.report)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"tick_acquisition_{args.trade_date}.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(json.dumps({
        "trade_date": args.trade_date,
        "symbols_attempted": len(cohort),
        "symbols_acquired": len(acquired),
        "events": len(events),
        "integrity_verdicts": report.verdict_counts,
        "integrity_failed": report.failed,
        "integrity_not_run": report.not_run,
        "reconciliation_status_counts": reconciliation["status_counts"],
        "match_rate_over_verifiable": reconciliation["match_rate_over_verifiable"],
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
