#!/usr/bin/env python3
"""Chunked, resumable PIT-safe market-panel catch-up.

TickFlow supplies historical OHLCV. Historical ST/risk-warning state must come
from a separately versioned dated PIT table; current instrument-name snapshots
are rejected. Staging is window-scoped and the final merge recomputes all
tradability flags for the catch-up window using the dated ST state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import repair_fresh_window_20260704 as rep  # noqa: E402

from quantagent.data.providers.st_pit import (  # noqa: E402
    HistoricalSTCoverageError,
    attach_historical_st,
    load_historical_st_evidence,
)
from quantagent.quant_math.ashare import board_price_limit_vector  # noqa: E402


PANEL = REPO / "runtime/data/v7/silver/market_panel/market_panel.parquet"
STAGING = PANEL.parent / "_staging_catchup"
CHUNK = 250


def _load_required_st_evidence(path: str | None):
    if not path:
        raise SystemExit(
            "historical ST PIT evidence is required; set --historical-st-path or "
            "QUANTAGENT_HISTORICAL_ST_PATH. Current snapshot broadcast is forbidden."
        )
    try:
        return load_historical_st_evidence(path)
    except HistoricalSTCoverageError as exc:
        raise SystemExit(f"historical ST PIT evidence invalid: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default=None, help="last date to fetch (default: today)")
    parser.add_argument(
        "--historical-st-path",
        default=os.environ.get("QUANTAGENT_HISTORICAL_ST_PATH"),
        help="dated PIT ST table (parquet/csv/jsonl); required",
    )
    args = parser.parse_args()
    evidence = _load_required_st_evidence(args.historical_st_path)
    started = time.time()
    STAGING.mkdir(exist_ok=True)

    panel = pd.read_parquet(PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    for column in ("is_st_provenance", "st_evidence_sha256"):
        if column not in panel.columns:
            panel[column] = pd.NA

    panel_max = panel["trade_date"].max()
    win_start = panel_max + pd.Timedelta(days=1)
    win_end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()

    cst_now = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    last_available = (
        cst_now.normalize()
        if cst_now.hour * 60 + cst_now.minute >= 16 * 60
        else cst_now.normalize() - pd.Timedelta(days=1)
    )
    if win_end > last_available:
        print(
            f"clamping window end {win_end.date()} -> {last_available.date()} "
            f"(close not published yet at {cst_now:%Y-%m-%d %H:%M} CST)",
            flush=True,
        )
        win_end = last_available
    if win_start > win_end:
        print(json.dumps({"appended": 0, "note": "panel already current"}))
        return 0
    if len(pd.bdate_range(win_start, win_end)) == 0:
        print(json.dumps({"appended": 0, "note": "no business day in window"}))
        return 0

    seed_symbols = sorted(
        panel.loc[panel["trade_date"] == panel_max, "symbol"].astype(str).unique()
    )

    manifest = STAGING / "window.json"
    window_tag = f"{win_start.date()}_{win_end.date()}"
    expected_manifest = {
        "window": window_tag,
        "historical_st_evidence_sha256": evidence.source_sha256,
    }
    existing_manifest = {}
    if manifest.exists():
        try:
            existing_manifest = json.loads(manifest.read_text())
        except Exception:
            existing_manifest = {}
    if existing_manifest != expected_manifest:
        for path in list(STAGING.glob("chunk_*.parquet")) + list(
            STAGING.glob("done_*.json")
        ):
            path.unlink()
        manifest.write_text(json.dumps(expected_manifest, sort_keys=True))

    done_symbols: set[str] = set()
    for path in sorted(STAGING.glob("done_*.json")):
        try:
            done_symbols |= set(json.loads(path.read_text()))
        except Exception:
            path.unlink()
    for path in sorted(STAGING.glob("chunk_*.parquet")):
        try:
            done_symbols |= set(
                pd.read_parquet(path, columns=["symbol"])["symbol"].astype(str)
            )
        except Exception:
            path.unlink()

    todo = [symbol for symbol in seed_symbols if symbol not in done_symbols]
    print(
        f"window {win_start.date()}..{win_end.date()} | seed {len(seed_symbols)} "
        f"| staged {len(done_symbols)} | todo {len(todo)}",
        flush=True,
    )

    tf = rep._tf_client()
    failed: list[str] = []
    for chunk_start in range(0, len(todo), CHUNK):
        chunk_symbols = todo[chunk_start : chunk_start + CHUNK]
        rows: list[pd.DataFrame] = []
        for symbol in chunk_symbols:
            kline = rep.fetch_with_retry(tf, symbol)
            if kline is None:
                failed.append(symbol)
                continue
            if not len(kline):
                continue
            kline = kline.copy()
            kline["symbol"] = symbol
            kline["trade_date"] = pd.to_datetime(kline["trade_date"])
            kline = kline[
                (kline["trade_date"] >= win_start)
                & (kline["trade_date"] <= win_end)
            ]
            if len(kline):
                kline["volume"] = pd.to_numeric(
                    kline["volume"], errors="coerce"
                ) * 100.0
                rows.append(
                    kline[
                        [
                            "symbol",
                            "trade_date",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "amount",
                        ]
                    ]
                )
        staged = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(
                columns=[
                    "symbol",
                    "trade_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                ]
            )
        )
        chunk_failed = set(failed) & set(chunk_symbols)
        (STAGING / f"done_{chunk_start // CHUNK:04d}.json").write_text(
            json.dumps(sorted(set(chunk_symbols) - chunk_failed))
        )
        staged.to_parquet(
            STAGING / f"chunk_{chunk_start // CHUNK:04d}_{int(time.time())}.parquet",
            index=False,
        )
        print(
            f"  staged {chunk_start + len(chunk_symbols)}/{len(todo)} "
            f"(failed {len(failed)}) {time.time() - started:.0f}s",
            flush=True,
        )

    chunks = [pd.read_parquet(path) for path in sorted(STAGING.glob("chunk_*.parquet"))]
    new = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if len(new):
        new = new.drop_duplicates(["symbol", "trade_date"], keep="first")
    if not len(new):
        for path in list(STAGING.glob("chunk_*.parquet")) + list(
            STAGING.glob("done_*.json")
        ):
            path.unlink()
        manifest.unlink(missing_ok=True)
        print(
            json.dumps(
                {"appended": 0, "n_failed": len(failed), "note": "no new rows fetched"}
            )
        )
        return 0

    for column in ("open", "high", "low", "close", "volume", "amount"):
        new[column] = pd.to_numeric(new[column], errors="coerce")
    new = new.dropna(subset=["trade_date", "symbol", "close"])
    try:
        new = attach_historical_st(new, evidence)
    except HistoricalSTCoverageError as exc:
        raise SystemExit(
            "refusing catch-up merge because historical ST coverage is incomplete: "
            f"{exc}"
        ) from exc

    new["available_at"] = new["trade_date"] + pd.Timedelta(days=1)
    new["source"] = "tickflow_catchup_chunked"
    new["source_type"] = "vendor_api"
    new["source_reliability"] = 0.9
    new["point_in_time_valid"] = True
    for column in panel.columns:
        if column not in new.columns:
            new[column] = np.nan
    new = new[list(panel.columns)]

    backup = PANEL.with_name(f"market_panel.pre_catchup_{win_start.date()}.tail.parquet")
    if not backup.exists():
        panel[panel["trade_date"] >= panel_max - pd.Timedelta(days=5)].to_parquet(
            backup, index=False
        )
    merged = pd.concat([panel, new], ignore_index=True)
    merged = merged.drop_duplicates(["symbol", "trade_date"], keep="first")

    window_mask = (merged["trade_date"] >= win_start) & (
        merged["trade_date"] <= win_end
    )
    chain = merged[
        (merged["trade_date"] >= panel_max - pd.Timedelta(days=5))
        & (merged["trade_date"] <= win_end)
    ][["symbol", "trade_date", "close", "volume"]].sort_values(
        ["symbol", "trade_date"]
    ).copy()
    chain["prev_close"] = chain.groupby("symbol")["close"].shift(1)
    target = chain[chain["trade_date"] >= win_start].copy()
    try:
        target = attach_historical_st(target, evidence)
    except HistoricalSTCoverageError as exc:
        raise SystemExit(
            "refusing flag rebuild because historical ST coverage is incomplete: "
            f"{exc}"
        ) from exc
    ratios = board_price_limit_vector(
        target["symbol"].astype(str),
        target["is_st"].astype(bool),
        trade_dates=target["trade_date"],
    )
    up_price = (target["prev_close"] * (1.0 + ratios)).round(2)
    down_price = (target["prev_close"] * (1.0 - ratios)).round(2)
    target["is_limit_up"] = (
        (target["close"].round(2) >= up_price - 0.005)
        & target["prev_close"].notna()
    )
    target["is_limit_down"] = (
        (target["close"].round(2) <= down_price + 0.005)
        & target["prev_close"].notna()
    )
    target["is_suspended"] = target["volume"].fillna(0) <= 0
    flags = target.set_index(["symbol", "trade_date"])[
        [
            "is_limit_up",
            "is_limit_down",
            "is_suspended",
            "is_st",
            "is_st_provenance",
            "st_evidence_sha256",
        ]
    ]
    index = merged.loc[window_mask].set_index(["symbol", "trade_date"]).index
    missing_flags = ~index.isin(flags.index)
    if bool(missing_flags.any()):
        raise SystemExit(
            f"flag rebuild missing {int(missing_flags.sum())} catch-up rows; refusing panel write"
        )
    for column in (
        "is_limit_up",
        "is_limit_down",
        "is_suspended",
        "is_st",
        "is_st_provenance",
        "st_evidence_sha256",
    ):
        merged.loc[window_mask, column] = flags[column].reindex(index).to_numpy()

    merged = merged.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    merged.to_parquet(PANEL, index=False)
    for path in list(STAGING.glob("chunk_*.parquet")) + list(
        STAGING.glob("done_*.json")
    ):
        path.unlink()
    manifest.unlink(missing_ok=True)

    window = merged[window_mask]
    report = {
        "appended": int(len(new)),
        "n_failed": len(failed),
        "failed_sample": failed[:20],
        "new_max_date": str(merged["trade_date"].max().date()),
        "dates_added": [
            str(date.date()) for date in sorted(window["trade_date"].unique())
        ],
        "coverage_per_date": {
            str(date.date()): int(value)
            for date, value in window.groupby("trade_date")["symbol"].nunique().items()
        },
        "historical_st_evidence_sha256": evidence.source_sha256,
        "historical_st_evidence_mode": evidence.mode,
        "runtime_s": round(time.time() - started, 1),
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
