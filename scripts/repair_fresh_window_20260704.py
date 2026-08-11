#!/usr/bin/env python3
"""PIT-safe repair of the 2026-05-19..2026-07-02 fresh-window ingest.

The original one-off repair used a current ST snapshot across the historical
window. Re-running that behavior is now prohibited. This version requires dated
ST/risk-warning evidence with ``available_at`` semantics, uses it for every
fresh-window row, and records the evidence digest on the repaired panel rows.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantagent.data.providers.st_pit import (
    HistoricalSTCoverageError,
    attach_historical_st,
    load_historical_st_evidence,
)
from quantagent.quant_math.ashare import board_price_limit_vector


PANEL = Path("runtime/data/v7/silver/market_panel/market_panel.parquet")
WIN_START = pd.Timestamp("2026-05-19")
WIN_END = pd.Timestamp("2026-07-02")
SEED_DATE = pd.Timestamp("2026-05-18")


def _tf_client():
    try:
        from dotenv import load_dotenv

        load_dotenv(".env", override=False)
    except Exception:
        pass
    import tickflow

    return tickflow.TickFlow(
        api_key=os.environ["TICKFLOW_API_KEY"],
        base_url=os.environ.get("TICKFLOW_API_ENDPOINT") or None,
    )


def _board_cap(symbol: str, is_st: bool, trade_date: object | None = None) -> float:
    """Compatibility helper retained for callers; delegates to canonical rules."""

    date = pd.Timestamp(trade_date or WIN_END)
    return float(
        board_price_limit_vector(
            pd.Series([str(symbol)]),
            pd.Series([bool(is_st)]),
            trade_dates=pd.Series([date]),
        ).iloc[0]
    )


def fetch_with_retry(tf, sym: str, attempts: int = 3):
    for index in range(attempts):
        try:
            return tf.klines.get(
                sym,
                period="1d",
                count=40,
                adjust="none",
                as_dataframe=True,
            )
        except Exception:
            if index < attempts - 1:
                time.sleep((2, 5, 10)[index])
    return None


def _load_required_evidence(path: str | None):
    if not path:
        raise SystemExit(
            "historical ST PIT evidence is required; set --historical-st-path or "
            "QUANTAGENT_HISTORICAL_ST_PATH. The original current-snapshot repair is quarantined."
        )
    try:
        return load_historical_st_evidence(path)
    except HistoricalSTCoverageError as exc:
        raise SystemExit(f"historical ST PIT evidence invalid: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols-file",
        default=None,
        help="targeted pass: newline-separated symbols instead of all seed-date symbols",
    )
    parser.add_argument(
        "--historical-st-path",
        default=os.environ.get("QUANTAGENT_HISTORICAL_ST_PATH"),
        help="dated PIT ST table (parquet/csv/jsonl); required",
    )
    args = parser.parse_args()
    evidence = _load_required_evidence(args.historical_st_path)

    panel = pd.read_parquet(PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    for column in ("is_st_provenance", "st_evidence_sha256"):
        if column not in panel.columns:
            panel[column] = pd.NA

    seed = panel[panel["trade_date"] == SEED_DATE]
    if args.symbols_file:
        symbols = sorted(set(Path(args.symbols_file).read_text().split()))
    else:
        symbols = sorted(seed["symbol"].astype(str).unique())
    have = set(
        map(
            tuple,
            panel[
                (panel["trade_date"] >= WIN_START)
                & (panel["trade_date"] <= WIN_END)
            ][["symbol", "trade_date"]]
            .astype({"symbol": str})
            .itertuples(index=False, name=None),
        )
    )
    print(
        f"symbols on seed date: {len(symbols)}; existing fresh rows: {len(have):,}",
        flush=True,
    )

    tf = _tf_client()
    rows: list[pd.DataFrame] = []
    failed: list[str] = []
    for index, symbol in enumerate(symbols):
        kline = fetch_with_retry(tf, symbol)
        if kline is None:
            failed.append(symbol)
        elif len(kline):
            kline = kline.copy()
            kline["symbol"] = symbol
            kline["trade_date"] = pd.to_datetime(kline["trade_date"])
            kline = kline[
                (kline["trade_date"] >= WIN_START)
                & (kline["trade_date"] <= WIN_END)
            ]
            kline = kline[
                [(symbol, date) not in have for date in kline["trade_date"]]
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
        if (index + 1) % 250 == 0:
            print(
                f"  {index + 1}/{len(symbols)} fetched (failed {len(failed)})",
                flush=True,
            )

    if not rows:
        print(json.dumps({"repaired_rows": 0, "failed": len(failed)}))
        return 0
    new = pd.concat(rows, ignore_index=True)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        new[column] = pd.to_numeric(new[column], errors="coerce")
    new = new.dropna(subset=["trade_date", "symbol", "close"])
    try:
        new = attach_historical_st(new, evidence)
    except HistoricalSTCoverageError as exc:
        raise SystemExit(
            "refusing fresh-window repair because historical ST coverage is incomplete: "
            f"{exc}"
        ) from exc
    new["available_at"] = new["trade_date"] + pd.Timedelta(days=1)
    new["source"] = "tickflow_daily_append_repair_20260704_pit"
    new["source_type"] = "vendor_api"
    new["source_reliability"] = 0.9
    new["point_in_time_valid"] = True
    for column in panel.columns:
        if column not in new.columns:
            new[column] = np.nan
    new = new[list(panel.columns)]

    backup = PANEL.with_name("market_panel.pre_repair_20260704_pit.tail.parquet")
    if not backup.exists():
        panel[panel["trade_date"] >= SEED_DATE - pd.Timedelta(days=5)].to_parquet(
            backup, index=False
        )

    merged = pd.concat([panel, new], ignore_index=True)
    merged = merged.drop_duplicates(["symbol", "trade_date"], keep="first")

    window_mask = (merged["trade_date"] >= WIN_START) & (
        merged["trade_date"] <= WIN_END
    )
    chain = merged[
        (merged["trade_date"] >= SEED_DATE) & (merged["trade_date"] <= WIN_END)
    ][["symbol", "trade_date", "close", "volume"]].sort_values(
        ["symbol", "trade_date"]
    ).copy()
    chain["prev_close"] = chain.groupby("symbol")["close"].shift(1)
    target = chain[chain["trade_date"] >= WIN_START].copy()
    try:
        target = attach_historical_st(target, evidence)
    except HistoricalSTCoverageError as exc:
        raise SystemExit(
            "refusing fresh-window flag rebuild because historical ST coverage is incomplete: "
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
    missing = ~index.isin(flags.index)
    if bool(missing.any()):
        raise SystemExit(
            f"historical ST/flag rebuild misses {int(missing.sum())} fresh-window rows; refusing write"
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

    window = merged[window_mask]
    report = {
        "repaired_rows_inserted": int(len(new)),
        "failed_symbols": failed[:50],
        "n_failed": len(failed),
        "fresh_days": int(window["trade_date"].nunique()),
        "coverage_min": int(window.groupby("trade_date")["symbol"].nunique().min()),
        "coverage_max": int(window.groupby("trade_date")["symbol"].nunique().max()),
        "has_20260519": bool((window["trade_date"] == WIN_START).any()),
        "limit_up_rate_mean": round(
            float(window.groupby("trade_date")["is_limit_up"].mean().mean()), 4
        ),
        "historical_st_evidence_sha256": evidence.source_sha256,
        "historical_st_evidence_mode": evidence.mode,
    }
    print(json.dumps(report, ensure_ascii=False))
    Path("runtime/logs/repair_fresh_window_20260704.report.json").write_text(
        json.dumps(report, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
