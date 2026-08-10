#!/usr/bin/env python3
"""Network/schema smoke for the public BaoStock market-data research path.

This command is deliberately *not* a factor backtest and its fixed symbols are
not a research universe. It answers a narrower operational question: can the
scheduled runtime log in, pull raw A-share daily/intraday bars, retrieve an
independent exchange calendar, and satisfy the repository's minimum schema and
sanity contracts?

The evidence emitted here must never be used to promote a factor, model, or
strategy. Public-provider timestamps/adjustments remain research inputs until
separately certified by the production PIT/integrity pipeline.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from quantagent.data.providers.base import ProviderRequest
from quantagent.data.providers.baostock_provider import BaoStockConfig, BaoStockProvider
from quantagent.factors.executable_labels import canonical_market_sessions


SMOKE_SCHEMA = "market_data_smoke_v1"
DEFAULT_SYMBOLS = ("600000.SH", "000001.SZ", "300750.SZ")


def _parse_symbols(value: str | None) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    return items or DEFAULT_SYMBOLS


def _fetch_calendar(
    provider: BaoStockProvider,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch a market calendar independently of the smoke symbols."""
    bs = provider._get_module()
    try:
        provider._login(bs)
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        if getattr(rs, "error_code", "0") != "0":
            raise RuntimeError(
                f"BaoStock query_trade_dates failed: {getattr(rs, 'error_msg', '')}"
            )
        rows: list[list[str]] = []
        while rs.next():
            rows.append(rs.get_row_data())
        fields = list(getattr(rs, "fields", []) or ["calendar_date", "is_trading_day"])
    finally:
        provider._logout(bs)
    if not rows:
        raise RuntimeError("BaoStock returned no market-calendar rows")
    calendar = pd.DataFrame(rows, columns=fields)
    required = {"calendar_date", "is_trading_day"}
    if not required.issubset(calendar.columns):
        raise RuntimeError(
            f"BaoStock calendar missing required columns: {sorted(required - set(calendar.columns))}"
        )
    calendar["calendar_date"] = pd.to_datetime(calendar["calendar_date"], errors="coerce")
    if calendar["calendar_date"].isna().any():
        raise RuntimeError("BaoStock calendar contains invalid calendar_date values")
    return calendar.sort_values("calendar_date").reset_index(drop=True)


def _validate_ohlcva(work: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Validate numeric/market invariants without self-referential comparisons."""
    out = work.copy()
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[numeric].isna().any().any():
        raise RuntimeError(f"{label} smoke has non-numeric OHLCVA values")
    if (out[["open", "high", "low", "close"]].min(axis=1) <= 0).any():
        raise RuntimeError(f"{label} smoke has non-positive OHLC values")
    if (out[["volume", "amount"]].min(axis=1) < 0).any():
        raise RuntimeError(f"{label} smoke has negative volume/amount")

    reference_high = out[["open", "close"]].max(axis=1)
    reference_low = out[["open", "close"]].min(axis=1)
    if (out["high"] < reference_high).any() or (out["high"] < out["low"]).any():
        raise RuntimeError(f"{label} smoke violates OHLC high invariant")
    if (out["low"] > reference_low).any():
        raise RuntimeError(f"{label} smoke violates OHLC low invariant")
    return out


def validate_daily_smoke(
    frame: pd.DataFrame,
    *,
    requested_symbols: Iterable[str],
    expected_adjust_flag: str = "3",
) -> dict[str, object]:
    required = {
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "tradestatus",
        "isST",
        "adjustflag",
        "available_at",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"daily smoke missing columns: {missing}")
    if frame.empty:
        raise RuntimeError("daily smoke returned zero rows")

    work = frame.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
    work["available_at"] = pd.to_datetime(work["available_at"], errors="coerce")
    if work[["trade_date", "available_at"]].isna().any().any():
        raise RuntimeError("daily smoke has invalid trade_date/available_at")
    if work.duplicated(["symbol", "trade_date"]).any():
        raise RuntimeError("daily smoke has duplicate symbol/trade_date rows")
    if (work["available_at"] < work["trade_date"]).any():
        raise RuntimeError("daily smoke has available_at before trade_date")
    work = _validate_ohlcva(work, label="daily")

    flags = set(work["adjustflag"].astype(str).str.strip().dropna().unique())
    if flags != {str(expected_adjust_flag)}:
        raise RuntimeError(
            f"daily smoke expected raw adjustflag={expected_adjust_flag}, observed={sorted(flags)}"
        )

    requested = set(str(symbol) for symbol in requested_symbols)
    observed = set(work["symbol"].astype(str).unique())
    missing_symbols = sorted(requested - observed)
    if missing_symbols:
        raise RuntimeError(f"daily smoke missing requested symbols: {missing_symbols}")

    return {
        "rows": int(len(work)),
        "symbols": sorted(observed),
        "trade_date_min": work["trade_date"].min().date().isoformat(),
        "trade_date_max": work["trade_date"].max().date().isoformat(),
        "adjustment": "raw",
        "adjust_flag": str(expected_adjust_flag),
        "suspended_rows": int((work["tradestatus"].astype(str) == "0").sum()),
        "st_rows": int((work["isST"].astype(str) == "1").sum()),
    }


def validate_minute_smoke(
    frame: pd.DataFrame,
    *,
    frequency: str,
    expected_adjust_flag: str = "3",
) -> dict[str, object]:
    required = {
        "symbol",
        "trade_date",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "adjustflag",
        "available_at",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{frequency}m smoke missing columns: {missing}")
    if frame.empty:
        raise RuntimeError(f"{frequency}m smoke returned zero rows")
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["available_at"] = pd.to_datetime(work["available_at"], errors="coerce")
    if work[["timestamp", "available_at"]].isna().any().any():
        raise RuntimeError(f"{frequency}m smoke has invalid timestamps")
    if work.duplicated(["symbol", "timestamp"]).any():
        raise RuntimeError(f"{frequency}m smoke has duplicate symbol/timestamp rows")
    if (work["available_at"] < work["timestamp"]).any():
        raise RuntimeError(f"{frequency}m smoke has available_at before bar timestamp")
    work = _validate_ohlcva(work, label=f"{frequency}m")
    flags = set(work["adjustflag"].astype(str).str.strip().dropna().unique())
    if flags != {str(expected_adjust_flag)}:
        raise RuntimeError(
            f"{frequency}m smoke expected raw adjustflag={expected_adjust_flag}, observed={sorted(flags)}"
        )
    return {
        "frequency_minutes": int(frequency),
        "rows": int(len(work)),
        "symbols": sorted(work["symbol"].astype(str).unique()),
        "timestamp_min": work["timestamp"].min().isoformat(),
        "timestamp_max": work["timestamp"].max().isoformat(),
        "adjustment": "raw",
    }


def validate_calendar_smoke(calendar: pd.DataFrame) -> dict[str, object]:
    flag = calendar["is_trading_day"].astype(str).str.strip().str.lower()
    sessions = canonical_market_sessions(
        calendar.loc[flag.isin({"1", "true", "yes"}), "calendar_date"].tolist()
    )
    if len(sessions) == 0:
        raise RuntimeError("market-data smoke found no trading sessions")
    return {
        "calendar_rows": int(len(calendar)),
        "trading_sessions": int(len(sessions)),
        "session_min": sessions.min().date().isoformat(),
        "session_max": sessions.max().date().isoformat(),
        "independent_of_symbol_bars": True,
    }


def run_smoke(
    *,
    symbols: tuple[str, ...],
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> dict[str, object]:
    provider = BaoStockProvider(config=BaoStockConfig(adjust_flag="3"))
    health = provider.health_check()
    if health.get("status") != "ok":
        raise RuntimeError(f"BaoStock health check failed: {health}")

    request = ProviderRequest(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        use_cache=False,
    )
    daily = provider.daily_ohlcv(request)
    if daily.frame is None or daily.frame.empty:
        raise RuntimeError(f"BaoStock daily smoke empty: warnings={list(daily.warnings)}")
    daily_evidence = validate_daily_smoke(
        daily.frame,
        requested_symbols=symbols,
        expected_adjust_flag="3",
    )

    # Native 5m/60m validates the provider/network shape only. The 10m and
    # session-aligned aggregation contract is a separate issue #100 change.
    minute_request = ProviderRequest(
        start_date=start_date,
        end_date=end_date,
        symbols=(symbols[0],),
        use_cache=False,
    )
    minute_evidence: dict[str, object] = {}
    minute_frames: dict[str, pd.DataFrame] = {}
    for frequency in ("5", "60"):
        result = provider.minute_ohlcv(minute_request, frequency=frequency)
        if result.frame is None or result.frame.empty:
            raise RuntimeError(
                f"BaoStock {frequency}m smoke empty: warnings={list(result.warnings)}"
            )
        minute_frames[frequency] = result.frame.copy()
        minute_evidence[frequency] = validate_minute_smoke(
            result.frame,
            frequency=frequency,
            expected_adjust_flag="3",
        )

    calendar = _fetch_calendar(provider, start_date=start_date, end_date=end_date)
    calendar_evidence = validate_calendar_smoke(calendar)

    output_dir.mkdir(parents=True, exist_ok=True)
    daily.frame.to_csv(output_dir / "daily_raw_sample.csv", index=False)
    for frequency, frame in minute_frames.items():
        frame.to_csv(output_dir / f"minute_{frequency}m_raw_sample.csv", index=False)
    calendar.to_csv(output_dir / "market_calendar.csv", index=False)

    manifest: dict[str, object] = {
        "schema": SMOKE_SCHEMA,
        "research_smoke_only": True,
        "performance_evidence": False,
        "factor_promotion": False,
        "model_promotion": False,
        "economic_live_eligible": False,
        "automatic_factor_activation": False,
        "fixed_symbols_are_research_universe": False,
        "purpose": "provider/network/schema/calendar infrastructure smoke only",
        "source": {
            "provider": daily.source,
            "public_network_data": True,
            "adjustment": "raw",
            "provider_point_in_time_claim": bool(daily.point_in_time),
            "production_integrity_certified": False,
            "note": (
                "Public-provider smoke proves connectivity/schema only. Provider availability "
                "timestamps, adjustment vintages, survivorship and corporate-action provenance "
                "remain subject to the production PIT/integrity gates."
            ),
        },
        "request": {
            "symbols": list(symbols),
            "start_date": start_date,
            "end_date": end_date,
        },
        "health": health,
        "daily": daily_evidence,
        "minute": minute_evidence,
        "calendar": calendar_evidence,
        "provider_warnings": list(daily.warnings),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Fixed infrastructure-smoke symbols; never treated as a research universe.",
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--lookback-days", type=int, default=21)
    parser.add_argument("--output-dir", default="runtime/research/market_data_smoke")
    args = parser.parse_args()

    end = date.fromisoformat(args.end_date) if args.end_date else date.today()
    start = (
        date.fromisoformat(args.start_date)
        if args.start_date
        else end - timedelta(days=max(int(args.lookback_days), 7))
    )
    if start > end:
        raise SystemExit("--start-date must be <= --end-date")

    manifest = run_smoke(
        symbols=_parse_symbols(args.symbols),
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
