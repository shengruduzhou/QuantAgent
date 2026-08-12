#!/usr/bin/env python3
"""Low-rate live AKShare source/schema/unit smoke.

This script is connectivity and data-contract evidence only. It deliberately
uses a tiny fixed infrastructure sample and can never become factor/performance
or live-trading evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import pandas as pd

from quantagent.data.providers.akshare_live_provider import (
    AkShareLiveProvider,
    _normalize_akshare_daily,
)
from quantagent.data.providers.base import ProviderRequest
from quantagent.data.trading_calendar import TradingCalendar


def _parse_symbols(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _plain_code(symbol: str) -> str:
    return str(symbol).split(".", 1)[0].zfill(6)


def _sina_symbol(symbol: str) -> str:
    text = str(symbol).strip().upper()
    if "." in text:
        code, exchange = text.split(".", 1)
        return f"{exchange.lower()}{code.zfill(6)}"
    code = text.zfill(6)
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def _calendar_window(ak, lookback_sessions: int) -> tuple[TradingCalendar, str, str, dict[str, object]]:
    raw = ak.tool_trade_date_hist_sina()
    if raw is None or raw.empty or "trade_date" not in raw.columns:
        raise RuntimeError("AKShare tool_trade_date_hist_sina returned no usable sessions")
    sessions = pd.DatetimeIndex(pd.to_datetime(raw["trade_date"], errors="coerce").dropna()).normalize()
    sessions = pd.DatetimeIndex(sorted(set(sessions)))
    today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    completed = sessions[sessions < today]
    # Exclude the newest known session so every requested bar can resolve to a
    # following session under QuantAgent's conservative T+1 availability rule.
    if len(completed) < max(lookback_sessions + 2, 5):
        raise RuntimeError("AKShare trading calendar does not cover enough completed sessions")
    end = completed[-2]
    eligible = completed[completed <= end]
    start = eligible[-lookback_sessions]
    calendar = TradingCalendar.from_dates(sessions)
    return (
        calendar,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        {
            "source": "akshare:tool_trade_date_hist_sina",
            "production_certified": False,
            "session_count": int(len(sessions)),
            "first_session": sessions[0].strftime("%Y-%m-%d"),
            "last_session": sessions[-1].strftime("%Y-%m-%d"),
        },
    )


def _sina_parity_probe(ak, symbol: str, start_date: str, end_date: str, calendar: TradingCalendar) -> dict[str, object]:
    compact_start = start_date.replace("-", "")
    compact_end = end_date.replace("-", "")
    try:
        em_raw = ak.stock_zh_a_hist(
            symbol=_plain_code(symbol),
            period="daily",
            start_date=compact_start,
            end_date=compact_end,
            adjust="",
        )
        sina_raw = ak.stock_zh_a_daily(
            symbol=_sina_symbol(symbol),
            start_date=compact_start,
            end_date=compact_end,
            adjust="",
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}:{exc}",
            "required_for_primary_smoke": False,
        }
    if em_raw is None or em_raw.empty or sina_raw is None or sina_raw.empty:
        return {
            "status": "empty",
            "required_for_primary_smoke": False,
            "east_money_rows": int(0 if em_raw is None else len(em_raw)),
            "sina_rows": int(0 if sina_raw is None else len(sina_raw)),
        }
    em = _normalize_akshare_daily(
        em_raw,
        symbol,
        source="east_money",
        adjust="",
        trading_calendar=calendar,
    )
    sina = _normalize_akshare_daily(
        sina_raw,
        symbol,
        source="sina",
        adjust="",
        trading_calendar=calendar,
    )
    joined = em[["trade_date", "close", "volume"]].merge(
        sina[["trade_date", "close", "volume"]],
        on="trade_date",
        suffixes=("_em", "_sina"),
    )
    joined = joined.dropna()
    if joined.empty:
        return {"status": "no_overlap", "required_for_primary_smoke": False}
    close_rel = (
        (joined["close_em"] - joined["close_sina"]).abs()
        / joined[["close_em", "close_sina"]].abs().max(axis=1).replace(0, pd.NA)
    ).dropna()
    volume_ratio = (
        joined.loc[(joined["volume_em"] > 0) & (joined["volume_sina"] > 0), "volume_sina"]
        / joined.loc[(joined["volume_em"] > 0) & (joined["volume_sina"] > 0), "volume_em"]
    ).dropna()
    return {
        "status": "observed",
        "required_for_primary_smoke": False,
        "overlap_rows": int(len(joined)),
        "max_close_relative_difference": None if close_rel.empty else float(close_rel.max()),
        "median_normalized_volume_ratio_sina_to_east_money": None
        if volume_ratio.empty
        else float(median(volume_ratio.tolist())),
        "canonical_volume_unit": "shares",
        "east_money_raw_volume_unit": "lots_100_shares",
        "sina_raw_volume_unit": "shares",
    }


def run_smoke(symbols: tuple[str, ...], lookback_sessions: int) -> dict[str, object]:
    import akshare as ak  # type: ignore

    calendar, start_date, end_date, calendar_meta = _calendar_window(ak, lookback_sessions)
    request = ProviderRequest(start_date=start_date, end_date=end_date, symbols=symbols)
    result = AkShareLiveProvider(
        allow_network=True,
        adjust="",
        # Primary smoke isolates the recommended EastMoney source. Sina is
        # exercised exactly once below as an optional fallback/parity probe.
        source_order=("east_money",),
        trading_calendar=calendar,
        calendar_source="akshare:tool_trade_date_hist_sina",
    ).daily_ohlcv(request)

    frame = result.frame
    per_symbol_rows = (
        {str(k): int(v) for k, v in frame.groupby("symbol").size().items()}
        if not frame.empty and "symbol" in frame.columns
        else {}
    )
    canonical_units_ok = bool(
        not frame.empty
        and "volume_unit" in frame.columns
        and set(frame["volume_unit"].dropna().astype(str)) == {"shares"}
        and "amount_unit" in frame.columns
        and set(frame["amount_unit"].dropna().astype(str)) == {"CNY"}
    )
    all_symbols_present = all(symbol in per_symbol_rows and per_symbol_rows[symbol] > 0 for symbol in symbols)
    primary_pass = bool(
        result.point_in_time
        and canonical_units_ok
        and all_symbols_present
        and not result.metadata.get("failed_symbols")
    )

    sina_probe = _sina_parity_probe(ak, symbols[0], start_date, end_date, calendar)
    return {
        "schema_version": "akshare_source_smoke_v1",
        "status": "passed" if primary_pass else "failed",
        "research_smoke_only": True,
        "performance_evidence": False,
        "factor_promotion": False,
        "model_promotion": False,
        "economic_live_eligible": False,
        "automatic_factor_activation": False,
        "production_integrity_certified": False,
        "akshare_version": str(getattr(ak, "__version__", "unknown")),
        "requested_symbols": list(symbols),
        "window": {"start_date": start_date, "end_date": end_date},
        "calendar": calendar_meta,
        "primary": {
            "source_order": ["east_money"],
            "rows": int(len(frame)),
            "per_symbol_rows": per_symbol_rows,
            "point_in_time": result.point_in_time,
            "quality_score": float(result.quality_score),
            "canonical_units_ok": canonical_units_ok,
            "all_symbols_present": all_symbols_present,
            "metadata": result.metadata,
            "warnings": list(result.warnings),
        },
        "sina_optional_parity": sina_probe,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        default="600000.SH,000001.SZ",
        help="Tiny infrastructure-smoke sample only; never a research universe",
    )
    parser.add_argument("--lookback-sessions", type=int, default=8)
    parser.add_argument("--output", default="runtime/research/akshare_source_smoke/manifest.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("at least one smoke symbol is required")
    payload = run_smoke(symbols, max(3, int(args.lookback_sessions)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
