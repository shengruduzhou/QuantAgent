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


def _prefixed_symbol(symbol: str) -> str:
    text = str(symbol).strip().upper()
    if "." in text:
        code, exchange = text.split(".", 1)
        return f"{exchange.lower()}{code.zfill(6)}"
    code = text.zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("4", "8", "92")):
        return f"bj{code}"
    return f"sz{code}"


def _calendar_window(
    ak, lookback_sessions: int
) -> tuple[TradingCalendar, str, str, dict[str, object]]:
    raw = ak.tool_trade_date_hist_sina()
    if raw is None or raw.empty or "trade_date" not in raw.columns:
        raise RuntimeError("AKShare tool_trade_date_hist_sina returned no usable sessions")
    sessions = pd.DatetimeIndex(
        pd.to_datetime(raw["trade_date"], errors="coerce").dropna()
    ).normalize()
    sessions = pd.DatetimeIndex(sorted(set(sessions)))
    today = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None).normalize()
    completed = sessions[sessions < today]
    if len(completed) < max(lookback_sessions + 2, 5):
        raise RuntimeError("AKShare trading calendar does not cover enough completed sessions")
    # Exclude the newest known session so every requested bar can resolve to a
    # following session under QuantAgent's conservative T+1 availability rule.
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


def _economic_scale_evidence(raw: pd.DataFrame) -> dict[str, object]:
    """Check that amount/volume has the same price scale as the daily bar.

    String unit labels cannot detect an upstream 100x lot/share regression. For
    positive-volume rows, CNY turnover divided by shares must be an attainable
    volume-weighted price and therefore lie inside the day's traded low/high
    range, modulo a small tolerance for vendor rounding.
    """
    required = ("low", "high", "volume", "amount")
    if any(column not in raw.columns for column in required):
        return {
            "economic_scale_check_passed": False,
            "economic_scale_rows": 0,
            "economic_scale_violations": 0,
            "reason": "missing_scale_columns",
        }
    numeric = raw.loc[:, list(required)].apply(pd.to_numeric, errors="coerce")
    valid = (
        numeric["low"].gt(0)
        & numeric["high"].ge(numeric["low"])
        & numeric["volume"].gt(0)
        & numeric["amount"].ge(0)
    )
    if not bool(valid.any()):
        return {
            "economic_scale_check_passed": False,
            "economic_scale_rows": 0,
            "economic_scale_violations": 0,
            "reason": "no_positive_volume_rows",
        }
    checked = numeric.loc[valid].copy()
    implied_vwap = checked["amount"] / checked["volume"]
    # 2% accommodates harmless vendor rounding while remaining orders of
    # magnitude away from the 100x error caused by confusing lots and shares.
    lower_bound = checked["low"] * 0.98
    upper_bound = checked["high"] * 1.02
    within = implied_vwap.ge(lower_bound) & implied_vwap.le(upper_bound)
    return {
        "economic_scale_check_passed": bool(within.all()),
        "economic_scale_rows": int(len(checked)),
        "economic_scale_violations": int((~within).sum()),
        "implied_vwap_min": float(implied_vwap.min()),
        "implied_vwap_max": float(implied_vwap.max()),
        "price_low_min": float(checked["low"].min()),
        "price_high_max": float(checked["high"].max()),
        "tolerance_fraction": 0.02,
    }


def _nonnegative_column(frame: pd.DataFrame, column: str) -> bool | None:
    """True/False if the column exists and is numeric, None when it is absent.

    None, not False: "the source does not publish this" and "the source
    published a negative value" are different facts and must not collapse.
    """
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return bool(values.ge(0).all())


def _tencent_probe(
    ak, symbol: str, start_date: str, end_date: str
) -> dict[str, object]:
    """Probe current AKShare Tencent schema/unit semantics independently."""
    api = getattr(ak, "stock_zh_a_hist_tx", None)
    if api is None:
        return {"status": "api_unavailable", "required_for_primary_smoke": False}
    try:
        raw = api(
            symbol=_prefixed_symbol(symbol),
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="",
            timeout=15,
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}:{exc}",
            "required_for_primary_smoke": False,
        }
    if raw is None or raw.empty:
        return {
            "status": "empty",
            "rows": int(0 if raw is None else len(raw)),
            "required_for_primary_smoke": False,
        }
    required = {"date", "open", "high", "low", "close", "volume", "amount"}
    missing = sorted(required.difference(raw.columns))
    scale = _economic_scale_evidence(raw) if not missing else {
        "economic_scale_check_passed": False,
        "economic_scale_rows": 0,
        "economic_scale_violations": 0,
        "reason": "schema_incomplete",
    }
    observed = not missing and bool(scale["economic_scale_check_passed"])
    return {
        "status": "observed" if observed else ("schema_failed" if missing else "economic_scale_failed"),
        "required_for_primary_smoke": False,
        "rows": int(len(raw)),
        "columns": [str(column) for column in raw.columns],
        "missing_columns": missing,
        # These are accepted as observed source units only after the independent
        # economic scale invariant above passes.
        "current_source_volume_unit": "shares" if observed else "unverified",
        "current_source_amount_unit": "CNY" if observed else "unverified",
        # Must tolerate an absent column: DataFrame.get returns None, which
        # pd.to_numeric turns into a bare float nan with no .dropna(). Evaluating
        # it unconditionally raised AttributeError and killed the whole smoke run
        # *in the very branch that had correctly detected the missing column*, so
        # the schema_failed verdict this function computes could never be
        # reported. A schema check that crashes on schema failure is not a check.
        "nonnegative_volume": _nonnegative_column(raw, "volume"),
        "nonnegative_amount": _nonnegative_column(raw, "amount"),
        "first_date": str(raw["date"].iloc[0]) if "date" in raw.columns else None,
        "last_date": str(raw["date"].iloc[-1]) if "date" in raw.columns else None,
        **scale,
    }


def _baidu_valuation_probe(ak, symbol: str) -> dict[str, object]:
    """Probe genuine dated A-share valuation history before routing to it."""
    api = getattr(ak, "stock_zh_valuation_baidu", None)
    if api is None:
        return {"status": "api_unavailable", "required_for_primary_smoke": False}
    indicators = {
        "pe_ttm": "市盈率(TTM)",
        "pb": "市净率",
        "market_cap": "总市值",
    }
    per_indicator: dict[str, object] = {}
    date_sets: list[set[pd.Timestamp]] = []
    all_ok = True
    for canonical, indicator in indicators.items():
        try:
            raw = api(symbol=_plain_code(symbol), indicator=indicator, period="近一年")
        except Exception as exc:
            per_indicator[canonical] = {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}:{exc}",
            }
            all_ok = False
            continue
        if raw is None or raw.empty or not {"date", "value"}.issubset(raw.columns):
            per_indicator[canonical] = {
                "status": "empty_or_schema_failed",
                "rows": int(0 if raw is None else len(raw)),
                "columns": [] if raw is None else [str(column) for column in raw.columns],
            }
            all_ok = False
            continue
        dates = pd.to_datetime(raw["date"], errors="coerce").dropna().dt.normalize()
        values = pd.to_numeric(raw["value"], errors="coerce")
        valid = dates.notna() & values.notna()
        valid_dates = set(dates.loc[valid])
        date_sets.append(valid_dates)
        per_indicator[canonical] = {
            "status": "observed" if valid.any() else "no_valid_rows",
            "rows": int(len(raw)),
            "valid_rows": int(valid.sum()),
            "first_date": None if not valid.any() else str(dates.loc[valid].min().date()),
            "last_date": None if not valid.any() else str(dates.loc[valid].max().date()),
            "min_value": None if not valid.any() else float(values.loc[valid].min()),
            "max_value": None if not valid.any() else float(values.loc[valid].max()),
        }
        all_ok = all_ok and bool(valid.any())
    overlap = set.intersection(*date_sets) if date_sets and len(date_sets) == len(indicators) else set()
    return {
        "status": "observed" if all_ok and overlap else "failed",
        "required_for_primary_smoke": False,
        "symbol": symbol,
        "function_name": "stock_zh_valuation_baidu",
        "period": "近一年",
        "indicators": per_indicator,
        "common_date_count": int(len(overlap)),
        "common_first_date": None if not overlap else str(min(overlap).date()),
        "common_last_date": None if not overlap else str(max(overlap).date()),
        "historical_dates_are_source_supplied": True,
        "production_integrity_certified": False,
    }


def _sina_parity_probe(
    ak,
    symbol: str,
    start_date: str,
    end_date: str,
    calendar: TradingCalendar,
) -> dict[str, object]:
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
            symbol=_prefixed_symbol(symbol),
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
        / joined[["close_em", "close_sina"]]
        .abs()
        .max(axis=1)
        .replace(0, pd.NA)
    ).dropna()
    volume_ratio = (
        joined.loc[
            (joined["volume_em"] > 0) & (joined["volume_sina"] > 0),
            "volume_sina",
        ]
        / joined.loc[
            (joined["volume_em"] > 0) & (joined["volume_sina"] > 0),
            "volume_em",
        ]
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
    source_order = ("east_money", "tencent")
    result = AkShareLiveProvider(
        allow_network=True,
        adjust="",
        source_order=source_order,
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
    all_symbols_present = all(
        symbol in per_symbol_rows and per_symbol_rows[symbol] > 0 for symbol in symbols
    )
    tencent_probe = _tencent_probe(ak, symbols[0], start_date, end_date)
    tencent_used = bool(result.metadata.get("source_counts", {}).get("tencent", 0))
    tencent_scale_ok = bool(
        not tencent_used
        or (
            tencent_probe.get("status") == "observed"
            and tencent_probe.get("economic_scale_check_passed") is True
        )
    )
    primary_pass = bool(
        result.point_in_time
        and canonical_units_ok
        and all_symbols_present
        and not result.metadata.get("failed_symbols")
        and tencent_scale_ok
    )

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
            "source_order": list(source_order),
            "rows": int(len(frame)),
            "per_symbol_rows": per_symbol_rows,
            "point_in_time": result.point_in_time,
            "quality_score": float(result.quality_score),
            "canonical_units_ok": canonical_units_ok,
            "all_symbols_present": all_symbols_present,
            "tencent_economic_scale_required": tencent_used,
            "tencent_economic_scale_ok": tencent_scale_ok,
            "metadata": result.metadata,
            "warnings": list(result.warnings),
        },
        "tencent_independent_probe": tencent_probe,
        "historical_valuation_baidu_probe": _baidu_valuation_probe(ak, symbols[0]),
        "sina_optional_parity": _sina_parity_probe(
            ak, symbols[0], start_date, end_date, calendar
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbols",
        default="600000.SH,000001.SZ",
        help="Tiny infrastructure-smoke sample only; never a research universe",
    )
    parser.add_argument("--lookback-sessions", type=int, default=8)
    parser.add_argument(
        "--output", default="runtime/research/akshare_source_smoke/manifest.json"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("at least one smoke symbol is required")
    payload = run_smoke(symbols, max(3, int(args.lookback_sessions)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
