"""Helpers for V7 real-data symbol and date-range resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class V7ResolvedDateRange:
    start_date: str
    end_date: str
    source: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def standard_a_symbol(symbol: str) -> str:
    """Return QuantAgent's canonical A-share symbol, e.g. ``600519.SH``."""
    text = str(symbol).strip()
    upper = text.upper()
    if "." in upper:
        code, exchange = upper.split(".", 1)
        return f"{code.zfill(6)}.{exchange}"
    lower = text.lower()
    if lower.startswith(("sh", "sz", "bj")) and len(text) >= 8:
        exchange = lower[:2].upper()
        return f"{text[2:].zfill(6)}.{exchange}"
    code = upper.zfill(6)
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def to_qlib_instrument(symbol: str) -> str:
    canonical = standard_a_symbol(symbol)
    code, exchange = canonical.split(".", 1)
    return f"{exchange}{code}"


def from_qlib_instrument(instrument: str) -> str:
    text = str(instrument).strip()
    lower = text.lower()
    if lower.startswith(("sh", "sz", "bj")) and len(text) >= 8:
        return standard_a_symbol(text)
    return standard_a_symbol(text)


def last_business_day(as_of_date: str | None = None) -> str:
    ts = pd.Timestamp(as_of_date).normalize() if as_of_date else pd.Timestamp.today().normalize()
    while ts.weekday() >= 5:
        ts -= pd.Timedelta(days=1)
    return ts.strftime("%Y-%m-%d")


def read_qlib_calendar_range(provider_uri: str | Path | None) -> V7ResolvedDateRange | None:
    if provider_uri is None:
        return None
    calendar = Path(provider_uri).expanduser() / "calendars" / "day.txt"
    if not calendar.exists():
        return None
    dates = [line.strip() for line in calendar.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not dates:
        return None
    return V7ResolvedDateRange(start_date=dates[0], end_date=dates[-1], source="qlib_calendar")


def next_business_day(date_text: str) -> str:
    ts = pd.Timestamp(date_text).normalize() + pd.offsets.BDay(1)
    return ts.strftime("%Y-%m-%d")


def resolve_akshare_market_fetch_range(
    *,
    start_date: str | None,
    end_date: str | None,
    provider_uri: str | Path | None = None,
    lake_root: str | Path | None = None,
    as_of_date: str | None = None,
) -> V7ResolvedDateRange:
    """Resolve an AkShare fetch window that continues existing local data.

    If ``start_date`` is omitted we first continue after the local Qlib
    calendar, because AkShare is the recent-data bridge beyond the official
    Qlib CN dump. If Qlib is absent we fall back to a valid existing market
    manifest, then to a conservative five-year lookback.
    """
    resolved_end = end_date or last_business_day(as_of_date)
    notes: list[str] = []
    if start_date:
        return V7ResolvedDateRange(start_date=start_date, end_date=resolved_end, source="explicit", notes=tuple(notes))

    qlib_range = read_qlib_calendar_range(provider_uri)
    if qlib_range is not None:
        resolved_start = next_business_day(qlib_range.end_date)
        return V7ResolvedDateRange(
            start_date=resolved_start,
            end_date=resolved_end,
            source="after_qlib_calendar",
            notes=(f"qlib_end_date={qlib_range.end_date}",),
        )

    manifest_range = _read_market_manifest_range(lake_root)
    if manifest_range is not None:
        return V7ResolvedDateRange(
            start_date=next_business_day(manifest_range.end_date),
            end_date=resolved_end,
            source="after_existing_market_manifest",
            notes=(f"market_manifest_end_date={manifest_range.end_date}",),
        )

    fallback_start = (pd.Timestamp(resolved_end) - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    notes.append("no_qlib_calendar_or_market_manifest_found")
    return V7ResolvedDateRange(start_date=fallback_start, end_date=resolved_end, source="five_year_fallback", notes=tuple(notes))


def list_qlib_feature_symbols(
    provider_uri: str | Path,
    *,
    include_indices: bool = False,
    max_symbols: int = 0,
) -> tuple[str, ...]:
    features_root = Path(provider_uri).expanduser() / "features"
    if not features_root.exists():
        return ()
    symbols: list[str] = []
    for path in sorted(features_root.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_dir():
            continue
        name = path.name.lower()
        if not include_indices and (name.startswith("sh000") or name.startswith("sz399")):
            continue
        if not name.startswith(("sh", "sz", "bj")):
            continue
        symbols.append(from_qlib_instrument(name))
        if max_symbols and len(symbols) >= max_symbols:
            break
    return tuple(symbols)


def _read_market_manifest_range(lake_root: str | Path | None) -> V7ResolvedDateRange | None:
    if lake_root is None:
        return None
    manifest_path = Path(lake_root) / "manifests" / "market_panel.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("quality_status") not in {"passed", "warning"}:
        return None
    if int(payload.get("row_count") or 0) <= 0:
        return None
    start = payload.get("start_date")
    end = payload.get("end_date")
    if not start or not end:
        return None
    return V7ResolvedDateRange(str(start), str(end), source="market_manifest")
