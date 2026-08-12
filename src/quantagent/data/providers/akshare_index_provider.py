"""AKShare equity-index / commodity / treasury-future provider.

These daily series are features consumed by an A-share strategy. Their
``available_at`` is therefore defined as the next **A-share decision session**
after the source observation date. This is a conservative consumption-time
contract, not a claim that the A-share calendar is the source futures-exchange
calendar. Missing decision-calendar evidence prevents PIT caching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.data.providers.akshare_calendar import (
    AkShareCalendarEvidence,
    load_akshare_research_calendar,
    next_session_available_at,
)
from quantagent.data.providers.base import ProviderResult, ProviderUnavailable
from quantagent.data.providers.pit_cache import (
    PITCacheConfig,
    PITTableSpec,
    PITTimeSeriesCache,
)
from quantagent.data.trading_calendar import TradingCalendar


INDEX_AVAILABLE_AT_LAG_DAYS = 1

EQUITY_INDICES: tuple[tuple[str, str, str], ...] = (
    ("sh000300", "csi300", "index"),
    ("sh000905", "csi500", "index"),
    ("sh000688", "csi_star50", "index"),
    ("sz399006", "chinext", "index"),
    ("sh000016", "sse50", "index"),
    ("sh000852", "csi1000", "index"),
)
COMMODITY_MAIN: tuple[tuple[str, str, str], ...] = (
    ("CU0", "shfe_copper", "commodity"),
    ("RB0", "shfe_rebar", "commodity"),
    ("AU0", "shfe_gold", "commodity"),
    ("SC0", "ine_crude", "commodity"),
    ("FG0", "czce_glass", "commodity"),
)
TREASURY_FUTURES: tuple[tuple[str, str, str], ...] = (
    ("T0", "ten_year_treasury", "treasury_future"),
    ("TF0", "five_year_treasury", "treasury_future"),
)

_TABLES: tuple[PITTableSpec, ...] = (
    PITTableSpec(
        name="equity_index",
        filename="equity_index.parquet",
        dedup_keys=("observation_date", "symbol"),
    ),
    PITTableSpec(
        name="commodity_main",
        filename="commodity_main.parquet",
        dedup_keys=("observation_date", "symbol"),
    ),
    PITTableSpec(
        name="treasury_future",
        filename="treasury_future.parquet",
        dedup_keys=("observation_date", "symbol"),
    ),
)


@dataclass
class AkShareIndexProvider:
    allow_network: bool = False
    root: str | None = None
    trading_calendar: TradingCalendar | None = None

    def __post_init__(self) -> None:
        root = self.root or str(
            quant_paths().data_root / "v7" / "raw" / "akshare" / "index"
        )
        self.cache = PITTimeSeriesCache(PITCacheConfig(root=root, tables=_TABLES))

    def fetch_all(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, ProviderResult]:
        ak = self._akshare()
        calendar_evidence = self._calendar_evidence(ak)
        results: dict[str, ProviderResult] = {}
        results["equity_index"] = self._fetch_index_group(
            ak,
            EQUITY_INDICES,
            "equity_index",
            endpoint="stock_zh_index_daily",
            start_date=start_date,
            end_date=end_date,
            calendar_evidence=calendar_evidence,
        )
        results["commodity_main"] = self._fetch_index_group(
            ak,
            COMMODITY_MAIN,
            "commodity_main",
            endpoint="futures_main_sina",
            start_date=start_date,
            end_date=end_date,
            calendar_evidence=calendar_evidence,
        )
        results["treasury_future"] = self._fetch_index_group(
            ak,
            TREASURY_FUTURES,
            "treasury_future",
            endpoint="futures_main_sina",
            start_date=start_date,
            end_date=end_date,
            calendar_evidence=calendar_evidence,
        )
        return results

    def load_pit(self, table: str, as_of_date: str) -> ProviderResult:
        return self.cache.load_pit_frame(table, as_of_date)

    def _akshare(self):
        if not self.allow_network:
            raise ProviderUnavailable(
                "AkShareIndexProvider network disabled; set allow_network=True explicitly"
            )
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise ProviderUnavailable("akshare is not installed") from exc
        return ak

    def _calendar_evidence(self, ak: object) -> AkShareCalendarEvidence:
        if self.trading_calendar is not None:
            return AkShareCalendarEvidence(
                self.trading_calendar,
                {
                    "source": "injected_trading_calendar",
                    "production_certified": False,
                    "status": "passed" if not self.trading_calendar.empty else "empty",
                },
            )
        return load_akshare_research_calendar(
            allow_network=self.allow_network, ak_module=ak
        )

    def _fetch_index_group(
        self,
        ak_mod,
        members: tuple[tuple[str, str, str], ...],
        table: str,
        endpoint: str,
        start_date: str | None,
        end_date: str | None,
        calendar_evidence: AkShareCalendarEvidence,
    ) -> ProviderResult:
        warnings: list[str] = list(calendar_evidence.warnings)
        frames: list[pd.DataFrame] = []
        failed_symbols: list[str] = []
        for symbol, label, kind in members:
            try:
                fn = getattr(ak_mod, endpoint, None)
                if fn is None:
                    warnings.append(f"missing_endpoint:{endpoint}")
                    failed_symbols.append(symbol)
                    continue
                raw = fn(symbol=symbol)
                normalised = _normalize_ohlcv(
                    raw,
                    symbol=symbol,
                    label=label,
                    kind=kind,
                    start_date=start_date,
                    end_date=end_date,
                    trading_calendar=calendar_evidence.calendar,
                )
                if not normalised.empty:
                    frames.append(normalised)
                else:
                    failed_symbols.append(symbol)
            except Exception as exc:
                failed_symbols.append(symbol)
                warnings.append(f"fetch_failed:{symbol}:{type(exc).__name__}:{exc}")
        if not frames:
            return ProviderResult(
                pd.DataFrame(),
                source=f"akshare_index:{table}",
                quality_score=0.0,
                warnings=tuple(dict.fromkeys(warnings)) or ("empty_response",),
                metadata={
                    "endpoint": endpoint,
                    "calendar": calendar_evidence.metadata,
                    "failed_symbols": failed_symbols,
                    "production_integrity_certified": False,
                },
            )
        combined = pd.concat(frames, ignore_index=True)
        pit_valid = bool(
            not failed_symbols
            and pd.to_datetime(combined["available_at"], errors="coerce").notna().all()
        )
        if not pit_valid:
            warnings.append("akshare_index_group_not_pit_complete:not_cached")
        else:
            self.cache.upsert(table, combined)
        return ProviderResult(
            combined.reset_index(drop=True),
            source=f"akshare_index:{table}",
            point_in_time=pit_valid,
            quality_score=0.78 if pit_valid else 0.35,
            warnings=tuple(dict.fromkeys(warnings)),
            metadata={
                "row_count": int(len(combined)),
                "path": str(self.cache.path_for(table)) if pit_valid else None,
                "cached_as_pit": pit_valid,
                "endpoint": endpoint,
                "akshare_version": str(getattr(ak_mod, "__version__", "unknown")),
                "failed_symbols": failed_symbols,
                "calendar": calendar_evidence.metadata,
                "availability_calendar_scope": "A-share decision sessions",
                "source_market_calendar_certified": False,
                "production_integrity_certified": False,
            },
        )


def _normalize_ohlcv(
    raw: pd.DataFrame,
    *,
    symbol: str,
    label: str,
    kind: str,
    start_date: str | None = None,
    end_date: str | None = None,
    trading_calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Map AKShare index/futures daily data to the A-share feature PIT schema."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = _first_match(df.columns, ("日期", "date", "trade_date"))
    open_col = _first_match(df.columns, ("开盘", "开盘价", "open"))
    high_col = _first_match(df.columns, ("最高", "最高价", "high"))
    low_col = _first_match(df.columns, ("最低", "最低价", "low"))
    close_col = _first_match(df.columns, ("收盘", "收盘价", "close"))
    vol_col = _first_match(df.columns, ("成交量", "volume"))
    amt_col = _first_match(df.columns, ("成交额", "amount"))
    if date_col is None or close_col is None:
        return pd.DataFrame()
    obs = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    out = pd.DataFrame(
        {
            "observation_date": obs,
            "symbol": symbol,
            "label": label,
            "kind": kind,
            "open": pd.to_numeric(df[open_col], errors="coerce") if open_col else pd.NA,
            "high": pd.to_numeric(df[high_col], errors="coerce") if high_col else pd.NA,
            "low": pd.to_numeric(df[low_col], errors="coerce") if low_col else pd.NA,
            "close": pd.to_numeric(df[close_col], errors="coerce"),
            "volume": pd.to_numeric(df[vol_col], errors="coerce") if vol_col else pd.NA,
            "amount": pd.to_numeric(df[amt_col], errors="coerce") if amt_col else pd.NA,
        }
    ).dropna(subset=["observation_date", "close"])
    if start_date:
        out = out[out["observation_date"] >= pd.Timestamp(start_date)]
    if end_date:
        out = out[out["observation_date"] <= pd.Timestamp(end_date)]
    if out.empty:
        return out
    out["available_at"] = next_session_available_at(
        out["observation_date"],
        trading_calendar,
        lag_sessions=INDEX_AVAILABLE_AT_LAG_DAYS,
    ).to_numpy()
    out["source"] = "akshare:index_or_futures_daily"
    out["availability_calendar_scope"] = "A-share decision sessions"
    return out


def _first_match(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    available = {str(c).strip(): str(c) for c in columns}
    for candidate in candidates:
        if candidate in available:
            return available[candidate]
    return None


__all__ = [
    "AkShareIndexProvider",
    "EQUITY_INDICES",
    "COMMODITY_MAIN",
    "TREASURY_FUTURES",
    "INDEX_AVAILABLE_AT_LAG_DAYS",
    "_normalize_ohlcv",
]
