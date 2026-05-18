"""AkShare equity-index / commodity / treasury-future provider.

Pulls daily OHLCV for the major Chinese equity indices, commodity main-
continuous futures, and treasury futures. All series share the same
``observation_date + close / open / high / low / volume / amount``
schema; ``available_at = observation_date + 1`` business day.

Treasury futures are particularly useful for the v7 "national-team flow"
thesis since they jointly price duration and policy expectations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.data.providers.base import ProviderResult, ProviderUnavailable
from quantagent.data.providers.pit_cache import PITCacheConfig, PITTableSpec, PITTimeSeriesCache


INDEX_AVAILABLE_AT_LAG_DAYS = 1

# (symbol, label, kind) — kind ∈ {"index", "commodity", "treasury_future"}.
# Equity indices use the Sina-style "sh"/"sz" prefix (akshare's
# stock_zh_index_daily endpoint demands it). Commodity / treasury futures
# use the main-continuous CFFEX/SHFE/CZCE codes consumed by futures_main_sina.
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
    PITTableSpec(name="equity_index", filename="equity_index.parquet",
                 dedup_keys=("observation_date", "symbol")),
    PITTableSpec(name="commodity_main", filename="commodity_main.parquet",
                 dedup_keys=("observation_date", "symbol")),
    PITTableSpec(name="treasury_future", filename="treasury_future.parquet",
                 dedup_keys=("observation_date", "symbol")),
)


@dataclass
class AkShareIndexProvider:
    allow_network: bool = False
    root: str | None = None

    def __post_init__(self) -> None:
        root = self.root or str(quant_paths().data_root / "v7" / "raw" / "akshare" / "index")
        self.cache = PITTimeSeriesCache(PITCacheConfig(root=root, tables=_TABLES))

    def fetch_all(self, start_date: str | None = None, end_date: str | None = None) -> dict[str, ProviderResult]:
        ak = self._akshare()
        results: dict[str, ProviderResult] = {}
        results["equity_index"] = self._fetch_index_group(
            ak, EQUITY_INDICES, "equity_index", endpoint="stock_zh_index_daily",
            start_date=start_date, end_date=end_date,
        )
        results["commodity_main"] = self._fetch_index_group(
            ak, COMMODITY_MAIN, "commodity_main", endpoint="futures_main_sina",
            start_date=start_date, end_date=end_date,
        )
        results["treasury_future"] = self._fetch_index_group(
            ak, TREASURY_FUTURES, "treasury_future", endpoint="futures_main_sina",
            start_date=start_date, end_date=end_date,
        )
        return results

    def load_pit(self, table: str, as_of_date: str) -> ProviderResult:
        return self.cache.load_pit_frame(table, as_of_date)

    # ------------------------------------------------------------------ #

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

    def _fetch_index_group(
        self,
        ak_mod,
        members: tuple[tuple[str, str, str], ...],
        table: str,
        endpoint: str,
        start_date: str | None,
        end_date: str | None,
    ) -> ProviderResult:
        warnings: list[str] = []
        frames: list[pd.DataFrame] = []
        for symbol, label, kind in members:
            try:
                fn = getattr(ak_mod, endpoint, None)
                if fn is None:
                    warnings.append(f"missing_endpoint:{endpoint}")
                    continue
                raw = fn(symbol=symbol)
                normalised = _normalize_ohlcv(raw, symbol=symbol, label=label, kind=kind,
                                              start_date=start_date, end_date=end_date)
                if not normalised.empty:
                    frames.append(normalised)
            except Exception as exc:
                warnings.append(f"fetch_failed:{symbol}:{type(exc).__name__}:{exc}")
        if not frames:
            return ProviderResult(
                pd.DataFrame(),
                source=f"akshare_index:{table}",
                quality_score=0.0,
                warnings=tuple(warnings) or ("empty_response",),
            )
        combined = pd.concat(frames, ignore_index=True)
        self.cache.upsert(table, combined)
        return ProviderResult(
            combined.reset_index(drop=True),
            source=f"akshare_index:{table}",
            quality_score=0.78,
            warnings=tuple(warnings),
            metadata={"row_count": int(len(combined)), "path": str(self.cache.path_for(table))},
        )


def _normalize_ohlcv(
    raw: pd.DataFrame,
    *,
    symbol: str,
    label: str,
    kind: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Map akshare's index/futures daily frame to our PIT schema."""
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
    obs = pd.to_datetime(df[date_col], errors="coerce")
    out = pd.DataFrame({
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
    }).dropna(subset=["observation_date", "close"])
    if start_date:
        out = out[out["observation_date"] >= pd.Timestamp(start_date)]
    if end_date:
        out = out[out["observation_date"] <= pd.Timestamp(end_date)]
    if out.empty:
        return out
    out["available_at"] = out["observation_date"] + pd.Timedelta(days=INDEX_AVAILABLE_AT_LAG_DAYS)
    out["source"] = "akshare:index_or_futures_daily"
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
