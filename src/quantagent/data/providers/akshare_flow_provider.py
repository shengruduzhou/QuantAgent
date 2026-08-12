"""AKShare capital-flow provider with explicit session availability.

Tracks Stock Connect net inflow and margin-financing balances. Daily observations
are eligible for the PIT cache only after their next A-share trading session is
resolved from the shared AKShare research calendar. No calendar-day or weekday
arithmetic is allowed to invent ``available_at``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
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


FLOW_AVAILABLE_AT_LAG_DAYS = 1
_NORTHBOUND_SUMMARY_CNY_PER_SOURCE_UNIT = 1e8

_TABLES: tuple[PITTableSpec, ...] = (
    PITTableSpec(
        name="northbound_flow",
        filename="northbound_flow.parquet",
        dedup_keys=("observation_date", "channel"),
    ),
    PITTableSpec(
        name="margin_balance",
        filename="margin_balance.parquet",
        dedup_keys=("observation_date", "market"),
    ),
)


@dataclass
class AkShareFlowProvider:
    allow_network: bool = False
    root: str | None = None
    trading_calendar: TradingCalendar | None = None

    def __post_init__(self) -> None:
        root = self.root or str(
            quant_paths().data_root / "v7" / "raw" / "akshare" / "flow"
        )
        self.cache = PITTimeSeriesCache(PITCacheConfig(root=root, tables=_TABLES))

    def fetch_all(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, ProviderResult]:
        ak = self._akshare()
        calendar_evidence = self._calendar_evidence(ak)
        calendar = calendar_evidence.calendar
        results: dict[str, ProviderResult] = {}
        results["northbound_flow"] = self._fetch_and_upsert(
            "northbound_flow",
            lambda: _normalize_northbound_history(
                _collect_northbound_history(ak), trading_calendar=calendar
            ),
            calendar_evidence=calendar_evidence,
            akshare_version=str(getattr(ak, "__version__", "unknown")),
        )
        results["margin_balance"] = self._fetch_and_upsert(
            "margin_balance",
            lambda: _normalize_margin_balance(
                _collect_margin(ak, start_date=start_date, end_date=end_date),
                trading_calendar=calendar,
            ),
            calendar_evidence=calendar_evidence,
            akshare_version=str(getattr(ak, "__version__", "unknown")),
        )
        return results

    def load_pit(self, table: str, as_of_date: str) -> ProviderResult:
        return self.cache.load_pit_frame(table, as_of_date)

    def _akshare(self):
        if not self.allow_network:
            raise ProviderUnavailable(
                "AkShareFlowProvider network disabled; set allow_network=True explicitly"
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

    def _fetch_and_upsert(
        self,
        table: str,
        fetcher,
        *,
        calendar_evidence: AkShareCalendarEvidence,
        akshare_version: str,
    ) -> ProviderResult:
        try:
            frame = fetcher()
        except Exception as exc:
            return ProviderResult(
                pd.DataFrame(),
                source=f"akshare_flow:{table}",
                quality_score=0.0,
                warnings=(f"fetch_failed:{type(exc).__name__}:{exc}",),
                metadata={
                    "calendar": calendar_evidence.metadata,
                    "akshare_version": akshare_version,
                    "production_integrity_certified": False,
                },
            )
        if frame is None or frame.empty:
            return ProviderResult(
                pd.DataFrame(),
                source=f"akshare_flow:{table}",
                quality_score=0.0,
                warnings=("empty_response",),
                metadata={
                    "calendar": calendar_evidence.metadata,
                    "akshare_version": akshare_version,
                    "production_integrity_certified": False,
                },
            )
        pit_valid = bool(
            "available_at" in frame.columns
            and pd.to_datetime(frame["available_at"], errors="coerce").notna().all()
        )
        warnings = list(calendar_evidence.warnings)
        if not pit_valid:
            warnings.append("akshare_flow_available_at_unresolved:not_cached_as_pit")
        else:
            self.cache.upsert(table, frame)
        return ProviderResult(
            frame.reset_index(drop=True),
            source=f"akshare_flow:{table}",
            point_in_time=pit_valid,
            quality_score=0.78 if pit_valid else 0.35,
            warnings=tuple(dict.fromkeys(warnings)),
            metadata={
                "row_count": int(len(frame)),
                "path": str(self.cache.path_for(table)) if pit_valid else None,
                "cached_as_pit": pit_valid,
                "calendar": calendar_evidence.metadata,
                "akshare_version": akshare_version,
                "production_integrity_certified": False,
            },
        )


# ---------------------------------------------------------------------- #
# Normalisers — pure functions                                            #
# ---------------------------------------------------------------------- #


def _normalize_northbound(
    raw: pd.DataFrame,
    *,
    trading_calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Normalise ``stock_hsgt_fund_flow_summary_em`` current-day rows.

    AKShare documents ``成交净买额`` / ``资金净流入`` from this endpoint in
    亿元. The canonical QuantAgent field is CNY, so conversion happens once at
    the ingestion boundary before per-channel aggregation.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = _first_match(df.columns, ("交易日", "日期", "trade_date", "date"))
    plate_col = _first_match(df.columns, ("板块",))
    direction_col = _first_match(df.columns, ("资金方向",))
    value_col = _first_match(df.columns, ("成交净买额", "资金净流入"))
    if date_col is None or value_col is None:
        return pd.DataFrame()
    obs = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    plate = (
        df[plate_col].astype(str)
        if plate_col
        else pd.Series([""] * len(df), index=df.index)
    )
    direction = (
        df[direction_col].astype(str)
        if direction_col
        else pd.Series([""] * len(df), index=df.index)
    )
    rows: list[dict[str, object]] = []
    by_date: dict[pd.Timestamp, float] = {}
    for ts, p, d, value in zip(obs, plate, direction, values):
        if pd.isna(ts) or pd.isna(value) or "北向" not in d:
            continue
        if "沪股通" in p:
            channel = "north_hgt"
        elif "深股通" in p:
            channel = "north_sgt"
        else:
            continue
        day = pd.Timestamp(ts).normalize()
        value_cny = float(value) * _NORTHBOUND_SUMMARY_CNY_PER_SOURCE_UNIT
        rows.append(
            {
                "observation_date": day,
                "channel": channel,
                "net_inflow_cny": value_cny,
            }
        )
        by_date[day] = by_date.get(day, 0.0) + value_cny
    for ts, total in by_date.items():
        rows.append(
            {
                "observation_date": ts,
                "channel": "north_total",
                "net_inflow_cny": total,
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["available_at"] = next_session_available_at(
        out["observation_date"], trading_calendar, lag_sessions=FLOW_AVAILABLE_AT_LAG_DAYS
    ).to_numpy()
    out["source"] = "akshare:stock_hsgt_fund_flow_summary_em"
    return out


def _normalize_margin_balance(
    combined: pd.DataFrame,
    *,
    trading_calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Normalise stacked margin-balance frames (SH / SZ)."""
    if combined is None or combined.empty:
        return pd.DataFrame()
    df = combined.copy()
    df.columns = [str(c).strip() for c in df.columns]
    required = {"observation_date", "market", "margin_balance_cny"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["margin_balance_cny"] = pd.to_numeric(
        df["margin_balance_cny"], errors="coerce"
    )
    if "short_balance_cny" in df.columns:
        df["short_balance_cny"] = pd.to_numeric(
            df["short_balance_cny"], errors="coerce"
        )
    df = df.dropna(subset=["observation_date", "margin_balance_cny"])
    df["available_at"] = next_session_available_at(
        df["observation_date"], trading_calendar, lag_sessions=FLOW_AVAILABLE_AT_LAG_DAYS
    ).to_numpy()
    df["source"] = "akshare:margin"
    return df


def _collect_northbound_history(ak_mod) -> pd.DataFrame:
    """Pull daily Stock-Connect northbound history for the three channels."""
    pieces: list[pd.DataFrame] = []
    for channel, symbol in (
        ("north_total", "北向资金"),
        ("north_hgt", "沪股通"),
        ("north_sgt", "深股通"),
    ):
        try:
            raw = _safe_call(ak_mod, "stock_hsgt_hist_em", symbol=symbol)
            if raw is None or raw.empty:
                continue
            df = raw.copy()
            df.columns = [str(c).strip() for c in df.columns]
            date_col = _first_match(df.columns, ("日期", "trade_date", "date"))
            net_col = _first_match(df.columns, ("当日成交净买额", "成交净买额"))
            if date_col is None or net_col is None:
                continue
            piece = pd.DataFrame(
                {
                    "observation_date": pd.to_datetime(
                        df[date_col], errors="coerce"
                    ),
                    "channel": channel,
                    # AKShare documents stock_hsgt_hist_em values in 亿元.
                    "net_inflow_cny": pd.to_numeric(
                        df[net_col], errors="coerce"
                    )
                    * 1e8,
                }
            ).dropna(subset=["observation_date", "net_inflow_cny"])
            pieces.append(piece)
        except Exception:
            continue
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _normalize_northbound_history(
    combined: pd.DataFrame,
    *,
    trading_calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Tag historical northbound rows with actual next-session availability."""
    if combined is None or combined.empty:
        return pd.DataFrame()
    out = combined.copy()
    out["observation_date"] = pd.to_datetime(
        out["observation_date"], errors="coerce"
    ).dt.normalize()
    out["available_at"] = next_session_available_at(
        out["observation_date"], trading_calendar, lag_sessions=FLOW_AVAILABLE_AT_LAG_DAYS
    ).to_numpy()
    out["source"] = "akshare:stock_hsgt_hist_em"
    return out


def _collect_margin(
    ak_mod,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Pull SH (and best-effort SZ) margin-financing balance history."""
    pieces: list[pd.DataFrame] = []
    sd = _compact(start_date) if start_date else "20100401"
    ed = _compact(end_date) if end_date else pd.Timestamp.today().strftime("%Y%m%d")
    try:
        raw = _safe_call(
            ak_mod, "stock_margin_sse", start_date=sd, end_date=ed
        )
        if raw is not None and not raw.empty:
            df = raw.copy()
            df.columns = [str(c).strip() for c in df.columns]
            date_col = _first_match(df.columns, ("信用交易日期", "日期", "date"))
            margin_col = _first_match(df.columns, ("融资融券余额", "融资余额"))
            short_col = _first_match(df.columns, ("融券余量金额", "融券余额"))
            if date_col is not None and margin_col is not None:
                pieces.append(
                    pd.DataFrame(
                        {
                            "observation_date": pd.to_datetime(
                                df[date_col].astype(str),
                                format="%Y%m%d",
                                errors="coerce",
                            ),
                            "market": "SH",
                            "margin_balance_cny": pd.to_numeric(
                                df[margin_col], errors="coerce"
                            ),
                            "short_balance_cny": (
                                pd.to_numeric(df[short_col], errors="coerce")
                                if short_col
                                else np.nan
                            ),
                        }
                    )
                )
    except Exception:
        pass
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _compact(date_str: str | None) -> str | None:
    if date_str is None:
        return None
    try:
        return pd.Timestamp(date_str).strftime("%Y%m%d")
    except Exception:
        return None


def _first_match(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    available = {str(c).strip(): str(c) for c in columns}
    for candidate in candidates:
        if candidate in available:
            return available[candidate]
    return None


def _safe_call(ak_mod, func_name: str, **kwargs):
    fn = getattr(ak_mod, func_name, None)
    if fn is None:
        raise AttributeError(f"akshare endpoint missing: {func_name}")
    return fn(**{k: value for k, value in kwargs.items() if value is not None})


__all__ = [
    "AkShareFlowProvider",
    "FLOW_AVAILABLE_AT_LAG_DAYS",
    "_normalize_northbound",
    "_normalize_northbound_history",
    "_normalize_margin_balance",
]
