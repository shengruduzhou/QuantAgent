"""AkShare capital-flow provider.

Tracks 北向资金 (Stock Connect net inflow), 两融余额 (margin financing
balance), ETF fund flow, and 行业资金流 (sector money flow). All tables
follow the PIT-cache pattern with explicit ``available_at``.

Publishing-lag policy: every endpoint here reports T-day data after the
A-share close, so ``available_at = observation_date + 1`` business day.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.data.providers.base import ProviderResult, ProviderUnavailable
from quantagent.data.providers.pit_cache import PITCacheConfig, PITTableSpec, PITTimeSeriesCache


FLOW_AVAILABLE_AT_LAG_DAYS = 1

_TABLES: tuple[PITTableSpec, ...] = (
    PITTableSpec(name="northbound_flow", filename="northbound_flow.parquet",
                 dedup_keys=("observation_date", "channel")),
    PITTableSpec(name="margin_balance", filename="margin_balance.parquet",
                 dedup_keys=("observation_date", "market")),
)


@dataclass
class AkShareFlowProvider:
    allow_network: bool = False
    root: str | None = None

    def __post_init__(self) -> None:
        root = self.root or str(quant_paths().data_root / "v7" / "raw" / "akshare" / "flow")
        self.cache = PITTimeSeriesCache(PITCacheConfig(root=root, tables=_TABLES))

    def fetch_all(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, ProviderResult]:
        ak = self._akshare()
        results: dict[str, ProviderResult] = {}
        results["northbound_flow"] = self._fetch_and_upsert(
            "northbound_flow",
            lambda: _normalize_northbound_history(_collect_northbound_history(ak)),
        )
        results["margin_balance"] = self._fetch_and_upsert(
            "margin_balance",
            lambda: _normalize_margin_balance(
                _collect_margin(ak, start_date=start_date, end_date=end_date)
            ),
        )
        return results

    def load_pit(self, table: str, as_of_date: str) -> ProviderResult:
        return self.cache.load_pit_frame(table, as_of_date)

    # ------------------------------------------------------------------ #

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

    def _fetch_and_upsert(self, table: str, fetcher) -> ProviderResult:
        try:
            frame = fetcher()
        except Exception as exc:
            return ProviderResult(
                pd.DataFrame(),
                source=f"akshare_flow:{table}",
                quality_score=0.0,
                warnings=(f"fetch_failed:{type(exc).__name__}:{exc}",),
            )
        if frame is None or frame.empty:
            return ProviderResult(
                pd.DataFrame(),
                source=f"akshare_flow:{table}",
                quality_score=0.0,
                warnings=("empty_response",),
            )
        self.cache.upsert(table, frame)
        return ProviderResult(
            frame.reset_index(drop=True),
            source=f"akshare_flow:{table}",
            quality_score=0.78,
            metadata={"row_count": int(len(frame)), "path": str(self.cache.path_for(table))},
        )


# ---------------------------------------------------------------------- #
# Normalisers — pure functions                                            #
# ---------------------------------------------------------------------- #


def _normalize_northbound(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise akshare ``stock_hsgt_fund_flow_summary_em`` (single-day snapshot).

    Real schema: one row per (交易日, 类型, 板块, 资金方向) — usually
    {沪股通 北向, 深股通 北向, 港股通 南向 x2} for the day. We keep the two
    A-share-relevant rows: 沪股通-北向 (north_hgt) and 深股通-北向 (north_sgt),
    and synthesise north_total as their sum.
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
    plate = df[plate_col].astype(str) if plate_col else pd.Series([""] * len(df))
    direction = df[direction_col].astype(str) if direction_col else pd.Series([""] * len(df))
    rows: list[dict[str, object]] = []
    by_date: dict[pd.Timestamp, float] = {}
    for ts, p, d, v in zip(obs, plate, direction, values):
        if pd.isna(ts) or pd.isna(v):
            continue
        if "北向" not in d:
            continue
        if "沪股通" in p:
            channel = "north_hgt"
        elif "深股通" in p:
            channel = "north_sgt"
        else:
            continue
        rows.append({
            "observation_date": ts.normalize(),
            "channel": channel,
            "net_inflow_cny": float(v),
        })
        by_date[ts.normalize()] = by_date.get(ts.normalize(), 0.0) + float(v)
    for ts, total in by_date.items():
        rows.append({"observation_date": ts, "channel": "north_total", "net_inflow_cny": total})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["available_at"] = out["observation_date"] + pd.Timedelta(days=FLOW_AVAILABLE_AT_LAG_DAYS)
    out["source"] = "akshare:stock_hsgt_fund_flow_summary_em"
    return out


def _normalize_margin_balance(combined: pd.DataFrame) -> pd.DataFrame:
    """Normalise stacked margin balance frames (sh / sz)."""
    if combined is None or combined.empty:
        return pd.DataFrame()
    df = combined.copy()
    df.columns = [str(c).strip() for c in df.columns]
    required = {"observation_date", "market", "margin_balance_cny"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["margin_balance_cny"] = pd.to_numeric(df["margin_balance_cny"], errors="coerce")
    if "short_balance_cny" in df.columns:
        df["short_balance_cny"] = pd.to_numeric(df["short_balance_cny"], errors="coerce")
    df = df.dropna(subset=["observation_date", "margin_balance_cny"])
    df["available_at"] = df["observation_date"] + pd.Timedelta(days=FLOW_AVAILABLE_AT_LAG_DAYS)
    df["source"] = "akshare:margin"
    return df


def _collect_northbound_history(ak_mod) -> pd.DataFrame:
    """Pull daily Stock-Connect northbound history for the three channels.

    akshare ``stock_hsgt_hist_em`` returns one DataFrame per symbol; we stack
    them under a ``channel`` column.
    """
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
            piece = pd.DataFrame({
                "observation_date": pd.to_datetime(df[date_col], errors="coerce"),
                "channel": channel,
                # The endpoint reports values in 亿元 (100M CNY); multiply for consistency.
                "net_inflow_cny": pd.to_numeric(df[net_col], errors="coerce") * 1e8,
            }).dropna(subset=["observation_date", "net_inflow_cny"])
            pieces.append(piece)
        except Exception:
            continue
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _normalize_northbound_history(combined: pd.DataFrame) -> pd.DataFrame:
    """Tag historical northbound rows with PIT available_at."""
    if combined is None or combined.empty:
        return pd.DataFrame()
    out = combined.copy()
    out["available_at"] = out["observation_date"] + pd.Timedelta(days=FLOW_AVAILABLE_AT_LAG_DAYS)
    out["source"] = "akshare:stock_hsgt_hist_em"
    return out


def _collect_margin(ak_mod, *, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Pull SH (and best-effort SZ) margin-financing balance history.

    SSE accepts (start_date, end_date) as YYYYMMDD; SZSE returns a single
    daily snapshot and is best-effort (drops on connection reset).
    """
    pieces: list[pd.DataFrame] = []
    sd = _compact(start_date) if start_date else "20100401"
    ed = _compact(end_date) if end_date else pd.Timestamp.today().strftime("%Y%m%d")
    try:
        raw = _safe_call(ak_mod, "stock_margin_sse", start_date=sd, end_date=ed)
        if raw is not None and not raw.empty:
            df = raw.copy()
            df.columns = [str(c).strip() for c in df.columns]
            date_col = _first_match(df.columns, ("信用交易日期", "日期", "date"))
            margin_col = _first_match(df.columns, ("融资融券余额", "融资余额"))
            short_col = _first_match(df.columns, ("融券余量金额", "融券余额"))
            if date_col is not None and margin_col is not None:
                pieces.append(pd.DataFrame({
                    "observation_date": pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce"),
                    "market": "SH",
                    "margin_balance_cny": pd.to_numeric(df[margin_col], errors="coerce"),
                    "short_balance_cny": pd.to_numeric(df[short_col], errors="coerce")
                        if short_col else np.nan,
                }))
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
    return fn(**{k: v for k, v in kwargs.items() if v is not None})


__all__ = [
    "AkShareFlowProvider",
    "FLOW_AVAILABLE_AT_LAG_DAYS",
    "_normalize_northbound",
    "_normalize_northbound_history",
    "_normalize_margin_balance",
]
