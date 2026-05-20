"""AkShare macro / bond / money-market provider.

Pulls the macro time-series that the V7 "national-team money flow" thesis
depends on: government yield curve, interbank rates (Shibor / DR / R),
central-bank open-market operations, aggregate financing, money supply,
CPI/PPI. Each table is stored under
``v7/raw/akshare/macro/<table>.parquet`` with an explicit
``available_at`` column following the publishing-lag policy below.

Publishing-lag policy (conservative):

* Daily curves & money-market rates (yield_curve, shibor, repo) →
  next-business-day available (publicly visible at T+1 close).
* Central-bank OMO (daily net injection) → T+1 (announced after market).
* Aggregate financing, money supply, CPI, PPI → +35 calendar days
  from observation date (PBoC / NBS typically release 10-20 days after
  month end; +35 days gives a healthy safety margin and avoids any
  preliminary-revision leakage).

All public methods are PIT-clean: ``fetch_*`` returns a normalised
DataFrame with ``available_at`` set, never the raw akshare frame. Tests
can call ``_normalize_*`` directly on hand-built akshare-shaped frames
without touching the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.data.providers.base import ProviderResult, ProviderUnavailable
from quantagent.data.providers.pit_cache import PITCacheConfig, PITTableSpec, PITTimeSeriesCache


MACRO_AVAILABLE_AT_LAG_DAYS = 35
DAILY_AVAILABLE_AT_LAG_DAYS = 1

_YIELD_CURVE_MATURITIES: tuple[str, ...] = ("1Y", "3Y", "5Y", "10Y", "30Y")
_SHIBOR_TENORS: tuple[str, ...] = ("O/N", "1W", "1M", "3M", "6M", "1Y")
_REPO_TENORS: tuple[str, ...] = ("DR007", "R007")

_TABLES: tuple[PITTableSpec, ...] = (
    PITTableSpec(name="yield_curve", filename="yield_curve.parquet",
                 dedup_keys=("observation_date", "maturity")),
    PITTableSpec(name="shibor", filename="shibor.parquet",
                 dedup_keys=("observation_date", "tenor")),
    PITTableSpec(name="repo", filename="repo.parquet",
                 dedup_keys=("observation_date", "tenor")),
    PITTableSpec(name="central_bank_balance", filename="central_bank_balance.parquet",
                 dedup_keys=("observation_date",)),
    PITTableSpec(name="aggregate_financing", filename="aggregate_financing.parquet",
                 dedup_keys=("observation_date",)),
    PITTableSpec(name="money_supply", filename="money_supply.parquet",
                 dedup_keys=("observation_date",)),
    PITTableSpec(name="cpi", filename="cpi.parquet",
                 dedup_keys=("observation_date",)),
    PITTableSpec(name="ppi", filename="ppi.parquet",
                 dedup_keys=("observation_date",)),
)


@dataclass
class AkShareMacroProvider:
    allow_network: bool = False
    root: str | None = None

    def __post_init__(self) -> None:
        root = self.root or str(quant_paths().data_root / "v7" / "raw" / "akshare" / "macro")
        self.cache = PITTimeSeriesCache(PITCacheConfig(root=root, tables=_TABLES))

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def fetch_all(self, start_date: str | None = None, end_date: str | None = None) -> dict[str, ProviderResult]:
        """Fetch every macro table and persist into the PIT cache."""
        ak = self._akshare()
        results: dict[str, ProviderResult] = {}
        # Yield-curve API limits each call to a roughly 1-year window, so chunk.
        yield_frames: list[pd.DataFrame] = []
        for chunk_start, chunk_end in _year_chunks(start_date, end_date):
            try:
                raw = _safe_call(ak, "bond_china_yield",
                                 start_date=_compact(chunk_start),
                                 end_date=_compact(chunk_end))
                yield_frames.append(_normalize_yield_curve(raw))
            except Exception:
                continue
        results["yield_curve"] = self._upsert_combined("yield_curve", yield_frames)
        # Shibor: one row per (date, tenor); fetch each tenor sequentially.
        shibor_frames: list[pd.DataFrame] = []
        for tenor in _SHIBOR_TENORS:
            try:
                raw = _safe_call(ak, "rate_interbank",
                                 market="上海银行同业拆借市场",
                                 symbol="Shibor人民币",
                                 indicator=_shibor_tenor_label(tenor))
                shibor_frames.append(_normalize_shibor(raw, tenor=tenor))
            except Exception:
                continue
        results["shibor"] = self._upsert_combined("shibor", shibor_frames)

        # repo_rate_hist breaks if only start_date is passed (KeyError 'frValueMap'
        # in akshare 1.18); call with no args to get the recent window safely.
        results["repo"] = self._fetch_and_upsert(
            "repo", lambda: _normalize_repo(_safe_call(ak, "repo_rate_hist")),
        )
        results["central_bank_balance"] = self._fetch_and_upsert(
            "central_bank_balance",
            lambda: _normalize_central_bank_balance(_safe_call(ak, "macro_china_central_bank_balance")),
        )
        results["aggregate_financing"] = self._fetch_and_upsert(
            "aggregate_financing",
            lambda: _normalize_aggregate_financing(_safe_call(ak, "macro_china_shrzgm")),
        )
        results["money_supply"] = self._fetch_and_upsert(
            "money_supply",
            lambda: _normalize_money_supply(_safe_call(ak, "macro_china_money_supply")),
        )
        results["cpi"] = self._fetch_and_upsert(
            "cpi", lambda: _normalize_cpi_ppi(_safe_call(ak, "macro_china_cpi_yearly"), kind="cpi"),
        )
        results["ppi"] = self._fetch_and_upsert(
            "ppi", lambda: _normalize_cpi_ppi(_safe_call(ak, "macro_china_ppi_yearly"), kind="ppi"),
        )
        return results

    def _upsert_combined(self, table: str, frames: list[pd.DataFrame]) -> ProviderResult:
        non_empty = [f for f in frames if f is not None and not f.empty]
        if not non_empty:
            return ProviderResult(
                pd.DataFrame(), source=f"akshare_macro:{table}", quality_score=0.0,
                warnings=("empty_response",),
            )
        combined = pd.concat(non_empty, ignore_index=True)
        self.cache.upsert(table, combined)
        return ProviderResult(
            combined.reset_index(drop=True), source=f"akshare_macro:{table}",
            quality_score=0.78,
            metadata={"row_count": int(len(combined)), "path": str(self.cache.path_for(table))},
        )

    def load_pit(self, table: str, as_of_date: str) -> ProviderResult:
        return self.cache.load_pit_frame(table, as_of_date)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _akshare(self):
        if not self.allow_network:
            raise ProviderUnavailable(
                "AkShareMacroProvider network disabled; set allow_network=True explicitly"
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
                source=f"akshare_macro:{table}",
                quality_score=0.0,
                warnings=(f"fetch_failed:{type(exc).__name__}:{exc}",),
            )
        if frame.empty:
            return ProviderResult(
                pd.DataFrame(),
                source=f"akshare_macro:{table}",
                quality_score=0.0,
                warnings=("empty_response",),
            )
        self.cache.upsert(table, frame)
        return ProviderResult(
            frame.reset_index(drop=True),
            source=f"akshare_macro:{table}",
            quality_score=0.78,
            metadata={"row_count": int(len(frame)), "path": str(self.cache.path_for(table))},
        )


# ---------------------------------------------------------------------- #
# Normalisers — pure functions, fully unit-testable without network       #
# ---------------------------------------------------------------------- #


_YIELD_CURVE_CHINESE_LABEL = {
    "1Y": "1年",
    "3Y": "3年",
    "5Y": "5年",
    "10Y": "10年",
    "30Y": "30年",
}
_YIELD_CURVE_TREASURY_NAME = "中债国债收益率曲线"


def _normalize_yield_curve(raw: pd.DataFrame) -> pd.DataFrame:
    """Map akshare ``bond_china_yield`` to our PIT schema.

    The endpoint returns multiple curves per date (国债 / 中短期票据 / 商业银行普通债 …).
    We keep only the government-bond curve ("中债国债收益率曲线") so each
    (observation_date, maturity) cell is unique.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = _first_match(df.columns, ("日期", "date", "trade_date", "observation_date"))
    if date_col is None:
        return pd.DataFrame()
    curve_col = _first_match(df.columns, ("曲线名称", "curve_name"))
    if curve_col is not None:
        treasury_mask = df[curve_col].astype(str).str.contains(_YIELD_CURVE_TREASURY_NAME, na=False)
        if treasury_mask.any():
            df = df[treasury_mask].copy()
    rows: list[dict[str, object]] = []
    observation = pd.to_datetime(df[date_col], errors="coerce")
    for maturity in _YIELD_CURVE_MATURITIES:
        for candidate in (
            _YIELD_CURVE_CHINESE_LABEL.get(maturity, maturity),
            f"{maturity}",
            f"中债{maturity}",
            f"中债国债到期收益率_{maturity}",
            f"中债国债到期收益率{maturity}",
        ):
            if candidate in df.columns:
                series = pd.to_numeric(df[candidate], errors="coerce")
                for ts, val in zip(observation, series):
                    if pd.isna(ts) or pd.isna(val):
                        continue
                    rows.append({
                        "observation_date": ts.normalize(),
                        "maturity": maturity,
                        "yield_pct": float(val),
                    })
                break
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["observation_date", "maturity"], keep="last")
    out["available_at"] = out["observation_date"] + pd.Timedelta(days=DAILY_AVAILABLE_AT_LAG_DAYS)
    out["source"] = "akshare:bond_china_yield"
    return out


def _normalize_shibor(raw: pd.DataFrame, tenor: str = "O/N") -> pd.DataFrame:
    """Normalise akshare ``rate_interbank`` for one Shibor tenor."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = _first_match(df.columns, ("报告日", "日期", "date"))
    rate_col = _first_match(df.columns, ("利率", "rate", "value"))
    if date_col is None or rate_col is None:
        return pd.DataFrame()
    observation = pd.to_datetime(df[date_col], errors="coerce")
    rate = pd.to_numeric(df[rate_col], errors="coerce")
    out = pd.DataFrame({
        "observation_date": observation,
        "tenor": tenor,
        "rate_pct": rate,
    }).dropna(subset=["observation_date", "rate_pct"])
    out["available_at"] = out["observation_date"] + pd.Timedelta(days=DAILY_AVAILABLE_AT_LAG_DAYS)
    out["source"] = "akshare:rate_interbank"
    return out


def _normalize_repo(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise akshare ``repo_rate_hist`` (wide → long for key tenors).

    Source schema: ``date, FR001, FR007, FR014, FDR001, FDR007, FDR014``.
    We keep FR007 (interbank repo 7d) and FDR007 (depository repo 7d, DR007 proxy)
    as the two policy-monitored benchmarks.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = _first_match(df.columns, ("date", "日期"))
    if date_col is None:
        return pd.DataFrame()
    obs = pd.to_datetime(df[date_col], errors="coerce")
    rows: list[dict[str, object]] = []
    tenor_map = {"FR007": "FR007", "FDR007": "DR007", "FR001": "FR001", "FDR001": "DR001"}
    for source_col, tenor_label in tenor_map.items():
        if source_col in df.columns:
            values = pd.to_numeric(df[source_col], errors="coerce")
            for ts, val in zip(obs, values):
                if pd.isna(ts) or pd.isna(val):
                    continue
                rows.append({
                    "observation_date": ts.normalize(),
                    "tenor": tenor_label,
                    "rate_pct": float(val),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["available_at"] = out["observation_date"] + pd.Timedelta(days=DAILY_AVAILABLE_AT_LAG_DAYS)
    out["source"] = "akshare:repo_rate_hist"
    return out


def _normalize_central_bank_balance(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise akshare ``macro_china_central_bank_balance`` (PBoC balance sheet).

    Replaces the obsolete OMO endpoint. We keep the top-line aggregates that
    most directly express "国家队" liquidity stance:
      ``总资产`` (total assets), ``储备货币`` (reserve money),
      ``国外资产`` (foreign assets), ``对其他存款性公司债权`` (claims on
      depository corporations — PBoC's open-market injection proxy).
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = _first_match(df.columns, ("统计时间", "月份", "日期", "date"))
    if date_col is None:
        return pd.DataFrame()
    # 统计时间 looks like "2026.4" — parse as YYYY-MM with day=last.
    obs = df[date_col].astype(str).apply(_parse_yearmonth)
    keep = {
        "总资产": "total_assets_cny",
        "储备货币": "reserve_money_cny",
        "国外资产": "foreign_assets_cny",
        "对其他存款性公司债权": "claims_on_depository_corp_cny",
        "对政府债权": "claims_on_government_cny",
    }
    payload: dict[str, object] = {"observation_date": obs}
    for source_col, target in keep.items():
        if source_col in df.columns:
            payload[target] = pd.to_numeric(df[source_col], errors="coerce")
    out = pd.DataFrame(payload).dropna(subset=["observation_date"])
    if out.empty:
        return out
    out["available_at"] = out["observation_date"] + pd.Timedelta(days=MACRO_AVAILABLE_AT_LAG_DAYS)
    out["source"] = "akshare:macro_china_central_bank_balance"
    return out


def _normalize_aggregate_financing(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise akshare ``macro_china_shrzgm`` (社会融资规模存量).

    Real ``月份`` is a compact ``YYYYMM`` string (e.g. ``"201501"``);
    parse via :func:`_parse_yearmonth`.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = _first_match(df.columns, ("月份", "日期", "date"))
    total_col = _first_match(df.columns, ("社会融资规模增量", "社融规模", "社融", "value"))
    if date_col is None or total_col is None:
        return pd.DataFrame()
    obs = df[date_col].apply(_parse_yearmonth)
    total = pd.to_numeric(df[total_col], errors="coerce")
    out = pd.DataFrame({"observation_date": obs, "aggregate_financing_cny": total}).dropna()
    if out.empty:
        return out
    out["available_at"] = out["observation_date"] + pd.Timedelta(days=MACRO_AVAILABLE_AT_LAG_DAYS)
    out["source"] = "akshare:macro_china_shrzgm"
    return out


def _normalize_money_supply(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise akshare ``macro_china_money_supply`` (M0/M1/M2).

    Real akshare columns: ``月份`` like ``2026年04月份`` plus
    ``货币和准货币(M2)-数量(亿元)``, ``货币(M1)-数量(亿元)``,
    ``流通中的现金(M0)-数量(亿元)``. We also keep the YoY-growth columns
    as separate features since the level alone is dominated by trend.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = _first_match(df.columns, ("月份", "日期", "date"))
    if date_col is None:
        return pd.DataFrame()
    obs = df[date_col].astype(str).apply(_parse_yearmonth)
    m0_col = _first_match(df.columns,
                          ("流通中的现金(M0)-数量(亿元)", "货币和准货币-M0", "M0"))
    m1_col = _first_match(df.columns,
                          ("货币(M1)-数量(亿元)", "货币-M1", "M1"))
    m2_col = _first_match(df.columns,
                          ("货币和准货币(M2)-数量(亿元)", "货币和准货币-M2", "M2"))
    m0_yoy = _first_match(df.columns, ("流通中的现金(M0)-同比增长",))
    m1_yoy = _first_match(df.columns, ("货币(M1)-同比增长",))
    m2_yoy = _first_match(df.columns, ("货币和准货币(M2)-同比增长",))
    out = pd.DataFrame({
        "observation_date": obs,
        "m0_cny": pd.to_numeric(df[m0_col], errors="coerce") if m0_col else np.nan,
        "m1_cny": pd.to_numeric(df[m1_col], errors="coerce") if m1_col else np.nan,
        "m2_cny": pd.to_numeric(df[m2_col], errors="coerce") if m2_col else np.nan,
        "m0_yoy_pct": pd.to_numeric(df[m0_yoy], errors="coerce") if m0_yoy else np.nan,
        "m1_yoy_pct": pd.to_numeric(df[m1_yoy], errors="coerce") if m1_yoy else np.nan,
        "m2_yoy_pct": pd.to_numeric(df[m2_yoy], errors="coerce") if m2_yoy else np.nan,
    }).dropna(subset=["observation_date"])
    out["available_at"] = out["observation_date"] + pd.Timedelta(days=MACRO_AVAILABLE_AT_LAG_DAYS)
    out["source"] = "akshare:macro_china_money_supply"
    return out


def _normalize_cpi_ppi(raw: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    """Normalise akshare ``macro_china_cpi_yearly`` / ``macro_china_ppi_yearly``."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = _first_match(df.columns, ("日期", "时间", "date", "observation_date"))
    value_col = _first_match(df.columns, ("今值", "value", "current_value", "今值(%)"))
    if date_col is None or value_col is None:
        return pd.DataFrame()
    obs = pd.to_datetime(df[date_col], errors="coerce")
    value = pd.to_numeric(df[value_col], errors="coerce")
    out = pd.DataFrame({
        "observation_date": obs,
        f"{kind}_yoy_pct": value,
    }).dropna()
    out["available_at"] = out["observation_date"] + pd.Timedelta(days=MACRO_AVAILABLE_AT_LAG_DAYS)
    out["source"] = f"akshare:macro_china_{kind}_yearly"
    return out


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #


def _first_match(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    available = {str(c).strip(): str(c) for c in columns}
    for candidate in candidates:
        if candidate in available:
            return available[candidate]
    # case-insensitive lower-cased fallback
    lower = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _compact(date_str: str | None) -> str | None:
    if date_str is None:
        return None
    try:
        return pd.Timestamp(date_str).strftime("%Y%m%d")
    except Exception:
        return None


def _dashed(date_str: str | None) -> str | None:
    if date_str is None:
        return None
    try:
        return pd.Timestamp(date_str).strftime("%Y-%m-%d")
    except Exception:
        return None


def _safe_call(ak_mod, func_name: str, **kwargs):
    fn = getattr(ak_mod, func_name, None)
    if fn is None:
        raise AttributeError(f"akshare endpoint missing: {func_name}")
    return fn(**{k: v for k, v in kwargs.items() if v is not None})


def _shibor_tenor_label(tenor: str) -> str:
    mapping = {"O/N": "隔夜", "1W": "1周", "1M": "1月", "3M": "3月", "6M": "6月", "1Y": "1年"}
    return mapping.get(tenor, tenor)


def _parse_yearmonth(value: object) -> pd.Timestamp:
    """Parse akshare's mixed Chinese yearmonth strings into a Timestamp.

    Accepts ``'2026年04月份'``, ``'2026.4'``, ``'2026-04'``, ``'2026/04'``,
    or the bare ``'201501'`` form used by ``macro_china_shrzgm``. Returns
    NaT on failure. The day is normalised to the month end so PIT
    ``available_at`` calculations behave consistently for monthly aggregates.
    """
    if value is None:
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    # YYYYMM compact form ("201501")
    if text.isdigit() and len(text) == 6:
        text = f"{text[:4]}-{text[4:]}"
    # Chinese form "2026年04月份"
    if "年" in text:
        text = text.replace("年", "-").replace("月份", "").replace("月", "")
    # Dot form "2026.4"
    if "." in text and "-" not in text:
        text = text.replace(".", "-")
    try:
        ts = pd.Timestamp(text + "-01") if len(text.split("-")) == 2 else pd.Timestamp(text)
        if pd.isna(ts):
            return pd.NaT
        return (ts + pd.offsets.MonthEnd(0)).normalize()
    except Exception:
        return pd.NaT


def _year_chunks(start_date: str | None, end_date: str | None) -> list[tuple[str, str]]:
    """Yield (chunk_start, chunk_end) pairs of at most ~1 calendar year.

    Used for endpoints (e.g. bond_china_yield) that silently return 0 rows
    when the requested window spans more than a year. Empty input → a
    single (None, None) call so the caller still attempts at least once.
    """
    if not start_date and not end_date:
        return [(None, None)]
    try:
        sd = pd.Timestamp(start_date) if start_date else pd.Timestamp("2010-01-01")
        ed = pd.Timestamp(end_date) if end_date else pd.Timestamp.today()
    except Exception:
        return [(start_date, end_date)]
    chunks: list[tuple[str, str]] = []
    cursor = sd
    while cursor <= ed:
        next_end = min(cursor + pd.DateOffset(years=1) - pd.Timedelta(days=1), ed)
        chunks.append((cursor.strftime("%Y-%m-%d"), next_end.strftime("%Y-%m-%d")))
        cursor = next_end + pd.Timedelta(days=1)
    return chunks


__all__ = [
    "AkShareMacroProvider",
    "MACRO_AVAILABLE_AT_LAG_DAYS",
    "DAILY_AVAILABLE_AT_LAG_DAYS",
    "_normalize_yield_curve",
    "_normalize_shibor",
    "_normalize_repo",
    "_normalize_central_bank_balance",
    "_normalize_aggregate_financing",
    "_normalize_money_supply",
    "_normalize_cpi_ppi",
    "_parse_yearmonth",
]
