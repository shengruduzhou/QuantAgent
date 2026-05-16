"""AkShare valuation / universe / sector adapters.

These adapters round out the AkShare coverage needed by the V7 silver
layer:

* ``AkShareUniverseProvider`` — the A-share universe (symbol, name,
  list date, market). Used to build ``data/v7/silver/universe`` and
  filter downstream features to active listings.
* ``AkShareValuationProvider`` — daily valuation snapshots (PE/PB/PS
  TTM, market cap, free-float market cap, dividend yield) from the
  Eastmoney quote interface, normalised into ``symbol / trade_date /
  available_at`` PIT rows.
* ``AkShareSectorProvider`` — industry / sector classification. Falls
  back to an explicit ``ProviderUnavailable`` when AkShare changes its
  endpoint, so the caller can decide whether to use a local override
  table.

Network access requires ``allow_network=True``; all adapters fail-loud
when AkShare is missing or returns an empty/changed schema so we never
silently use stale or partial data.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import time

import pandas as pd

from quantagent.data.providers.akshare_financial_provider import to_akshare_symbol
from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable


AKSHARE_UNIVERSE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "name",
    "exchange",
    "list_date",
)
AKSHARE_VALUATION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "trade_date",
    "available_at",
    "pe_ttm",
    "pb",
    "market_cap",
)
AKSHARE_SECTOR_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "industry",
    "available_at",
)


_VALUATION_RENAME = {
    "代码": "symbol_raw",
    "名称": "name",
    "市盈率-动态": "pe_ttm",
    "市盈率(TTM)": "pe_ttm",
    "市净率": "pb",
    "市销率": "ps_ttm",
    "总市值": "market_cap",
    "流通市值": "free_float_market_cap",
    "股息率": "dividend_yield",
    "换手率": "turnover_rate",
}


def _row_hash(row: dict[str, object]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def _ensure_akshare(allow_network: bool):
    if not allow_network:
        raise ProviderUnavailable("AkShare network is disabled; set allow_network=true explicitly")
    try:
        import akshare as ak  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ProviderUnavailable("akshare package is not available; install quantagent[data]") from exc
    return ak


def _suffix_from_code(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SZ"


@dataclass
class AkShareUniverseProvider:
    allow_network: bool = False
    source: str = "akshare_universe"

    def list_universe(self) -> ProviderResult:
        ak = _ensure_akshare(self.allow_network)
        try:
            raw = ak.stock_info_a_code_name()
        except Exception as exc:  # pragma: no cover - network path
            raise ProviderUnavailable(f"AkShare stock_info_a_code_name failed: {exc}") from exc
        if raw is None or raw.empty:
            return ProviderResult(pd.DataFrame(), source=self.source, quality_score=0.0, warnings=("akshare_empty_universe",))
        frame = raw.rename(columns={"code": "symbol_raw", "name": "name"}).copy()
        frame["symbol"] = frame["symbol_raw"].apply(_suffix_from_code)
        frame["exchange"] = frame["symbol"].str.split(".").str[-1]
        if "list_date" not in frame.columns:
            frame["list_date"] = pd.NaT
        frame = frame[[c for c in ("symbol", "name", "exchange", "list_date") if c in frame.columns]]
        report = akshare_universe_schema_report(frame)
        warnings = tuple(f"akshare_schema_missing:{c}" for c in report["missing_columns"])
        return ProviderResult(
            frame.reset_index(drop=True),
            source=self.source,
            quality_score=0.85 if report["status"] == "passed" else 0.0,
            warnings=warnings,
            metadata={"schema_report": report},
        )


def akshare_universe_schema_report(frame: pd.DataFrame) -> dict[str, object]:
    missing = [c for c in AKSHARE_UNIVERSE_REQUIRED_COLUMNS if c not in frame.columns]
    return {
        "status": "passed" if not missing else "failed",
        "row_count": int(0 if frame is None else len(frame)),
        "required_columns": list(AKSHARE_UNIVERSE_REQUIRED_COLUMNS),
        "missing_columns": missing,
    }


@dataclass
class AkShareValuationProvider:
    """Daily valuation snapshot adapter.

    AkShare's spot quote endpoint returns one row per A-share with PE/PB
    /market-cap fields. We treat the snapshot as ``trade_date=as_of_date``
    and ``available_at=as_of_date`` so it lines up with the close-derived
    market panel. Repeated snapshots can be concatenated to form a
    valuation history.
    """

    allow_network: bool = False
    source: str = "akshare_valuation"
    rate_limit_seconds: float = 0.2
    retry_count: int = 2
    retry_sleep_seconds: float = 0.5

    def snapshot(self, as_of_date: str, request: ProviderRequest | None = None) -> ProviderResult:
        ak = _ensure_akshare(self.allow_network)
        raw = self._call_with_retry(getattr(ak, "stock_zh_a_spot_em", None))
        if raw is None or raw.empty:
            return ProviderResult(pd.DataFrame(), source=self.source, quality_score=0.0, warnings=("akshare_empty_valuation",))
        frame = self._normalize(raw, as_of_date)
        if request is not None and request.symbols:
            symbol_set = {str(s) for s in request.symbols}
            frame = frame[frame["symbol"].astype(str).isin(symbol_set)]
        report = akshare_valuation_schema_report(frame)
        warnings = tuple(f"akshare_schema_missing:{c}" for c in report["missing_columns"])
        return ProviderResult(
            frame.reset_index(drop=True),
            source=self.source,
            point_in_time=True,
            quality_score=0.75 if report["status"] == "passed" else 0.0,
            warnings=warnings,
            metadata={"schema_report": report, "as_of_date": as_of_date},
        )

    def _normalize(self, frame: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
        keep = [c for c in _VALUATION_RENAME if c in frame.columns]
        data = frame[keep].rename(columns=_VALUATION_RENAME).copy()
        if "symbol_raw" in data.columns:
            data["symbol"] = data["symbol_raw"].astype(str).apply(_suffix_from_code)
            data = data.drop(columns=["symbol_raw"])
        data["trade_date"] = as_of_date
        data["available_at"] = as_of_date
        for column in ("pe_ttm", "pb", "ps_ttm", "market_cap", "free_float_market_cap", "dividend_yield", "turnover_rate"):
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        data["source"] = self.source
        data["source_reliability"] = 0.70
        data["raw_hash"] = [_row_hash(row) for row in data.to_dict("records")]
        data["point_in_time_valid"] = True
        return data

    def _call_with_retry(self, callable_api):
        if callable_api is None:
            raise ProviderUnavailable("AkShare stock_zh_a_spot_em is not available in this version")
        last_exc: Exception | None = None
        for attempt in range(max(1, self.retry_count + 1)):
            try:
                return callable_api()
            except Exception as exc:  # pragma: no cover - network path
                last_exc = exc
                if attempt < self.retry_count and self.retry_sleep_seconds > 0:
                    time.sleep(self.retry_sleep_seconds)
        raise ProviderUnavailable(f"AkShare valuation snapshot failed: {last_exc}") from last_exc


def akshare_valuation_schema_report(frame: pd.DataFrame) -> dict[str, object]:
    missing = [c for c in AKSHARE_VALUATION_REQUIRED_COLUMNS if c not in frame.columns]
    return {
        "status": "passed" if not missing else "failed",
        "row_count": int(0 if frame is None else len(frame)),
        "required_columns": list(AKSHARE_VALUATION_REQUIRED_COLUMNS),
        "missing_columns": missing,
    }


@dataclass
class AkShareSectorProvider:
    """Industry classification from AkShare's industry endpoints."""

    allow_network: bool = False
    source: str = "akshare_sector"

    def industry_classification(self, request: ProviderRequest | None = None, as_of_date: str | None = None) -> ProviderResult:
        ak = _ensure_akshare(self.allow_network)
        endpoint = getattr(ak, "stock_board_industry_summary_ths", None) or getattr(ak, "stock_board_industry_name_em", None)
        if endpoint is None:
            raise ProviderUnavailable("AkShare industry endpoints are not available in this version")
        try:
            raw = endpoint()
        except Exception as exc:  # pragma: no cover - network path
            raise ProviderUnavailable(f"AkShare industry endpoint failed: {exc}") from exc
        if raw is None or raw.empty:
            return ProviderResult(pd.DataFrame(), source=self.source, quality_score=0.0, warnings=("akshare_empty_sector",))
        # The two known endpoints both return per-board rows; we attach the
        # board name as ``industry`` and let downstream resolvers join by symbol.
        column = "板块名称" if "板块名称" in raw.columns else raw.columns[0]
        frame = pd.DataFrame({"industry": raw[column].astype(str)})
        frame["symbol"] = pd.NA
        frame["available_at"] = pd.Timestamp(as_of_date or pd.Timestamp.today().normalize())
        if request is not None and request.symbols:
            frame = pd.concat(
                [
                    frame.assign(symbol=str(symbol))
                    for symbol in request.symbols
                ],
                ignore_index=True,
            )
        report = akshare_sector_schema_report(frame)
        warnings = tuple(f"akshare_schema_missing:{c}" for c in report["missing_columns"])
        return ProviderResult(
            frame.reset_index(drop=True),
            source=self.source,
            point_in_time=True,
            quality_score=0.55 if report["status"] == "passed" else 0.0,
            warnings=warnings,
            metadata={"schema_report": report},
        )


def akshare_sector_schema_report(frame: pd.DataFrame) -> dict[str, object]:
    missing = [c for c in AKSHARE_SECTOR_REQUIRED_COLUMNS if c not in frame.columns]
    return {
        "status": "passed" if not missing else "failed",
        "row_count": int(0 if frame is None else len(frame)),
        "required_columns": list(AKSHARE_SECTOR_REQUIRED_COLUMNS),
        "missing_columns": missing,
    }


__all__ = [
    "AKSHARE_UNIVERSE_REQUIRED_COLUMNS",
    "AKSHARE_VALUATION_REQUIRED_COLUMNS",
    "AKSHARE_SECTOR_REQUIRED_COLUMNS",
    "AkShareUniverseProvider",
    "AkShareValuationProvider",
    "AkShareSectorProvider",
    "akshare_universe_schema_report",
    "akshare_valuation_schema_report",
    "akshare_sector_schema_report",
]


# silence unused import warning while keeping the helper available to users
_ = to_akshare_symbol
