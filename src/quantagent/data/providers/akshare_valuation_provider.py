"""AkShare valuation / universe / sector adapters.

Network access requires ``allow_network=True``. The sector adapter resolves
symbol-level membership through board constituent endpoints or a local mapping;
it never cross-joins every industry onto every symbol.
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
    "市盈率-TTM": "pe_ttm",
    "市净率": "pb",
    "市净率-MRQ": "pb",
    "市销率": "ps_ttm",
    "市销率-TTM": "ps_ttm",
    "总市值": "market_cap",
    "流通市值": "free_float_market_cap",
    "股息率": "dividend_yield",
    "换手率": "turnover_rate",
    "PEG": "peg",
    "EV/EBITDA-24A": "ev_ebitda",
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
        frame = raw.rename(columns={"code": "symbol_raw", "name": "name", "代码": "symbol_raw", "名称": "name"}).copy()
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
    """Daily valuation snapshot adapter from AkShare spot quote data."""

    allow_network: bool = False
    source: str = "akshare_valuation"
    rate_limit_seconds: float = 0.2
    retry_count: int = 2
    retry_sleep_seconds: float = 0.5

    def snapshot(self, as_of_date: str, request: ProviderRequest | None = None) -> ProviderResult:
        ak = _ensure_akshare(self.allow_network)
        warnings: list[str] = []
        try:
            raw = self._call_with_retry(getattr(ak, "stock_zh_a_spot_em", None))
        except ProviderUnavailable as exc:
            if request is None or not request.symbols:
                raise
            warnings.append(f"akshare_valuation_spot_em_failed:{exc}")
            raw = self._fallback_symbol_snapshots(ak, request.symbols, as_of_date, warnings)
        if raw is None or raw.empty:
            return ProviderResult(pd.DataFrame(), source=self.source, quality_score=0.0, warnings=tuple(warnings or ["akshare_empty_valuation"]))
        frame = self._normalize(raw, as_of_date)
        if request is not None and request.symbols:
            symbol_set = {str(s) for s in request.symbols}
            frame = frame[frame["symbol"].astype(str).isin(symbol_set)]
        report = akshare_valuation_schema_report(frame)
        warnings.extend(f"akshare_schema_missing:{c}" for c in report["missing_columns"])
        return ProviderResult(
            frame.reset_index(drop=True),
            source=self.source,
            point_in_time=True,
            quality_score=0.75 if report["status"] == "passed" else 0.0,
            warnings=tuple(warnings),
            metadata={
                "schema_report": report,
                "as_of_date": as_of_date,
                "function_name": "stock_zh_a_spot_em|stock_individual_info_em|stock_zh_valuation_comparison_em",
            },
        )

    def _normalize(self, frame: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
        keep_candidates = tuple(dict.fromkeys(tuple(_VALUATION_RENAME) + tuple(_VALUATION_RENAME.values())))
        keep = [c for c in keep_candidates if c in frame.columns]
        data = frame[keep].rename(columns=_VALUATION_RENAME).copy()
        if "symbol_raw" in data.columns:
            data["symbol"] = data["symbol_raw"].astype(str).apply(_suffix_from_code)
            data = data.drop(columns=["symbol_raw"])
        data["trade_date"] = as_of_date
        data["available_at"] = as_of_date
        for column in ("pe_ttm", "pb", "ps_ttm", "peg", "ev_ebitda", "market_cap", "free_float_market_cap", "dividend_yield", "turnover_rate"):
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        data["source"] = self.source
        data["source_reliability"] = 0.70
        data["raw_hash"] = [_row_hash(row) for row in data.to_dict("records")]
        data["point_in_time_valid"] = True
        return data

    def _fallback_symbol_snapshots(
        self,
        ak: object,
        symbols: tuple[str, ...],
        as_of_date: str,
        warnings: list[str],
    ) -> pd.DataFrame:
        info_api = getattr(ak, "stock_individual_info_em", None)
        comparison_api = getattr(ak, "stock_zh_valuation_comparison_em", None)
        if info_api is None or comparison_api is None:
            warnings.append("akshare_valuation_symbol_fallback_unavailable")
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            try:
                info = self._call_with_retry(lambda: info_api(symbol=_plain_code(symbol)))  # type: ignore[misc]
                comparison = self._call_with_retry(lambda: comparison_api(symbol=_em_symbol(symbol)))  # type: ignore[misc]
            except ProviderUnavailable as exc:
                warnings.append(f"akshare_valuation_symbol_fallback_failed:{symbol}:{exc}")
                continue
            row = _valuation_row_from_symbol_frames(symbol, info, comparison, as_of_date)
            if row:
                frames.append(pd.DataFrame([row]))
            if self.rate_limit_seconds > 0:
                time.sleep(self.rate_limit_seconds)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

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


def _plain_code(symbol: str) -> str:
    text = str(symbol).split(".")[0]
    lower = text.lower()
    for prefix in ("sh", "sz", "bj"):
        if lower.startswith(prefix):
            return text[len(prefix):]
    return text.zfill(6)


def _em_symbol(symbol: str) -> str:
    code = _plain_code(symbol)
    upper = str(symbol).upper()
    if upper.endswith(".SH") or upper.startswith("SH") or code.startswith(("6", "9")):
        return f"SH{code}"
    if upper.endswith(".BJ") or upper.startswith("BJ") or code.startswith(("4", "8")):
        return f"BJ{code}"
    return f"SZ{code}"


def _series_from_item_value(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty or not {"item", "value"}.issubset(frame.columns):
        return pd.Series(dtype=object)
    return frame.dropna(subset=["item"]).drop_duplicates(subset=["item"], keep="last").set_index("item")["value"]


def _valuation_row_from_symbol_frames(
    symbol: str,
    info: pd.DataFrame,
    comparison: pd.DataFrame,
    as_of_date: str,
) -> dict[str, object]:
    info_values = _series_from_item_value(info)
    code = _plain_code(symbol)
    comparison_row = pd.Series(dtype=object)
    if comparison is not None and not comparison.empty:
        data = comparison.copy()
        if "代码" in data.columns:
            matched = data[data["代码"].astype(str).str.zfill(6) == code]
            comparison_row = matched.iloc[0] if not matched.empty else data.iloc[0]
        else:
            comparison_row = data.iloc[0]
    row = {
        "symbol_raw": code,
        "name": info_values.get("股票简称", comparison_row.get("简称")),
        "trade_date": as_of_date,
        "available_at": as_of_date,
        "market_cap": info_values.get("总市值"),
        "free_float_market_cap": info_values.get("流通市值"),
        "pe_ttm": comparison_row.get("市盈率-TTM"),
        "pb": comparison_row.get("市净率-MRQ"),
        "ps_ttm": comparison_row.get("市销率-TTM"),
        "peg": comparison_row.get("PEG"),
        "ev_ebitda": comparison_row.get("EV/EBITDA-24A"),
    }
    return {key: value for key, value in row.items() if value is not None}


@dataclass
class AkShareSectorProvider:
    """Industry classification from AkShare with strict symbol-level joins."""

    allow_network: bool = False
    source: str = "akshare_sector"
    rate_limit_seconds: float = 0.2
    retry_count: int = 1
    retry_sleep_seconds: float = 0.5
    local_mapping: pd.DataFrame | None = None

    def industry_classification(
        self,
        request: ProviderRequest | None = None,
        as_of_date: str | None = None,
    ) -> ProviderResult:
        as_of = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.today().normalize()
        if self.local_mapping is not None and not self.local_mapping.empty:
            frame = self._normalize_local(self.local_mapping, as_of, request)
            return self._wrap(frame, source=f"{self.source}:local", quality=0.85)
        ak = _ensure_akshare(self.allow_network)
        list_endpoint = getattr(ak, "stock_board_industry_name_em", None) or getattr(
            ak, "stock_board_industry_summary_ths", None
        )
        cons_endpoint = getattr(ak, "stock_board_industry_cons_em", None) or getattr(
            ak, "stock_board_industry_cons_ths", None
        )
        if list_endpoint is None or cons_endpoint is None:
            raise ProviderUnavailable(
                "AkShare symbol-level sector mapping requires board-list and board-constituent endpoints. "
                "Supply a local sector mapping CSV instead."
            )
        try:
            boards = list_endpoint()
        except Exception as exc:  # pragma: no cover - network path
            raise ProviderUnavailable(f"AkShare industry list endpoint failed: {exc}") from exc
        if boards is None or boards.empty:
            return ProviderResult(pd.DataFrame(), source=self.source, quality_score=0.0, warnings=("akshare_empty_sector_board_list",))
        column = "板块名称" if "板块名称" in boards.columns else boards.columns[0]
        board_names = boards[column].astype(str).dropna().unique().tolist()
        rows: list[dict[str, object]] = []
        symbols_filter = {str(s) for s in request.symbols} if request is not None and request.symbols else None
        for board in board_names:
            try:
                members = cons_endpoint(symbol=board)
            except Exception:  # pragma: no cover - network path
                continue
            if members is None or members.empty:
                continue
            code_col = next((c for c in ("代码", "code", "symbol_raw", "股票代码") if c in members.columns), None)
            if code_col is None:
                continue
            for raw_code in members[code_col].astype(str):
                symbol = _suffix_from_code(raw_code)
                if symbols_filter is not None and symbol not in symbols_filter:
                    continue
                rows.append({"symbol": symbol, "industry": board, "available_at": as_of})
            if self.rate_limit_seconds > 0:
                time.sleep(self.rate_limit_seconds)
        if not rows:
            return ProviderResult(pd.DataFrame(), source=self.source, quality_score=0.0, warnings=("akshare_empty_sector_membership",))
        frame = pd.DataFrame(rows).drop_duplicates(subset=("symbol", "industry"))
        return self._wrap(frame, source=self.source, quality=0.75)

    @staticmethod
    def _normalize_local(
        frame: pd.DataFrame,
        as_of: pd.Timestamp,
        request: ProviderRequest | None,
    ) -> pd.DataFrame:
        data = frame.copy()
        if "symbol" not in data.columns or "industry" not in data.columns:
            raise ProviderUnavailable("local sector mapping must contain 'symbol' and 'industry' columns")
        if "available_at" not in data.columns:
            data["available_at"] = as_of
        if request is not None and request.symbols:
            symbols = {str(s) for s in request.symbols}
            data = data[data["symbol"].astype(str).isin(symbols)]
        return data.reset_index(drop=True)

    def _wrap(self, frame: pd.DataFrame, *, source: str, quality: float) -> ProviderResult:
        report = akshare_sector_schema_report(frame)
        warnings = tuple(f"akshare_schema_missing:{c}" for c in report["missing_columns"])
        return ProviderResult(
            frame.reset_index(drop=True),
            source=source,
            point_in_time=True,
            quality_score=quality if report["status"] == "passed" else 0.0,
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


_ = to_akshare_symbol
