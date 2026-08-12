"""AKShare valuation / universe / sector adapters.

Network access requires ``allow_network=True``. Current snapshots are never
backdated. Historical valuation uses source-supplied dates from AKShare's Baidu
valuation endpoint and resolves conservative next-session availability through
an explicit calendar.

The sector adapter resolves symbol-level membership through board constituent
endpoints or a local mapping; current network membership cannot be relabelled as
an arbitrary historical classification date.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import time

import pandas as pd

from quantagent.data.providers.akshare_financial_provider import to_akshare_symbol
from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable
from quantagent.data.trading_calendar import TradingCalendar


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

_HISTORICAL_VALUATION_INDICATORS: dict[str, str] = {
    "pe_ttm": "市盈率(TTM)",
    "pb": "市净率",
    "market_cap": "总市值",
}
_BAIDU_MARKET_CAP_SOURCE_UNIT = "1e8_CNY"
_CANONICAL_MARKET_CAP_UNIT = "CNY"


def _row_hash(row: dict[str, object]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def _ensure_akshare(allow_network: bool):
    if not allow_network:
        raise ProviderUnavailable(
            "AkShare network is disabled; set allow_network=true explicitly"
        )
    try:
        import akshare as ak  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ProviderUnavailable(
            "akshare package is not available; install quantagent[data]"
        ) from exc
    return ak


def _suffix_from_code(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("4", "8", "92")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def _shanghai_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="Asia/Shanghai")


def _shanghai_today() -> pd.Timestamp:
    return _shanghai_now().tz_localize(None).normalize()


def _history_period_for_start(start_date: str) -> str:
    start = pd.Timestamp(start_date).normalize()
    age_days = max(0, int((_shanghai_today() - start).days))
    if age_days <= 370:
        return "近一年"
    if age_days <= 3 * 370:
        return "近三年"
    if age_days <= 5 * 370:
        return "近五年"
    if age_days <= 10 * 370:
        return "近十年"
    return "全部"


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
            return ProviderResult(
                pd.DataFrame(),
                source=self.source,
                quality_score=0.0,
                warnings=("akshare_empty_universe",),
            )
        frame = raw.rename(
            columns={
                "code": "symbol_raw",
                "name": "name",
                "代码": "symbol_raw",
                "名称": "name",
            }
        ).copy()
        frame["symbol"] = frame["symbol_raw"].apply(_suffix_from_code)
        frame["exchange"] = frame["symbol"].str.split(".").str[-1]
        if "list_date" not in frame.columns:
            frame["list_date"] = pd.NaT
        frame = frame[
            [c for c in ("symbol", "name", "exchange", "list_date") if c in frame.columns]
        ]
        report = akshare_universe_schema_report(frame)
        warnings = tuple(f"akshare_schema_missing:{c}" for c in report["missing_columns"])
        return ProviderResult(
            frame.reset_index(drop=True),
            source=self.source,
            point_in_time=False,
            quality_score=0.85 if report["status"] == "passed" else 0.0,
            warnings=warnings + ("akshare_current_universe_snapshot_not_historical_pit",),
            metadata={
                "schema_report": report,
                "snapshot_only": True,
                "production_integrity_certified": False,
            },
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
    """Current valuation snapshots plus genuine dated valuation history.

    ``snapshot`` is current-only. Supplying an old date is rejected instead of
    relabelling today's spot PE/PB/market-cap as historical data.

    ``historical`` uses ``stock_zh_valuation_baidu`` source dates. Baidu displays
    A-share total market capitalisation in 亿; QuantAgent converts that series by
    ``1e8`` so canonical ``market_cap`` is CNY. Historical bars are conservatively
    available from the next explicit trading session.
    """

    allow_network: bool = False
    source: str = "akshare_valuation"
    rate_limit_seconds: float = 0.2
    retry_count: int = 2
    retry_sleep_seconds: float = 0.5
    trading_calendar: TradingCalendar | None = None

    def snapshot(
        self, as_of_date: str, request: ProviderRequest | None = None
    ) -> ProviderResult:
        requested_date = pd.Timestamp(as_of_date).normalize()
        today = _shanghai_today()
        if requested_date != today:
            raise ProviderUnavailable(
                "AKShare spot valuation is a current snapshot and cannot be backdated; "
                "use AkShareValuationProvider.historical() for historical as-of dates"
            )
        ak = _ensure_akshare(self.allow_network)
        warnings: list[str] = []
        observed_at = _shanghai_now()
        try:
            raw = self._call_with_retry(
                getattr(ak, "stock_zh_a_spot_em", None),
                label="stock_zh_a_spot_em",
            )
        except ProviderUnavailable as exc:
            if request is None or not request.symbols:
                raise
            warnings.append(f"akshare_valuation_spot_em_failed:{exc}")
            raw = self._fallback_symbol_snapshots(ak, request.symbols, warnings)
        if raw is None or raw.empty:
            return ProviderResult(
                pd.DataFrame(),
                source=self.source,
                quality_score=0.0,
                warnings=tuple(warnings or ["akshare_empty_valuation"]),
            )
        frame = self._normalize_current_snapshot(raw, observed_at)
        if request is not None and request.symbols:
            symbol_set = {str(s) for s in request.symbols}
            frame = frame[frame["symbol"].astype(str).isin(symbol_set)]
        report = akshare_valuation_schema_report(frame)
        warnings.extend(f"akshare_schema_missing:{c}" for c in report["missing_columns"])
        point_in_time = bool(
            not frame.empty
            and report["status"] == "passed"
            and frame["point_in_time_valid"].fillna(False).astype(bool).all()
        )
        return ProviderResult(
            frame.reset_index(drop=True),
            source=f"{self.source}:current_snapshot",
            point_in_time=point_in_time,
            quality_score=0.75 if point_in_time else 0.0,
            warnings=tuple(warnings),
            metadata={
                "schema_report": report,
                "as_of_date": as_of_date,
                "snapshot_observed_at": observed_at.isoformat(),
                "snapshot_only": True,
                "function_name": (
                    "stock_zh_a_spot_em|stock_individual_info_em|"
                    "stock_zh_valuation_comparison_em"
                ),
                "akshare_version": str(getattr(ak, "__version__", "unknown")),
                "market_cap_unit": _CANONICAL_MARKET_CAP_UNIT,
                "production_integrity_certified": False,
            },
        )

    def historical(self, request: ProviderRequest) -> ProviderResult:
        """Fetch source-dated PE(TTM), PB and total market-cap history."""
        if not request.symbols:
            raise ProviderUnavailable(
                "AKShare historical valuation requires explicit symbols; a current universe "
                "must not be projected backward"
            )
        ak = _ensure_akshare(self.allow_network)
        api = getattr(ak, "stock_zh_valuation_baidu", None)
        if api is None:
            raise ProviderUnavailable(
                "akshare stock_zh_valuation_baidu is unavailable in this version"
            )
        period = _history_period_for_start(request.start_date)
        frames: list[pd.DataFrame] = []
        warnings: list[str] = []
        failed_symbols: list[str] = []
        indicator_failures: dict[str, list[str]] = {}
        for symbol in request.symbols:
            parts: list[pd.DataFrame] = []
            failures: list[str] = []
            for canonical, indicator in _HISTORICAL_VALUATION_INDICATORS.items():
                try:
                    raw = self._call_with_retry(
                        lambda api=api, symbol=symbol, indicator=indicator: api(
                            symbol=_plain_code(symbol),
                            indicator=indicator,
                            period=period,
                        ),
                        label=f"stock_zh_valuation_baidu:{indicator}",
                    )
                except ProviderUnavailable as exc:
                    failures.append(f"{canonical}:{exc}")
                    continue
                part = _normalise_baidu_indicator(raw, canonical)
                if part.empty:
                    failures.append(f"{canonical}:empty_or_invalid")
                    continue
                parts.append(part)
            if failures or len(parts) != len(_HISTORICAL_VALUATION_INDICATORS):
                failed_symbols.append(str(symbol))
                indicator_failures[str(symbol)] = failures
                warnings.append(
                    f"akshare_historical_valuation_incomplete:{symbol}:" + ";".join(failures)
                )
                continue
            merged = parts[0]
            for part in parts[1:]:
                merged = merged.merge(part, on="trade_date", how="inner", validate="one_to_one")
            start = pd.Timestamp(request.start_date).normalize()
            end = pd.Timestamp(request.end_date).normalize()
            merged["trade_date"] = pd.to_datetime(merged["trade_date"], errors="coerce").dt.normalize()
            merged = merged.loc[
                merged["trade_date"].notna()
                & (merged["trade_date"] >= start)
                & (merged["trade_date"] <= end)
            ].copy()
            if merged.empty:
                failed_symbols.append(str(symbol))
                warnings.append(f"akshare_historical_valuation_no_rows_in_request:{symbol}")
                continue
            merged["symbol"] = str(symbol)
            merged["market_cap_raw"] = merged["market_cap"]
            merged["market_cap"] = pd.to_numeric(
                merged["market_cap"], errors="coerce"
            ) * 100_000_000.0
            merged["market_cap_source_unit"] = _BAIDU_MARKET_CAP_SOURCE_UNIT
            merged["market_cap_unit"] = _CANONICAL_MARKET_CAP_UNIT
            merged["source"] = "akshare:stock_zh_valuation_baidu"
            merged["source_reliability"] = 0.72
            merged["source_date_supplied"] = True
            if self.trading_calendar is not None and not self.trading_calendar.empty:
                available = pd.Series(
                    [
                        self.trading_calendar.next_trading_day(value, lag_days=1)
                        for value in merged["trade_date"]
                    ],
                    index=merged.index,
                    dtype="datetime64[ns]",
                )
            else:
                available = pd.Series(
                    pd.NaT, index=merged.index, dtype="datetime64[ns]"
                )
            merged["available_at"] = available
            merged["point_in_time_valid"] = available.notna()
            merged["raw_hash"] = [
                _row_hash(row) for row in merged.to_dict("records")
            ]
            frames.append(merged)
            if self.rate_limit_seconds > 0:
                time.sleep(self.rate_limit_seconds)

        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        report = akshare_valuation_schema_report(frame)
        if failed_symbols:
            warnings.append(
                "akshare_historical_valuation_failed_symbols:"
                + ",".join(sorted(set(failed_symbols)))
            )
        if self.trading_calendar is None or self.trading_calendar.empty:
            warnings.append("akshare_valuation_calendar_missing:pit_not_certified")
        point_in_time = bool(
            not frame.empty
            and not failed_symbols
            and report["status"] == "passed"
            and "point_in_time_valid" in frame.columns
            and frame["point_in_time_valid"].fillna(False).astype(bool).all()
        )
        return ProviderResult(
            frame.reset_index(drop=True),
            source=f"{self.source}:historical_baidu",
            point_in_time=point_in_time,
            quality_score=0.72 if point_in_time else (0.35 if not frame.empty else 0.0),
            warnings=tuple(dict.fromkeys(warnings)),
            metadata={
                "schema_report": report,
                "function_name": "stock_zh_valuation_baidu",
                "akshare_version": str(getattr(ak, "__version__", "unknown")),
                "period": period,
                "indicators": dict(_HISTORICAL_VALUATION_INDICATORS),
                "failed_symbols": failed_symbols,
                "indicator_failures": indicator_failures,
                "requested_symbol_count": int(len(request.symbols)),
                "fetched_symbol_count": int(
                    frame["symbol"].nunique() if not frame.empty else 0
                ),
                "market_cap_source_unit": _BAIDU_MARKET_CAP_SOURCE_UNIT,
                "market_cap_unit": _CANONICAL_MARKET_CAP_UNIT,
                "calendar_bound": bool(
                    self.trading_calendar is not None and not self.trading_calendar.empty
                ),
                "calendar_production_certified": False,
                "production_integrity_certified": False,
            },
        )

    def _normalize_current_snapshot(
        self, frame: pd.DataFrame, observed_at: pd.Timestamp
    ) -> pd.DataFrame:
        keep_candidates = tuple(
            dict.fromkeys(tuple(_VALUATION_RENAME) + tuple(_VALUATION_RENAME.values()))
        )
        keep = [c for c in keep_candidates if c in frame.columns]
        data = frame[keep].rename(columns=_VALUATION_RENAME).copy()
        if "symbol_raw" in data.columns:
            data["symbol"] = data["symbol_raw"].astype(str).apply(_suffix_from_code)
            data = data.drop(columns=["symbol_raw"])
        data["trade_date"] = observed_at.tz_localize(None).normalize()
        data["available_at"] = observed_at.isoformat()
        for column in (
            "pe_ttm",
            "pb",
            "ps_ttm",
            "peg",
            "ev_ebitda",
            "market_cap",
            "free_float_market_cap",
            "dividend_yield",
            "turnover_rate",
        ):
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        data["market_cap_unit"] = _CANONICAL_MARKET_CAP_UNIT
        data["source"] = f"{self.source}:current_snapshot"
        data["source_reliability"] = 0.70
        data["point_in_time_valid"] = True
        data["raw_hash"] = [_row_hash(row) for row in data.to_dict("records")]
        return data

    # Backwards-compatible internal helper used by tests/forensics. It is
    # deliberately non-PIT when asked to label a current snapshot with any date
    # other than today, so no caller can obtain the old backdating behaviour.
    def _normalize(self, frame: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
        observed = _shanghai_now()
        data = self._normalize_current_snapshot(frame, observed)
        requested = pd.Timestamp(as_of_date).normalize()
        if requested != observed.tz_localize(None).normalize():
            data["trade_date"] = requested
            data["available_at"] = pd.NaT
            data["point_in_time_valid"] = False
            data["source"] = f"{self.source}:backdate_blocked"
        return data

    def _fallback_symbol_snapshots(
        self,
        ak: object,
        symbols: tuple[str, ...],
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
                info = self._call_with_retry(
                    lambda: info_api(symbol=_plain_code(symbol)),
                    label="stock_individual_info_em",
                )
                comparison = self._call_with_retry(
                    lambda: comparison_api(symbol=_em_symbol(symbol)),
                    label="stock_zh_valuation_comparison_em",
                )
            except ProviderUnavailable as exc:
                warnings.append(f"akshare_valuation_symbol_fallback_failed:{symbol}:{exc}")
                continue
            row = _valuation_row_from_symbol_frames(symbol, info, comparison)
            if row:
                frames.append(pd.DataFrame([row]))
            if self.rate_limit_seconds > 0:
                time.sleep(self.rate_limit_seconds)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _call_with_retry(self, callable_api, *, label: str):
        if callable_api is None:
            raise ProviderUnavailable(f"AkShare {label} is not available in this version")
        last_exc: Exception | None = None
        for attempt in range(max(1, self.retry_count + 1)):
            try:
                return callable_api()
            except Exception as exc:  # pragma: no cover - network path
                last_exc = exc
                if attempt < self.retry_count and self.retry_sleep_seconds > 0:
                    time.sleep(self.retry_sleep_seconds)
        raise ProviderUnavailable(f"AkShare {label} request failed: {last_exc}") from last_exc


def _normalise_baidu_indicator(
    frame: pd.DataFrame | None, canonical_name: str
) -> pd.DataFrame:
    if frame is None or frame.empty or not {"date", "value"}.issubset(frame.columns):
        return pd.DataFrame()
    data = frame[["date", "value"]].copy()
    data["trade_date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data[canonical_name] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["trade_date", canonical_name])
    data = data.drop_duplicates(subset=["trade_date"], keep="last")
    return data[["trade_date", canonical_name]].sort_values("trade_date").reset_index(drop=True)


def akshare_valuation_schema_report(frame: pd.DataFrame) -> dict[str, object]:
    missing = [c for c in AKSHARE_VALUATION_REQUIRED_COLUMNS if c not in frame.columns]
    pit_violations = 0
    if "available_at" in frame.columns:
        pit_violations += int(pd.to_datetime(frame["available_at"], errors="coerce").isna().sum())
    if "point_in_time_valid" in frame.columns:
        pit_violations += int((~frame["point_in_time_valid"].fillna(False).astype(bool)).sum())
    required_nulls = 0
    for column in ("pe_ttm", "pb", "market_cap"):
        if column in frame.columns:
            required_nulls += int(pd.to_numeric(frame[column], errors="coerce").isna().sum())
    return {
        "status": "passed"
        if not missing and pit_violations == 0 and required_nulls == 0
        else "failed",
        "row_count": int(0 if frame is None else len(frame)),
        "required_columns": list(AKSHARE_VALUATION_REQUIRED_COLUMNS),
        "missing_columns": missing,
        "pit_violation_count": pit_violations,
        "required_value_null_count": required_nulls,
        "canonical_market_cap_unit": _CANONICAL_MARKET_CAP_UNIT,
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
    if upper.endswith(".BJ") or upper.startswith("BJ") or code.startswith(("4", "8", "92")):
        return f"BJ{code}"
    return f"SZ{code}"


def _series_from_item_value(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty or not {"item", "value"}.issubset(frame.columns):
        return pd.Series(dtype=object)
    return (
        frame.dropna(subset=["item"])
        .drop_duplicates(subset=["item"], keep="last")
        .set_index("item")["value"]
    )


def _valuation_row_from_symbol_frames(
    symbol: str,
    info: pd.DataFrame,
    comparison: pd.DataFrame,
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
    """Industry classification from AKShare with strict temporal truth.

    Network board membership is a **current snapshot**. It may be used only for
    today's classification. Historical sector research requires a local mapping
    that already carries its own ``available_at`` evidence; current members are
    never relabelled with an arbitrary old date.
    """

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
        requested_as_of = (
            pd.Timestamp(as_of_date).normalize() if as_of_date else _shanghai_today()
        )
        if self.local_mapping is not None and not self.local_mapping.empty:
            frame = self._normalize_local(self.local_mapping, requested_as_of, request)
            return self._wrap(
                frame,
                source=f"{self.source}:local",
                quality=0.85,
                snapshot_only=False,
            )
        if requested_as_of != _shanghai_today():
            raise ProviderUnavailable(
                "AKShare network sector endpoints expose current membership only; "
                "historical as_of_date requires a dated local PIT mapping"
            )
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
                "Supply a local dated sector mapping CSV instead."
            )
        try:
            boards = list_endpoint()
        except Exception as exc:  # pragma: no cover - network path
            raise ProviderUnavailable(f"AkShare industry list endpoint failed: {exc}") from exc
        if boards is None or boards.empty:
            return ProviderResult(
                pd.DataFrame(),
                source=self.source,
                quality_score=0.0,
                warnings=("akshare_empty_sector_board_list",),
            )
        column = "板块名称" if "板块名称" in boards.columns else boards.columns[0]
        board_names = boards[column].astype(str).dropna().unique().tolist()
        observed_at = _shanghai_now()
        rows: list[dict[str, object]] = []
        symbols_filter = (
            {str(s) for s in request.symbols}
            if request is not None and request.symbols
            else None
        )
        for board in board_names:
            try:
                members = cons_endpoint(symbol=board)
            except Exception:  # pragma: no cover - network path
                continue
            if members is None or members.empty:
                continue
            code_col = next(
                (
                    c
                    for c in ("代码", "code", "symbol_raw", "股票代码")
                    if c in members.columns
                ),
                None,
            )
            if code_col is None:
                continue
            for raw_code in members[code_col].astype(str):
                symbol = _suffix_from_code(raw_code)
                if symbols_filter is not None and symbol not in symbols_filter:
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "industry": board,
                        "available_at": observed_at.isoformat(),
                    }
                )
            if self.rate_limit_seconds > 0:
                time.sleep(self.rate_limit_seconds)
        if not rows:
            return ProviderResult(
                pd.DataFrame(),
                source=self.source,
                quality_score=0.0,
                warnings=("akshare_empty_sector_membership",),
            )
        frame = pd.DataFrame(rows).drop_duplicates(subset=("symbol", "industry"))
        return self._wrap(
            frame,
            source=f"{self.source}:current_snapshot",
            quality=0.75,
            snapshot_only=True,
        )

    @staticmethod
    def _normalize_local(
        frame: pd.DataFrame,
        as_of: pd.Timestamp,
        request: ProviderRequest | None,
    ) -> pd.DataFrame:
        data = frame.copy()
        if "symbol" not in data.columns or "industry" not in data.columns:
            raise ProviderUnavailable(
                "local sector mapping must contain 'symbol' and 'industry' columns"
            )
        if "available_at" not in data.columns:
            if as_of != _shanghai_today():
                raise ProviderUnavailable(
                    "historical local sector mapping requires explicit available_at evidence"
                )
            data["available_at"] = _shanghai_now().isoformat()
        if request is not None and request.symbols:
            symbols = {str(s) for s in request.symbols}
            data = data[data["symbol"].astype(str).isin(symbols)]
        return data.reset_index(drop=True)

    def _wrap(
        self,
        frame: pd.DataFrame,
        *,
        source: str,
        quality: float,
        snapshot_only: bool,
    ) -> ProviderResult:
        report = akshare_sector_schema_report(frame)
        warnings = tuple(f"akshare_schema_missing:{c}" for c in report["missing_columns"])
        return ProviderResult(
            frame.reset_index(drop=True),
            source=source,
            point_in_time=bool(report["status"] == "passed"),
            quality_score=quality if report["status"] == "passed" else 0.0,
            warnings=warnings,
            metadata={
                "schema_report": report,
                "snapshot_only": snapshot_only,
                "production_integrity_certified": False,
            },
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
