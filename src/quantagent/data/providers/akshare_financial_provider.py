"""AkShare financial-statement adapter used as a fallback for TuShare.

The adapter is intentionally strict: network access is opt-in, each output
frame must carry PIT keys, and AkShare API drift is returned as explicit
warnings or ``ProviderUnavailable`` instead of synthetic rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import time
from typing import Callable

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable
from quantagent.data.providers.tushare_financial_provider import _available_at
from quantagent.data.trading_calendar import TradingCalendar


_COMMON_RENAME = {
    "报告日期": "report_period",
    "报告日": "report_period",
    "报表日期": "report_period",
    "公告日期": "ann_date",
    "公告日": "ann_date",
    "更新日期": "update_date",
    "Report Date": "report_period",
    "Ann Date": "ann_date",
    "Update Date": "update_date",
}

_INCOME_RENAME = _COMMON_RENAME | {
    "营业总收入": "revenue",
    "营业收入": "revenue_alt",
    "营业利润": "operating_profit",
    "归属于母公司股东的净利润": "net_income_attr_parent",
    "净利润": "net_income",
    "基本每股收益": "eps",
    "研发费用": "rd_expense",
    "营业成本": "cogs",
    "Revenue": "revenue",
}

_BALANCE_RENAME = _COMMON_RENAME | {
    "资产总计": "total_assets",
    "负债合计": "total_liabilities",
    "股东权益合计": "equity",
    "货币资金": "cash",
    "应收账款": "receivables",
    "存货": "inventory",
    "固定资产": "fixed_assets",
    "在建工程": "construction_in_progress",
    "商誉": "goodwill",
}

_CASHFLOW_RENAME = _COMMON_RENAME | {
    "经营活动产生的现金流量净额": "operating_cash_flow",
    "投资活动产生的现金流量净额": "investing_cash_flow",
    "筹资活动产生的现金流量净额": "financing_cash_flow",
    "购建固定资产、无形资产和其他长期资产支付的现金": "capex",
}

_INDICATOR_RENAME = _COMMON_RENAME | {
    "净资产收益率": "roe",
    "加权净资产收益率": "roe_weighted",
    "总资产报酬率": "roa",
    "销售毛利率": "gross_margin",
    "销售净利率": "net_margin",
    "资产负债率": "debt_to_asset",
    "流动比率": "current_ratio",
    "速动比率": "quick_ratio",
    "存货周转率": "inventory_turnover",
    "应收账款周转率": "receivables_turnover",
    "总资产周转率": "asset_turnover",
    "每股收益": "eps_indicator",
    "每股净资产": "bps",
    "经营现金流量对净利润比率": "ocf_to_profit",
    "roe": "roe",
    "roa": "roa",
    "grossprofit_margin": "gross_margin",
    "netprofit_margin": "net_margin",
    "debt_to_assets": "debt_to_asset",
    "current_ratio": "current_ratio",
    "quick_ratio": "quick_ratio",
}

_DIVIDEND_RENAME = _COMMON_RENAME | {
    "代码": "symbol_raw",
    "名称": "name",
    "派息": "cash_dividend",
    "送股": "stock_dividend",
    "转增": "transfer_share",
    "除权除息日": "ex_dividend_date",
    "股权登记日": "record_date",
}


AKSHARE_FINANCIAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "report_period",
    "ann_date",
    "available_at",
)

AKSHARE_FINANCIAL_CANONICAL_COLUMNS: tuple[str, ...] = tuple(
    sorted(
        set(_INCOME_RENAME.values())
        | set(_BALANCE_RENAME.values())
        | set(_CASHFLOW_RENAME.values())
        | set(_INDICATOR_RENAME.values())
        | set(_DIVIDEND_RENAME.values())
        | {
            "symbol",
            "available_at",
            "source",
            "source_reliability",
            "raw_hash",
            "point_in_time_valid",
        }
    )
)


@dataclass
class AkShareFinancialProvider:
    """AkShare adapter that emits PIT-friendly statement frames."""

    allow_network: bool = False
    available_lag_days: int = 1
    source: str = "akshare_financial_provider"
    retry_count: int = 2
    retry_sleep_seconds: float = 0.5
    rate_limit_seconds: float = 0.2
    trading_calendar: TradingCalendar | None = None

    def income(self, request: ProviderRequest) -> ProviderResult:
        return self._fetch_financial_report_sina("income", "利润表", _INCOME_RENAME, request)

    def balance_sheet(self, request: ProviderRequest) -> ProviderResult:
        return self._fetch_financial_report_sina("balance_sheet", "资产负债表", _BALANCE_RENAME, request)

    def cashflow(self, request: ProviderRequest) -> ProviderResult:
        return self._fetch_financial_report_sina("cashflow", "现金流量表", _CASHFLOW_RENAME, request)

    def financial_indicator(self, request: ProviderRequest) -> ProviderResult:
        return self._fetch_symbol_api(
            statement_name="financial_indicator",
            api_name="stock_financial_analysis_indicator",
            rename=_INDICATOR_RENAME,
            request=request,
            call_builder=lambda api, symbol: api(symbol=_plain_code(symbol)),
        )

    def dividend(self, request: ProviderRequest) -> ProviderResult:
        return self._fetch_symbol_api(
            statement_name="dividend",
            api_name="stock_history_dividend_detail",
            rename=_DIVIDEND_RENAME,
            request=request,
            call_builder=lambda api, symbol: api(symbol=_plain_code(symbol), indicator="分红"),
        )

    def all_statements(self, request: ProviderRequest) -> dict[str, ProviderResult]:
        return {
            "income": self.income(request),
            "balance_sheet": self.balance_sheet(request),
            "cashflow": self.cashflow(request),
            "financial_indicator": self.financial_indicator(request),
            "dividend": self.dividend(request),
        }

    def health_check(self, request: ProviderRequest | None = None) -> dict[str, object]:
        if not self.allow_network:
            return {"status": "disabled", "reason": "allow_network_false"}
        try:
            import akshare as ak  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            return {"status": "unavailable", "reason": f"akshare_unavailable:{type(exc).__name__}"}
        if request is None:
            return {"status": "passed", "source": self.source}
        try:
            result = self.income(request)
        except ProviderUnavailable as exc:
            return {"status": "unavailable", "reason": str(exc)}
        return {
            "status": "passed" if result.quality_score > 0 else "failed",
            "source": result.source,
            "warnings": list(result.warnings),
            "schema_report": result.metadata.get("schema_report", {}),
        }

    def _fetch_financial_report_sina(
        self,
        statement_name: str,
        category: str,
        rename: dict[str, str],
        request: ProviderRequest,
    ) -> ProviderResult:
        return self._fetch_symbol_api(
            statement_name=statement_name,
            api_name="stock_financial_report_sina",
            rename=rename,
            request=request,
            call_builder=lambda api, symbol: self._call_financial_report_sina(api, symbol, category),
            params={"category": category},
        )

    def _fetch_symbol_api(
        self,
        *,
        statement_name: str,
        api_name: str,
        rename: dict[str, str],
        request: ProviderRequest,
        call_builder: Callable[[object, str], pd.DataFrame],
        params: dict[str, object] | None = None,
    ) -> ProviderResult:
        if not self.allow_network:
            raise ProviderUnavailable(
                "AkShare financial download is disabled; set data.allow_network=true explicitly"
            )
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable("akshare package is not available; install quantagent[data]") from exc
        if not request.symbols:
            raise ProviderUnavailable(f"AkShare {api_name} requires explicit symbols")
        callable_api = getattr(ak, api_name, None)
        if callable_api is None:
            raise ProviderUnavailable(f"akshare {api_name} is not available in this version")

        frames: list[pd.DataFrame] = []
        warnings: list[str] = []
        failed_symbols: list[str] = []
        for symbol in request.symbols:
            try:
                raw = self._call_with_retry(call_builder, callable_api, symbol)
            except ProviderUnavailable as exc:
                failed_symbols.append(str(symbol))
                warnings.append(f"akshare_{statement_name}_failed:{symbol}:{exc}")
                continue
            if raw is None or raw.empty:
                warnings.append(f"akshare_empty_{statement_name}:{symbol}")
            else:
                frames.append(self._normalize(raw, rename, symbol, statement_name))
            if self.rate_limit_seconds > 0:
                time.sleep(self.rate_limit_seconds)

        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        schema_report = akshare_financial_schema_report(frame)
        warnings.extend(f"akshare_schema_missing:{column}" for column in schema_report["missing_columns"])
        metadata = {
            "source": "akshare",
            "function_name": api_name,
            "params": {
                "start_date": request.start_date,
                "end_date": request.end_date,
                "symbols": list(request.symbols),
                **dict(params or {}),
            },
            "row_count": int(len(frame)),
            "schema_hash": _schema_hash(frame),
            "fetched_at": _utc_now_iso(),
            "warnings": warnings,
            "failed_symbols": failed_symbols,
            "schema_report": schema_report,
            "statement_name": statement_name,
        }
        return ProviderResult(
            frame,
            source=f"{self.source}:{api_name}:{statement_name}",
            point_in_time=True,
            quality_score=0.72 if not frame.empty and schema_report["status"] == "passed" else 0.0,
            warnings=tuple(warnings),
            metadata=metadata,
        )

    def _normalize(
        self,
        frame: pd.DataFrame,
        rename: dict[str, str],
        symbol: str,
        statement_name: str = "",
    ) -> pd.DataFrame:
        keep = [column for column in rename if column in frame.columns]
        data = frame[keep].rename(columns=rename).copy()
        data["symbol"] = symbol
        if "ann_date" not in data.columns and "update_date" in data.columns:
            data["ann_date"] = data["update_date"]
        for column in ("ann_date", "report_period", "ex_dividend_date", "record_date"):
            if column in data.columns:
                data[column] = pd.to_datetime(data[column].astype(str), errors="coerce").dt.strftime("%Y-%m-%d")
        if "report_period" not in data.columns:
            data["report_period"] = data.get("ann_date", pd.Series([""] * len(data), index=data.index))
        if "ann_date" not in data.columns and "report_period" in data.columns:
            data["ann_date"] = data["report_period"]
        if "ann_date" in data.columns:
            if self.trading_calendar is not None and not self.trading_calendar.empty:
                data["available_at"] = self.trading_calendar.resolve_available_at(
                    data["ann_date"], lag_days=self.available_lag_days
                ).dt.strftime("%Y-%m-%d")
            else:
                data["available_at"] = _available_at(data["ann_date"], self.available_lag_days)
        data["source"] = f"{self.source}:{statement_name}" if statement_name else self.source
        data["source_reliability"] = 0.72
        data["raw_hash"] = [_row_hash(row) for row in data.to_dict("records")]
        data["point_in_time_valid"] = True
        return data

    def _call_financial_report_sina(self, callable_api: object, symbol: str, category: str) -> pd.DataFrame:
        try:
            return callable_api(stock=to_akshare_symbol(symbol), symbol=category)  # type: ignore[misc]
        except TypeError:
            return callable_api(stock=to_akshare_symbol(symbol))  # type: ignore[misc]

    def _call_with_retry(
        self,
        call_builder: Callable[[object, str], pd.DataFrame],
        callable_api: object,
        symbol: str,
    ) -> pd.DataFrame:
        last_exc: Exception | None = None
        for attempt in range(max(1, self.retry_count + 1)):
            try:
                return call_builder(callable_api, symbol)
            except Exception as exc:  # pragma: no cover - network path
                last_exc = exc
                if attempt < self.retry_count and self.retry_sleep_seconds > 0:
                    time.sleep(self.retry_sleep_seconds)
        raise ProviderUnavailable(f"AkShare financial request failed for {symbol}: {last_exc}") from last_exc


def _plain_code(symbol: str) -> str:
    text = str(symbol).split(".")[0]
    lower = text.lower()
    for prefix in ("sh", "sz", "bj"):
        if lower.startswith(prefix):
            return text[len(prefix):]
    return text


def to_akshare_symbol(symbol: str) -> str:
    text = str(symbol).strip()
    upper = text.upper()
    code = upper.split(".")[0]
    if upper.endswith(".SH"):
        return f"sh{code}"
    if upper.endswith(".SZ"):
        return f"sz{code}"
    if upper.endswith(".BJ"):
        return f"bj{code}"
    if upper.startswith(("SH", "SZ", "BJ")):
        return text.lower()
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("0", "2", "3")):
        return f"sz{code}"
    if code.startswith(("4", "8")):
        return f"bj{code}"
    return code


def akshare_financial_schema_report(frame: pd.DataFrame) -> dict[str, object]:
    missing = [column for column in AKSHARE_FINANCIAL_REQUIRED_COLUMNS if column not in frame.columns]
    pit_violations = 0
    if "available_at" in frame.columns:
        parsed = pd.to_datetime(frame["available_at"], errors="coerce")
        pit_violations = int(parsed.isna().sum())
    return {
        "status": "passed" if not missing and pit_violations == 0 else "failed",
        "row_count": int(0 if frame is None else len(frame)),
        "required_columns": list(AKSHARE_FINANCIAL_REQUIRED_COLUMNS),
        "canonical_columns": list(AKSHARE_FINANCIAL_CANONICAL_COLUMNS),
        "missing_columns": missing,
        "pit_violation_count": pit_violations,
    }


def _row_hash(row: dict[str, object]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def _schema_hash(frame: pd.DataFrame) -> str:
    if frame is None:
        return ""
    payload = json.dumps(
        [{"name": str(column), "dtype": str(dtype)} for column, dtype in zip(frame.columns, frame.dtypes)],
        sort_keys=True,
        ensure_ascii=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
