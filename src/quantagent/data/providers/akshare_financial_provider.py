"""AkShare financial-statement adapter used as a fallback for TuShare.

The provider mirrors :class:`TuShareFinancialProvider` so the upstream
joiner can route through whichever source is available without changing
its internal schema. AkShare is generally noisier than TuShare and the
quality scores reflect that.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable
from quantagent.data.providers.tushare_financial_provider import _available_at


_INCOME_RENAME = {
    "报告日": "report_period",
    "公告日期": "ann_date",
    "营业总收入": "revenue",
    "营业收入": "revenue_alt",
    "营业利润": "operating_profit",
    "归属于母公司股东的净利润": "net_income_attr_parent",
    "净利润": "net_income",
    "基本每股收益": "eps",
    "研发费用": "rd_expense",
    "营业成本": "cogs",
}

_BALANCE_RENAME = {
    "报告日": "report_period",
    "公告日期": "ann_date",
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

_CASHFLOW_RENAME = {
    "报告日": "report_period",
    "公告日期": "ann_date",
    "经营活动产生的现金流量净额": "operating_cash_flow",
    "投资活动产生的现金流量净额": "investing_cash_flow",
    "筹资活动产生的现金流量净额": "financing_cash_flow",
    "购建固定资产、无形资产和其他长期资产支付的现金": "capex",
}


@dataclass
class AkShareFinancialProvider:
    """AkShare adapter that emits PIT-friendly statement frames."""

    allow_network: bool = False
    available_lag_days: int = 1
    source: str = "akshare_financial_provider"

    def income(self, request: ProviderRequest) -> ProviderResult:
        return self._fetch_statement("stock_financial_report_sina", "利润表", _INCOME_RENAME, request)

    def balance_sheet(self, request: ProviderRequest) -> ProviderResult:
        return self._fetch_statement("stock_financial_report_sina", "资产负债表", _BALANCE_RENAME, request)

    def cashflow(self, request: ProviderRequest) -> ProviderResult:
        return self._fetch_statement("stock_financial_report_sina", "现金流量表", _CASHFLOW_RENAME, request)

    def all_statements(self, request: ProviderRequest) -> dict[str, ProviderResult]:
        return {
            "income": self.income(request),
            "balance_sheet": self.balance_sheet(request),
            "cashflow": self.cashflow(request),
        }

    def _fetch_statement(
        self,
        api: str,
        symbol_argument: str,
        rename: dict[str, str],
        request: ProviderRequest,
    ) -> ProviderResult:
        if not self.allow_network:
            raise ProviderUnavailable(
                "AkShare financial download is disabled; set data.allow_network=true explicitly"
            )
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable("akshare package is not available") from exc
        if not request.symbols:
            raise ProviderUnavailable(f"AkShare {api} requires explicit symbols")
        frames: list[pd.DataFrame] = []
        callable_api = getattr(ak, api, None)
        if callable_api is None:
            raise ProviderUnavailable(f"akshare {api} is not available in this version")
        for symbol in request.symbols:
            try:
                raw = callable_api(stock=_plain_code(symbol), symbol=symbol_argument)
            except TypeError:
                raw = callable_api(stock=_plain_code(symbol))
            if raw is None or raw.empty:
                continue
            frames.append(self._normalize(raw, rename, symbol))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return ProviderResult(
            frame,
            source=f"{self.source}:{api}:{symbol_argument}",
            point_in_time=True,
            quality_score=0.72 if not frame.empty else 0.0,
            warnings=() if not frame.empty else (f"akshare_empty_{symbol_argument}",),
            metadata={"api": api, "category": symbol_argument},
        )

    def _normalize(self, frame: pd.DataFrame, rename: dict[str, str], symbol: str) -> pd.DataFrame:
        keep = [column for column in rename if column in frame.columns]
        data = frame[keep].rename(columns=rename).copy()
        data["symbol"] = symbol
        for column in ("ann_date", "report_period"):
            if column in data.columns:
                data[column] = pd.to_datetime(data[column].astype(str), errors="coerce").dt.strftime("%Y-%m-%d")
        if "ann_date" not in data.columns and "report_period" in data.columns:
            data["ann_date"] = data["report_period"]
        if "ann_date" in data.columns:
            data["available_at"] = _available_at(data["ann_date"], self.available_lag_days)
        data["source"] = self.source
        data["source_reliability"] = 0.72
        data["point_in_time_valid"] = True
        return data


def _plain_code(symbol: str) -> str:
    return str(symbol).split(".")[0]
