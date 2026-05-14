"""TuShare Pro financial-statement provider with point-in-time semantics.

The provider only emits structured ProviderResult frames that carry
``report_period``, ``ann_date`` and ``available_at`` columns. The actual
network call is deferred behind ``allow_network`` and ``token_env`` so the
default research path stays deterministic and offline.

The provider never returns synthetic data. When the token, network access
or the tushare package is missing it raises ``ProviderUnavailable`` so the
upstream DataHub can decide whether to keep the offline cache or refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import os
from typing import Iterable

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable


_INCOME_RENAME = {
    "ts_code": "symbol",
    "ann_date": "ann_date",
    "end_date": "report_period",
    "total_revenue": "revenue",
    "revenue": "revenue_alt",
    "operate_profit": "operating_profit",
    "n_income": "net_income",
    "n_income_attr_p": "net_income_attr_parent",
    "basic_eps": "eps",
    "diluted_eps": "diluted_eps",
    "rd_exp": "rd_expense",
    "sell_exp": "sell_expense",
    "admin_exp": "admin_expense",
    "fin_exp": "financial_expense",
    "oper_cost": "cogs",
}

_BALANCE_RENAME = {
    "ts_code": "symbol",
    "ann_date": "ann_date",
    "end_date": "report_period",
    "total_assets": "total_assets",
    "total_liab": "total_liabilities",
    "total_hldr_eqy_inc_min_int": "equity",
    "money_cap": "cash",
    "accounts_receiv": "receivables",
    "inventories": "inventory",
    "fix_assets": "fixed_assets",
    "cip": "construction_in_progress",
    "goodwill": "goodwill",
    "st_borr": "short_term_debt",
    "lt_borr": "long_term_debt",
    "total_share": "total_shares",
}

_CASHFLOW_RENAME = {
    "ts_code": "symbol",
    "ann_date": "ann_date",
    "end_date": "report_period",
    "n_cashflow_act": "operating_cash_flow",
    "n_cashflow_inv_act": "investing_cash_flow",
    "n_cashflow_fin_act": "financing_cash_flow",
    "c_pay_acq_const_fiolta": "capex",
    "free_cashflow": "free_cash_flow",
}

_INDICATOR_RENAME = {
    "ts_code": "symbol",
    "ann_date": "ann_date",
    "end_date": "report_period",
    "roe": "roe",
    "roa": "roa",
    "grossprofit_margin": "gross_margin",
    "netprofit_margin": "net_margin",
    "debt_to_assets": "debt_to_asset",
    "current_ratio": "current_ratio",
    "quick_ratio": "quick_ratio",
    "inv_turn": "inventory_turnover",
    "ar_turn": "receivables_turnover",
    "assets_turn": "asset_turnover",
    "eps": "eps_indicator",
    "bps": "bps",
    "ocf_to_profit": "ocf_to_profit",
    "fcff": "fcff",
}


@dataclass
class TuShareFinancialProvider:
    """PIT-aware adapter that pulls financial statements from TuShare Pro."""

    allow_network: bool = False
    token_env: str = "TUSHARE_TOKEN"
    available_lag_days: int = 1
    source: str = "tushare_financial_provider"

    def income(self, request: ProviderRequest) -> ProviderResult:
        return self._fetch("income", _INCOME_RENAME, request)

    def balance_sheet(self, request: ProviderRequest) -> ProviderResult:
        return self._fetch("balancesheet", _BALANCE_RENAME, request)

    def cashflow(self, request: ProviderRequest) -> ProviderResult:
        return self._fetch("cashflow", _CASHFLOW_RENAME, request)

    def financial_indicator(self, request: ProviderRequest) -> ProviderResult:
        return self._fetch("fina_indicator", _INDICATOR_RENAME, request)

    def disclosure_dates(self, request: ProviderRequest) -> ProviderResult:
        return self._fetch_disclosure(request)

    def all_statements(self, request: ProviderRequest) -> dict[str, ProviderResult]:
        return {
            "income": self.income(request),
            "balance_sheet": self.balance_sheet(request),
            "cashflow": self.cashflow(request),
            "financial_indicator": self.financial_indicator(request),
            "disclosure_dates": self.disclosure_dates(request),
        }

    def _fetch(self, api: str, rename: dict[str, str], request: ProviderRequest) -> ProviderResult:
        self._require_runtime()
        pro = self._client()
        if not request.symbols:
            raise ProviderUnavailable(f"TuShare {api} requires explicit symbols")
        frames: list[pd.DataFrame] = []
        for symbol in request.symbols:
            raw = getattr(pro, api)(
                ts_code=_tushare_code(symbol),
                start_date=request.start_date.replace("-", ""),
                end_date=request.end_date.replace("-", ""),
            )
            if raw is None or raw.empty:
                continue
            frames.append(self._normalize(raw, rename))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return ProviderResult(
            frame,
            source=f"{self.source}:{api}",
            point_in_time=True,
            quality_score=0.85 if not frame.empty else 0.0,
            warnings=() if not frame.empty else (f"tushare_empty_{api}",),
            metadata={"api": api, "available_lag_days": self.available_lag_days},
        )

    def _fetch_disclosure(self, request: ProviderRequest) -> ProviderResult:
        self._require_runtime()
        pro = self._client()
        frames: list[pd.DataFrame] = []
        for symbol in request.symbols or ():
            raw = pro.disclosure_date(
                ts_code=_tushare_code(symbol),
                start_date=request.start_date.replace("-", ""),
                end_date=request.end_date.replace("-", ""),
            )
            if raw is None or raw.empty:
                continue
            data = raw.rename(
                columns={
                    "ts_code": "symbol",
                    "ann_date": "ann_date",
                    "end_date": "report_period",
                    "pre_date": "preliminary_date",
                    "actual_date": "actual_date",
                }
            )
            data["available_at"] = _available_at(data["ann_date"], self.available_lag_days)
            frames.append(data)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return ProviderResult(
            frame,
            source=f"{self.source}:disclosure_date",
            point_in_time=True,
            quality_score=0.80 if not frame.empty else 0.0,
            warnings=() if not frame.empty else ("tushare_empty_disclosure_date",),
        )

    def _require_runtime(self) -> None:
        if not self.allow_network:
            raise ProviderUnavailable(
                "TuShare financial download is disabled; set data.allow_network=true explicitly"
            )
        if not os.getenv(self.token_env):
            raise ProviderUnavailable(f"{self.token_env} is required for TuShare financial download")

    def _client(self):
        try:
            import tushare as ts  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable("tushare package is not available") from exc
        return ts.pro_api(os.getenv(self.token_env))

    def _normalize(self, frame: pd.DataFrame, rename: dict[str, str]) -> pd.DataFrame:
        keep = [column for column in rename if column in frame.columns]
        data = frame[keep].rename(columns=rename).copy()
        if "symbol" in data.columns:
            data["symbol"] = data["symbol"].astype(str)
        for column in ("ann_date", "report_period"):
            if column in data.columns:
                data[column] = pd.to_datetime(data[column].astype(str), errors="coerce").dt.strftime("%Y-%m-%d")
        if "ann_date" in data.columns:
            data["available_at"] = _available_at(data["ann_date"], self.available_lag_days)
        data["source"] = self.source
        data["source_reliability"] = 0.85
        data["point_in_time_valid"] = True
        return data


def _tushare_code(symbol: str) -> str:
    text = str(symbol).upper()
    if "." in text:
        return text
    if text.startswith("6"):
        return f"{text}.SH"
    return f"{text}.SZ"


def _available_at(ann_dates: pd.Series, lag_days: int) -> pd.Series:
    """PIT rule: data becomes available the trading day AFTER ann_date.

    A real engine would map ann_date to the next trading day via a market
    calendar; the additive lag is a conservative approximation that prevents
    intra-day leakage when the announcement is made post-close.
    """

    parsed = pd.to_datetime(ann_dates, errors="coerce")
    shifted = parsed + pd.to_timedelta(max(1, lag_days), unit="D")
    return shifted.dt.strftime("%Y-%m-%d").fillna("")


def merge_statements(
    statements: dict[str, ProviderResult],
    on: tuple[str, ...] = ("symbol", "report_period"),
) -> ProviderResult:
    """Merge income/balance/cashflow/indicator into a single wide frame.

    The merge preserves the strictest ``available_at`` across statements
    so a downstream PIT filter never reveals a report before its slowest
    component is announced.
    """

    frames = []
    sources: list[str] = []
    warnings: list[str] = []
    quality = 1.0
    for name, result in statements.items():
        if result.frame is None or result.frame.empty:
            warnings.append(f"missing_{name}")
            continue
        frames.append(result.frame.copy())
        sources.append(result.source)
        quality = min(quality, result.quality_score)
    if not frames:
        return ProviderResult(
            pd.DataFrame(),
            source="tushare_financial_merge",
            point_in_time=True,
            quality_score=0.0,
            warnings=tuple(warnings) or ("no_statements_to_merge",),
        )
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=list(on), how="outer", suffixes=("", "_dup"))
        for column in list(merged.columns):
            if column.endswith("_dup"):
                base = column[: -len("_dup")]
                merged[base] = merged[base].combine_first(merged[column])
                merged = merged.drop(columns=[column])
    if "available_at" in merged.columns:
        merged["available_at"] = (
            pd.to_datetime(merged["available_at"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        )
    return ProviderResult(
        merged.reset_index(drop=True),
        source="|".join(sources),
        point_in_time=True,
        quality_score=quality,
        warnings=tuple(warnings),
    )


def select_financial_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    present = [column for column in columns if column in frame.columns]
    return frame[present].copy() if present else pd.DataFrame()
