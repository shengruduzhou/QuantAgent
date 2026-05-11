from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantagent.data.providers.base import FullDataProvider, ProviderRequest, ProviderResult
from quantagent.data.providers.mock_provider import MockProvider


@dataclass
class LocalCsvProvider(FullDataProvider):
    root: str | Path = "data/local"
    fallback: MockProvider | None = None

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        return self._read_or_fallback("daily_ohlcv.csv", request, "daily_ohlcv")

    def adjusted_prices(self, request: ProviderRequest) -> ProviderResult:
        return self._read_or_fallback("adjusted_prices.csv", request, "adjusted_prices")

    def tradability(self, request: ProviderRequest) -> ProviderResult:
        return self._read_or_fallback("tradability.csv", request, "tradability")

    def news(self, request: ProviderRequest) -> ProviderResult:
        return self._read_or_fallback("news.csv", request, "news")

    def fundamentals(self, request: ProviderRequest) -> ProviderResult:
        return self._read_or_fallback("fundamentals.csv", request, "fundamentals")

    def macro(self, request: ProviderRequest) -> ProviderResult:
        return self._read_or_fallback("macro.csv", request, "macro")

    def fund_flow(self, request: ProviderRequest) -> ProviderResult:
        return self._read_or_fallback("fund_flow.csv", request, "fund_flow")

    def trading_days(self, request: ProviderRequest) -> ProviderResult:
        return self._read_or_fallback("trading_days.csv", request, "trading_days")

    def commodity(self, request: ProviderRequest) -> ProviderResult:
        return self._read_or_fallback("commodity.csv", request, "commodity")

    def index_daily(self, request: ProviderRequest) -> ProviderResult:
        return self._read_or_fallback("index_daily.csv", request, "index_daily")

    def _read_or_fallback(self, file_name: str, request: ProviderRequest, method: str) -> ProviderResult:
        path = Path(self.root) / file_name
        if path.exists():
            frame = pd.read_csv(path)
            frame = _filter_dates(frame, request)
            if request.symbols and "symbol" in frame.columns:
                frame = frame[frame["symbol"].astype(str).isin(request.symbols)]
            return ProviderResult(frame.reset_index(drop=True), source=f"local_csv:{path}", metadata={"path": str(path)})
        fallback = self.fallback or MockProvider()
        result = getattr(fallback, method)(request)
        return ProviderResult(
            result.frame,
            source=result.source,
            point_in_time=result.point_in_time,
            quality_score=min(result.quality_score, 0.85),
            warnings=result.warnings + (f"missing_local_csv:{path}",),
            metadata=result.metadata | {"fallback": "mock"},
        )


def _filter_dates(frame: pd.DataFrame, request: ProviderRequest) -> pd.DataFrame:
    if "trade_date" not in frame.columns:
        return frame
    data = frame.copy()
    dates = pd.to_datetime(data["trade_date"])
    mask = (dates >= pd.Timestamp(request.start_date)) & (dates <= pd.Timestamp(request.end_date))
    return data.loc[mask]

