from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ProviderRequest:
    start_date: str
    end_date: str
    symbols: tuple[str, ...] = ()
    universe: str | None = None
    fields: tuple[str, ...] = ()
    use_cache: bool = True


@dataclass(frozen=True)
class ProviderResult:
    frame: pd.DataFrame
    source: str
    point_in_time: bool = True
    quality_score: float = 1.0
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class ProviderUnavailable(RuntimeError):
    """Raised when an external provider cannot be used in this runtime."""


class MarketDataProvider(ABC):
    @abstractmethod
    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult: ...

    @abstractmethod
    def adjusted_prices(self, request: ProviderRequest) -> ProviderResult: ...

    @abstractmethod
    def tradability(self, request: ProviderRequest) -> ProviderResult: ...


class NewsDataProvider(ABC):
    @abstractmethod
    def news(self, request: ProviderRequest) -> ProviderResult: ...


class FundamentalsProvider(ABC):
    @abstractmethod
    def fundamentals(self, request: ProviderRequest) -> ProviderResult: ...


class MacroDataProvider(ABC):
    @abstractmethod
    def macro(self, request: ProviderRequest) -> ProviderResult: ...


class FundFlowProvider(ABC):
    @abstractmethod
    def fund_flow(self, request: ProviderRequest) -> ProviderResult: ...


class TradingCalendarProvider(ABC):
    @abstractmethod
    def trading_days(self, request: ProviderRequest) -> ProviderResult: ...


class CommodityDataProvider(ABC):
    @abstractmethod
    def commodity(self, request: ProviderRequest) -> ProviderResult: ...


class IndexDataProvider(ABC):
    @abstractmethod
    def index_daily(self, request: ProviderRequest) -> ProviderResult: ...


class FullDataProvider(
    MarketDataProvider,
    NewsDataProvider,
    FundamentalsProvider,
    MacroDataProvider,
    FundFlowProvider,
    TradingCalendarProvider,
    CommodityDataProvider,
    IndexDataProvider,
    ABC,
):
    """Convenience base for providers that implement the full V6 contract."""

