from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.data.providers.base import FullDataProvider, ProviderRequest, ProviderResult


DEFAULT_SYMBOLS = ("600000.SH", "600001.SH", "000001.SZ", "300750.SZ", "688981.SH", "000858.SZ")


@dataclass
class MockProvider(FullDataProvider):
    seed: int = 42
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    source: str = "mock_provider"

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        symbols = request.symbols or self.symbols
        dates = _business_days(request.start_date, request.end_date)
        rng = np.random.default_rng(self.seed)
        rows: list[dict[str, object]] = []
        sectors = ("bank", "tech", "consumer", "ev", "semiconductor", "liquor")
        for j, symbol in enumerate(symbols):
            drift = 0.02 + 0.004 * (j % 3)
            close = 20 + j * 3 + np.cumsum(rng.normal(drift, 0.25 + 0.02 * j, len(dates)))
            volume = np.maximum(10_000, 1_000_000 + rng.normal(0, 30_000, len(dates)).cumsum())
            for i, date in enumerate(dates):
                suspended = bool(i == 3 and j == 0)
                limit_up = bool(i == 5 and j == 1)
                limit_down = bool(i == 6 and j == 2)
                rows.append(
                    {
                        "trade_date": date,
                        "symbol": symbol,
                        "open": float(close[i] * 0.996),
                        "high": float(close[i] * (1.10 if limit_up else 1.015)),
                        "low": float(close[i] * (0.90 if limit_down else 0.985)),
                        "close": float(max(close[i], 1.0)),
                        "volume": 0.0 if suspended else float(volume[i]),
                        "amount": 0.0 if suspended else float(volume[i] * close[i]),
                        "is_suspended": suspended,
                        "is_limit_up": limit_up,
                        "is_limit_down": limit_down,
                        "is_st": bool(symbol.endswith(".BJ")),
                        "listed_days": int(500 + i),
                        "sector": sectors[j % len(sectors)],
                    }
                )
        return ProviderResult(pd.DataFrame(rows), source=self.source, metadata={"mock": True})

    def adjusted_prices(self, request: ProviderRequest) -> ProviderResult:
        frame = self.daily_ohlcv(request).frame
        frame["adjust_factor"] = 1.0
        frame["adj_close"] = frame["close"]
        return ProviderResult(frame, source=self.source, metadata={"mock": True})

    def tradability(self, request: ProviderRequest) -> ProviderResult:
        frame = self.daily_ohlcv(request).frame
        cols = ["trade_date", "symbol", "is_suspended", "is_limit_up", "is_limit_down", "is_st", "listed_days"]
        return ProviderResult(frame[cols], source=self.source, metadata={"mock": True})

    def news(self, request: ProviderRequest) -> ProviderResult:
        symbols = request.symbols or self.symbols
        dates = _business_days(request.start_date, request.end_date)
        rows = []
        for i, symbol in enumerate(symbols):
            if not len(dates):
                continue
            date = dates[min(i, len(dates) - 1)]
            rows.append(
                {
                    "timestamp": pd.Timestamp(date) + pd.Timedelta(hours=10),
                    "symbol": symbol,
                    "sector": "tech" if "688" in symbol or "300" in symbol else "bank",
                    "title": "policy support and earnings growth" if i % 2 == 0 else "margin pressure and risk warning",
                    "summary": "synthetic mock news for V6 agent evidence",
                    "event_type": "news",
                    "polarity": 0.6 if i % 2 == 0 else -0.4,
                }
            )
        return ProviderResult(pd.DataFrame(rows), source=self.source, metadata={"mock": True})

    def fundamentals(self, request: ProviderRequest) -> ProviderResult:
        symbols = request.symbols or self.symbols
        dates = _business_days(request.start_date, request.end_date)
        base_date = pd.Timestamp(dates[0] if len(dates) else request.start_date)
        rows = []
        for i, symbol in enumerate(symbols):
            rows.append(
                {
                    "symbol": symbol,
                    "announcement_time": base_date + pd.Timedelta(days=2, hours=16),
                    "report_period": "2025Q4",
                    "roe": 0.08 + 0.01 * i,
                    "debt_to_asset": 0.35 + 0.03 * i,
                    "revenue_growth": 0.04 + 0.02 * (i % 3),
                    "profit_growth": 0.03 + 0.01 * (i % 4),
                    "cashflow_quality": 0.6 + 0.05 * (i % 3),
                }
            )
        return ProviderResult(pd.DataFrame(rows), source=self.source, metadata={"mock": True})

    def macro(self, request: ProviderRequest) -> ProviderResult:
        dates = _business_days(request.start_date, request.end_date)
        frame = pd.DataFrame(
            {
                "trade_date": dates,
                "pmi": 50.0,
                "cpi": 0.02,
                "ppi": 0.01,
                "m2": 0.08,
                "social_financing": 0.09,
                "fx_usdcny": 7.1,
                "rate_10y": 0.025,
            }
        )
        return ProviderResult(frame, source=self.source, metadata={"mock": True})

    def fund_flow(self, request: ProviderRequest) -> ProviderResult:
        symbols = request.symbols or self.symbols
        dates = _business_days(request.start_date, request.end_date)
        rng = np.random.default_rng(self.seed + 1)
        rows = [
            {
                "trade_date": date,
                "symbol": symbol,
                "northbound_flow": float(rng.normal(0, 1_000_000)),
                "main_money_flow": float(rng.normal(0, 2_000_000)),
                "margin_financing": float(abs(rng.normal(20_000_000, 1_000_000))),
                "etf_flow": float(rng.normal(0, 500_000)),
            }
            for date in dates
            for symbol in symbols
        ]
        return ProviderResult(pd.DataFrame(rows), source=self.source, metadata={"mock": True})

    def trading_days(self, request: ProviderRequest) -> ProviderResult:
        frame = pd.DataFrame({"trade_date": _business_days(request.start_date, request.end_date), "is_trading_day": True})
        return ProviderResult(frame, source=self.source, metadata={"mock": True})

    def commodity(self, request: ProviderRequest) -> ProviderResult:
        dates = _business_days(request.start_date, request.end_date)
        commodities = ("crude_oil", "copper", "gold", "iron_ore", "thermal_coal")
        rows = []
        for j, name in enumerate(commodities):
            price = 100 + j * 20 + np.sin(np.arange(len(dates)) / 4.0) * (j + 1)
            for date, value in zip(dates, price):
                rows.append({"trade_date": date, "commodity": name, "close": float(value), "return": 0.001 * (j + 1)})
        return ProviderResult(pd.DataFrame(rows), source=self.source, metadata={"mock": True})

    def index_daily(self, request: ProviderRequest) -> ProviderResult:
        dates = _business_days(request.start_date, request.end_date)
        indices = request.symbols or ("000300.SH", "000905.SH", "399006.SZ", "000688.SH", "000001.SH", "399001.SZ")
        rows = []
        for j, symbol in enumerate(indices):
            close = 3000 + j * 100 + np.cumsum(np.full(len(dates), 1.0 + j * 0.1))
            for date, value in zip(dates, close):
                rows.append({"trade_date": date, "symbol": symbol, "open": value * 0.999, "high": value * 1.003, "low": value * 0.997, "close": value, "volume": 1_000_000})
        return ProviderResult(pd.DataFrame(rows), source=self.source, metadata={"mock": True})


def _business_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="B")

