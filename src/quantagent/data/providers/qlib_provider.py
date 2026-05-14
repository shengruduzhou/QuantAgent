from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable


@dataclass
class QlibProvider:
    """Optional qlib adapter for local PIT market data."""

    provider_uri: str | None = None
    region: str = "cn"

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        try:
            import qlib  # type: ignore
            from qlib.data import D  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable("pyqlib is not available") from exc
        if not self.provider_uri:
            raise ProviderUnavailable("qlib provider_uri is required for V7 qlib data")
        qlib.init(provider_uri=self.provider_uri, region=self.region)
        instruments = list(request.symbols) if request.symbols else request.universe
        if not instruments:
            raise ProviderUnavailable("qlib request requires symbols or universe")
        fields = ["$open", "$high", "$low", "$close", "$volume", "$amount"]
        frame = D.features(instruments, fields, start_time=request.start_date, end_time=request.end_date, freq="day")
        if frame.empty:
            return ProviderResult(pd.DataFrame(), source="qlib_provider", quality_score=0.0, warnings=("qlib_empty_daily_ohlcv",))
        data = frame.reset_index().rename(
            columns={
                "datetime": "trade_date",
                "instrument": "symbol",
                "$open": "open",
                "$high": "high",
                "$low": "low",
                "$close": "close",
                "$volume": "volume",
                "$amount": "amount",
            }
        )
        data["available_at"] = data["trade_date"]
        data["source"] = "qlib"
        data["source_type"] = "market_data"
        data["source_reliability"] = 0.90
        data["point_in_time_valid"] = True
        return ProviderResult(data, source="qlib_provider", point_in_time=True, quality_score=0.90)
