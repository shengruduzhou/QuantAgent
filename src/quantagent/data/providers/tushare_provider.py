from __future__ import annotations

from dataclasses import dataclass
import os

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable
from quantagent.data.providers.mock_provider import MockProvider


@dataclass
class TuShareProvider(MockProvider):
    """Optional TuShare adapter skeleton. Tokens are read only from env."""

    allow_network: bool = False
    token_env: str = "TUSHARE_TOKEN"
    source: str = "tushare_provider"

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        if not self.allow_network or not os.getenv(self.token_env):
            result = super().daily_ohlcv(request)
            return ProviderResult(
                result.frame,
                source=result.source,
                point_in_time=result.point_in_time,
                quality_score=min(result.quality_score, 0.80),
                warnings=result.warnings + ("tushare_token_or_network_unavailable",),
                metadata=result.metadata | {"fallback": True},
            )
        try:
            import tushare as ts  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ProviderUnavailable("TuShare is not available") from exc
        raise ProviderUnavailable("TuShare live download is intentionally isolated to integration runtime")

