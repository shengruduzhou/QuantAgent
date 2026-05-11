from __future__ import annotations

from dataclasses import dataclass

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable
from quantagent.data.providers.mock_provider import MockProvider


@dataclass
class AkShareProvider(MockProvider):
    """Optional AkShare adapter skeleton with mock fallback semantics."""

    allow_network: bool = False
    source: str = "akshare_provider"

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        if not self.allow_network:
            result = super().daily_ohlcv(request)
            return _fallback_result(result, "akshare_network_disabled")
        try:
            import akshare as ak  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            if request.use_cache:
                return _fallback_result(super().daily_ohlcv(request), f"akshare_unavailable:{type(exc).__name__}")
            raise ProviderUnavailable("AkShare is not available") from exc
        raise ProviderUnavailable("AkShare live download is intentionally configured by integration tests or CLI runtime only")


def _fallback_result(result: ProviderResult, warning: str) -> ProviderResult:
    return ProviderResult(
        frame=result.frame,
        source=result.source,
        point_in_time=result.point_in_time,
        quality_score=min(result.quality_score, 0.80),
        warnings=result.warnings + (warning,),
        metadata=result.metadata | {"fallback": True},
    )

