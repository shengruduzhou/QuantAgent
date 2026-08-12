from __future__ import annotations

from dataclasses import dataclass

from quantagent.data.providers.akshare_calendar import load_akshare_research_calendar
from quantagent.data.providers.akshare_live_provider import AkShareLiveProvider
from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable


@dataclass
class AkShareProvider:
    """Compatibility facade for the governed live AKShare provider.

    This class used to inherit ``MockProvider`` and silently return synthetic
    OHLCV when AKShare networking was disabled.  That made a caller such as
    ``train-v8-pipeline --use-akshare`` capable of consuming mock bars while
    believing it had selected AKShare.  The compatibility name is retained for
    older call sites, but data identity is now fail-closed: this facade never
    fabricates market data.

    ``allow_network`` defaults to ``True`` because every in-repository runtime
    call site instantiates this compatibility facade only after the operator has
    explicitly selected AKShare (for example ``--use-akshare``).  Callers that
    need an offline object must set ``allow_network=False`` and handle
    ``ProviderUnavailable`` rather than receiving synthetic evidence.
    """

    allow_network: bool = True
    adjust: str = ""
    source: str = "akshare_provider_compat"

    def daily_ohlcv(self, request: ProviderRequest) -> ProviderResult:
        if not self.allow_network:
            raise ProviderUnavailable(
                "AKShare network is disabled; compatibility provider refuses mock fallback"
            )

        calendar_evidence = load_akshare_research_calendar(allow_network=True)
        if calendar_evidence.calendar.empty:
            raise ProviderUnavailable(
                "AKShare research calendar is unavailable; refusing to invent PIT sessions"
            )

        result = AkShareLiveProvider(
            allow_network=True,
            adjust=self.adjust,
            trading_calendar=calendar_evidence.calendar,
            calendar_source=str(calendar_evidence.metadata.get("source") or ""),
        ).daily_ohlcv(request)

        # Preserve the governed live-provider identity rather than relabelling
        # the result as the deprecated compatibility class.
        return ProviderResult(
            frame=result.frame,
            source=result.source,
            point_in_time=result.point_in_time,
            quality_score=result.quality_score,
            warnings=tuple(
                dict.fromkeys(
                    (*result.warnings, "akshare_compat_provider_delegated_to_live_provider")
                )
            ),
            metadata={
                **result.metadata,
                "compatibility_provider": self.source,
                "mock_fallback": False,
                "calendar": calendar_evidence.metadata,
            },
        )
