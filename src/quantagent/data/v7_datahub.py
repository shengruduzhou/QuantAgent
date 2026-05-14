from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable
from quantagent.data.providers.disclosure_provider import DisclosureWebProvider
from quantagent.data.providers.news_provider import NewsWebProvider
from quantagent.data.providers.policy_web_provider import PolicyWebProvider
from quantagent.data.providers.qlib_provider import QlibProvider
from quantagent.data.providers.tradingview_provider import TradingViewPublicProvider
from quantagent.data.providers.v7_research_provider import LocalV7ResearchProvider, V7ResearchDataBundle


V7ProviderMode = Literal["strict_local", "online", "mock"]


class V7DataQualityError(RuntimeError):
    """Raised when V7 research would silently use missing or synthetic data."""


@dataclass(frozen=True)
class V7DataHubConfig:
    root: str = "data/v7"
    provider_mode: V7ProviderMode = "strict_local"
    allow_synthetic_fallback: bool = False
    allow_network: bool = False
    policy_urls: tuple[str, ...] = ()
    news_urls: tuple[str, ...] = ()
    disclosure_urls: tuple[str, ...] = ()
    tradingview_urls: tuple[str, ...] = ()
    qlib_provider_uri: str | None = None
    qlib_region: str = "cn"
    required_tables: tuple[str, ...] = ("policies", "base_universe", "market_state")


@dataclass(frozen=True)
class V7DataHubResult:
    bundle: V7ResearchDataBundle
    provider_mode: V7ProviderMode
    allow_synthetic_fallback: bool
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class V7DataHub:
    """Unified PIT data entrypoint for V7 research."""

    def __init__(self, config: V7DataHubConfig | dict[str, Any] | None = None) -> None:
        self.config = _coerce_config(config)

    def load(self, request: ProviderRequest, as_of_date: str) -> V7DataHubResult:
        bundle = LocalV7ResearchProvider(self.config.root).load_bundle(request, as_of_date)
        if self.config.provider_mode == "online":
            bundle = self._merge_online_sources(bundle, request, as_of_date)
        warnings = _bundle_warnings(bundle)
        if self.config.provider_mode == "strict_local":
            self._validate_strict(bundle)
        if self.config.provider_mode == "online" and not self.config.allow_synthetic_fallback:
            self._validate_online(bundle)
        return V7DataHubResult(
            bundle=bundle,
            provider_mode=self.config.provider_mode,
            allow_synthetic_fallback=self.config.allow_synthetic_fallback or self.config.provider_mode == "mock",
            warnings=warnings,
            metadata={
                "root": self.config.root,
                "provider_mode": self.config.provider_mode,
                "allow_network": self.config.allow_network,
            },
        )

    def _validate_strict(self, bundle: V7ResearchDataBundle) -> None:
        missing = [name for name in self.config.required_tables if _empty_result(getattr(bundle, name))]
        if _empty_result(bundle.company_theme_map) and _empty_result(bundle.company_profiles):
            missing.append("company_theme_map_or_company_profiles")
        if missing:
            raise V7DataQualityError(
                "V7 strict_local data is incomplete; refusing synthetic fallback: " + ", ".join(sorted(set(missing)))
            )

    def _validate_online(self, bundle: V7ResearchDataBundle) -> None:
        missing = [name for name in self.config.required_tables if _empty_result(getattr(bundle, name))]
        if missing:
            raise V7DataQualityError(
                "V7 online data is incomplete and synthetic fallback is disabled: " + ", ".join(sorted(set(missing)))
            )

    def _merge_online_sources(
        self,
        bundle: V7ResearchDataBundle,
        request: ProviderRequest,
        as_of_date: str,
    ) -> V7ResearchDataBundle:
        policies = bundle.policies
        news = bundle.news
        announcements = bundle.announcements
        market_panel = bundle.market_panel
        if self.config.policy_urls:
            policies = _prefer_non_empty(
                policies,
                PolicyWebProvider(allow_network=self.config.allow_network).fetch_policy_documents(
                    request,
                    self.config.policy_urls,
                    as_of_date=as_of_date,
                ),
            )
        if self.config.news_urls:
            news = _prefer_non_empty(
                news,
                NewsWebProvider(allow_network=self.config.allow_network).fetch_news(
                    request,
                    self.config.news_urls,
                    as_of_date=as_of_date,
                ),
            )
        if self.config.disclosure_urls:
            announcements = _prefer_non_empty(
                announcements,
                DisclosureWebProvider(allow_network=self.config.allow_network).fetch_announcements(
                    request,
                    self.config.disclosure_urls,
                    as_of_date=as_of_date,
                ),
            )
        if self.config.tradingview_urls:
            tradingview = TradingViewPublicProvider(allow_network=self.config.allow_network).fetch_public_pages(
                request,
                self.config.tradingview_urls,
                as_of_date=as_of_date,
            )
            news = _merge_frames(news, tradingview)
        if self.config.qlib_provider_uri:
            try:
                qlib_market = QlibProvider(
                    provider_uri=self.config.qlib_provider_uri,
                    region=self.config.qlib_region,
                ).daily_ohlcv(request)
                market_panel = _prefer_non_empty(market_panel, qlib_market)
            except ProviderUnavailable as exc:
                market_panel = _append_warning(market_panel, f"qlib_unavailable:{exc}")
        return V7ResearchDataBundle(
            policies=policies,
            theme_metrics=bundle.theme_metrics,
            base_universe=bundle.base_universe,
            company_profiles=bundle.company_profiles,
            company_theme_map=bundle.company_theme_map,
            fundamentals=bundle.fundamentals,
            news=news,
            market_state=bundle.market_state,
            market_panel=market_panel,
            factors=bundle.factors,
            positions=bundle.positions,
            announcements=announcements,
            metadata=bundle.metadata | {"online_merge_as_of_date": as_of_date},
        )


def _coerce_config(config: V7DataHubConfig | dict[str, Any] | None) -> V7DataHubConfig:
    if config is None:
        return V7DataHubConfig()
    if isinstance(config, V7DataHubConfig):
        return config
    mode = str(config.get("provider_mode", config.get("default_provider_mode", "strict_local")))
    if mode == "mock_or_local":
        mode = "mock"
    if mode not in {"strict_local", "online", "mock"}:
        raise ValueError("data.provider_mode must be strict_local, online, or mock")
    return V7DataHubConfig(
        root=str(config.get("v7_root", "data/v7")),
        provider_mode=mode,  # type: ignore[arg-type]
        allow_synthetic_fallback=bool(config.get("allow_synthetic_fallback", mode == "mock")),
        allow_network=bool(config.get("allow_network", False)),
        policy_urls=tuple(str(item) for item in config.get("policy_urls", ())),
        news_urls=tuple(str(item) for item in config.get("news_urls", ())),
        disclosure_urls=tuple(str(item) for item in config.get("disclosure_urls", ())),
        tradingview_urls=tuple(str(item) for item in config.get("tradingview_urls", ())),
        qlib_provider_uri=str(config["qlib_provider_uri"]) if config.get("qlib_provider_uri") else None,
        qlib_region=str(config.get("qlib_region", "cn")),
        required_tables=tuple(str(item) for item in config.get("required_tables", ("policies", "base_universe", "market_state"))),
    )


def _empty_result(result: ProviderResult) -> bool:
    return result.frame is None or result.frame.empty


def _prefer_non_empty(primary: ProviderResult, secondary: ProviderResult) -> ProviderResult:
    if not _empty_result(primary):
        return primary
    if not _empty_result(secondary):
        return secondary
    return ProviderResult(
        pd.DataFrame(),
        source=f"{primary.source}|{secondary.source}",
        point_in_time=primary.point_in_time and secondary.point_in_time,
        quality_score=min(primary.quality_score, secondary.quality_score),
        warnings=primary.warnings + secondary.warnings,
        metadata=primary.metadata | {"secondary": secondary.metadata},
    )


def _merge_frames(primary: ProviderResult, secondary: ProviderResult) -> ProviderResult:
    if _empty_result(primary):
        return secondary
    if _empty_result(secondary):
        return primary
    frame = pd.concat([primary.frame, secondary.frame], ignore_index=True, sort=False)
    return ProviderResult(
        frame,
        source=f"{primary.source}|{secondary.source}",
        point_in_time=primary.point_in_time and secondary.point_in_time,
        quality_score=min(primary.quality_score, secondary.quality_score),
        warnings=primary.warnings + secondary.warnings,
        metadata=primary.metadata | {"secondary": secondary.metadata},
    )


def _append_warning(result: ProviderResult, warning: str) -> ProviderResult:
    return ProviderResult(
        result.frame,
        source=result.source,
        point_in_time=result.point_in_time,
        quality_score=result.quality_score,
        warnings=result.warnings + (warning,),
        metadata=result.metadata,
    )


def _bundle_warnings(bundle: V7ResearchDataBundle) -> tuple[str, ...]:
    warnings: list[str] = []
    for result in (
        bundle.policies,
        bundle.theme_metrics,
        bundle.base_universe,
        bundle.company_profiles,
        bundle.company_theme_map,
        bundle.fundamentals,
        bundle.news,
        bundle.market_state,
        bundle.market_panel,
        bundle.factors,
        bundle.positions,
        bundle.announcements,
    ):
        warnings.extend(result.warnings)
    return tuple(warnings)
