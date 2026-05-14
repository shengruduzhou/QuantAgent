from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable
from quantagent.data.providers.web_crawler import PublicWebCrawler, documents_to_result


class TradingViewPublicProvider:
    """Optional public TradingView page crawler for sentiment context, not market truth."""

    def __init__(self, allow_network: bool = False) -> None:
        self.crawler = PublicWebCrawler(allow_network=allow_network)

    def fetch_public_pages(
        self,
        request: ProviderRequest,
        urls: Iterable[str],
        *,
        as_of_date: str,
    ) -> ProviderResult:
        del request
        url_tuple = tuple(urls)
        try:
            documents = self.crawler.fetch_documents(
                url_tuple,
                as_of_date=as_of_date,
                source_type="alternative_data",
                source_reliability=0.35,
            )
        except ProviderUnavailable as exc:
            return ProviderResult(
                pd.DataFrame(),
                source="tradingview_public_provider",
                quality_score=0.0,
                warnings=(str(exc),),
                metadata={"allow_network": False, "url_count": len(url_tuple)},
            )
        result = documents_to_result(documents, "tradingview_public_provider")
        frame = result.frame
        if not frame.empty:
            frame = frame.rename(columns={"content_hash": "news_id", "body": "summary"})
            frame["source"] = "tradingview_public"
            frame["source_type"] = "alternative_data"
            frame["is_primary_source"] = False
            frame["is_official"] = False
            frame["cross_validation_count"] = 0
            frame["rumor_risk"] = 0.50
        return ProviderResult(
            frame=frame,
            source="tradingview_public_provider",
            point_in_time=True,
            quality_score=0.35 if not frame.empty else 0.0,
            warnings=result.warnings,
            metadata={"url_count": len(url_tuple), "usage": "sentiment_context_only"},
        )
