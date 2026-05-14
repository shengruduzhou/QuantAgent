from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable
from quantagent.data.providers.web_crawler import PublicWebCrawler, documents_to_result


class DisclosureWebProvider:
    """Fetch public announcement/disclosure pages for evidence normalization."""

    def __init__(self, allow_network: bool = False) -> None:
        self.crawler = PublicWebCrawler(allow_network=allow_network)

    def fetch_announcements(
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
                source_type="company_announcement",
                source_reliability=0.90,
            )
        except ProviderUnavailable as exc:
            return ProviderResult(
                pd.DataFrame(),
                source="disclosure_web_provider",
                quality_score=0.0,
                warnings=(str(exc),),
                metadata={"allow_network": False, "url_count": len(url_tuple)},
            )
        result = documents_to_result(documents, "disclosure_web_provider")
        frame = result.frame
        if not frame.empty:
            frame = frame.rename(columns={"content_hash": "announcement_id"})
            frame["source_type"] = "company_announcement"
            frame["is_primary_source"] = True
            frame["is_official"] = True
        return ProviderResult(
            frame=frame,
            source="disclosure_web_provider",
            point_in_time=True,
            quality_score=0.85 if not frame.empty else 0.0,
            warnings=result.warnings,
            metadata={"url_count": len(url_tuple)},
        )
