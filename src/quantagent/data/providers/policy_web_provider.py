from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult, ProviderUnavailable
from quantagent.data.providers.web_crawler import PublicWebCrawler, documents_to_result


class PolicyWebProvider:
    """Fetch public policy documents into V7 PIT policy rows."""

    def __init__(self, allow_network: bool = False) -> None:
        self.crawler = PublicWebCrawler(allow_network=allow_network)

    def fetch_policy_documents(
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
                source_type="official_policy",
                source_reliability=0.88,
            )
        except ProviderUnavailable as exc:
            return ProviderResult(
                pd.DataFrame(),
                source="policy_web_provider",
                quality_score=0.0,
                warnings=(str(exc),),
                metadata={"allow_network": False, "url_count": len(url_tuple)},
            )
        frame = documents_to_result(documents, "policy_web_provider").frame
        if not frame.empty:
            frame = frame.rename(columns={"content_hash": "document_id"})
            frame["source_level"] = frame["source"].map(_source_level)
            frame["raw_reference"] = frame.apply(lambda row: {"url": row["url"], "hash": row["hash"]}, axis=1)
        return ProviderResult(
            frame=frame,
            source="policy_web_provider",
            point_in_time=True,
            quality_score=0.82 if not frame.empty else 0.0,
            warnings=(),
            metadata={"url_count": len(url_tuple)},
        )


def _source_level(source: str) -> str:
    host = source.lower()
    if any(token in host for token in ("www.gov.cn", "gov.cn")):
        return "central"
    if any(token in host for token in ("miit", "ndrc", "most", "mof")):
        return "ministry"
    return "media_interpretation"
