from __future__ import annotations

from typing import Iterable

import pandas as pd

from quantagent.data.crawler.config import CrawlerConfig
from quantagent.data.crawler.fetcher import CrawlFetcher
from quantagent.data.crawler.parser import CrawledDocument
from quantagent.data.crawler.proxy_pool import ProxyProvider
from quantagent.data.providers.base import ProviderResult


class PublicWebCrawler:
    """Production-style public crawler facade used by V7 ingestors.

    The class keeps the previous public API while delegating network, rate
    limit, robots, proxy, canonicalization, parser and metrics behavior to
    the crawler package.
    """

    def __init__(
        self,
        allow_network: bool = False,
        timeout_seconds: float = 10.0,
        max_links_per_index: int = 50,
        *,
        config: CrawlerConfig | None = None,
        proxy_provider: ProxyProvider | None = None,
    ) -> None:
        self.config = config or CrawlerConfig(
            allow_network=allow_network,
            timeout_seconds=timeout_seconds,
            max_links_per_index=max_links_per_index,
        )
        self.fetcher = CrawlFetcher.build(self.config, proxy_provider=proxy_provider)

    @property
    def metrics(self):
        return self.fetcher.metrics

    def fetch_documents(
        self,
        urls: Iterable[str],
        *,
        as_of_date: str,
        source_type: str,
        source_reliability: float,
    ) -> list[CrawledDocument]:
        return self.fetcher.fetch_documents(
            urls,
            as_of_date=as_of_date,
            source_type=source_type,
            source_reliability=source_reliability,
        )

    def discover_documents(
        self,
        index_urls: Iterable[str],
        *,
        as_of_date: str,
        source_type: str,
        source_reliability: float,
    ) -> list[CrawledDocument]:
        return self.fetcher.discover_documents(
            index_urls,
            as_of_date=as_of_date,
            source_type=source_type,
            source_reliability=source_reliability,
        )


def documents_to_result(documents: list[CrawledDocument], source: str, warnings: tuple[str, ...] = ()) -> ProviderResult:
    frame = pd.DataFrame([doc.__dict__ for doc in documents])
    if not frame.empty:
        frame["hash"] = frame["content_hash"]
        frame["point_in_time_valid"] = pd.to_datetime(frame["available_at"]) <= pd.to_datetime(frame["ingested_at"])
    return ProviderResult(
        frame=frame,
        source=source,
        point_in_time=True,
        quality_score=0.80 if documents else 0.0,
        warnings=warnings,
        metadata={
            "document_count": len(documents),
            "blocked_count": int(frame["blocked"].sum()) if not frame.empty and "blocked" in frame else 0,
        },
    )
