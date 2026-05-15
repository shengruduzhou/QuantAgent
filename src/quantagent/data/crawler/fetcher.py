from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from quantagent.data.crawler.canonicalize import canonicalize_url
from quantagent.data.crawler.config import CrawlerConfig
from quantagent.data.crawler.http_client import CrawlerHttpClient
from quantagent.data.crawler.metrics import CrawlerMetrics
from quantagent.data.crawler.parser import CrawledDocument, extract_article_links, parse_document
from quantagent.data.crawler.proxy_pool import ProxyProvider
from quantagent.data.providers.base import ProviderUnavailable


@dataclass
class CrawlFetcher:
    config: CrawlerConfig = field(default_factory=CrawlerConfig)
    http_client: CrawlerHttpClient | None = None
    metrics: CrawlerMetrics = field(default_factory=CrawlerMetrics)
    etag_cache: dict[str, str] = field(default_factory=dict)
    last_modified_cache: dict[str, str] = field(default_factory=dict)
    seen_hashes: set[str] = field(default_factory=set)

    @classmethod
    def build(
        cls,
        config: CrawlerConfig,
        *,
        proxy_provider: ProxyProvider | None = None,
    ) -> "CrawlFetcher":
        metrics = CrawlerMetrics()
        return cls(
            config=config,
            http_client=CrawlerHttpClient(config, metrics=metrics, proxy_provider=proxy_provider),
            metrics=metrics,
        )

    def fetch_documents(
        self,
        urls: Iterable[str],
        *,
        as_of_date: str,
        source_type: str,
        source_reliability: float,
    ) -> list[CrawledDocument]:
        if not self.config.allow_network:
            raise ProviderUnavailable("public web crawling is disabled; set data.allow_network=true explicitly")
        documents: list[CrawledDocument] = []
        for url in urls:
            document = self.fetch_one(
                url,
                as_of_date=as_of_date,
                source_type=source_type,
                source_reliability=source_reliability,
            )
            if document is None:
                continue
            if document.raw_hash in self.seen_hashes and not document.blocked:
                self.metrics.deduplicated += 1
                continue
            self.seen_hashes.add(document.raw_hash)
            documents.append(document)
        return documents

    def discover_documents(
        self,
        index_urls: Iterable[str],
        *,
        as_of_date: str,
        source_type: str,
        source_reliability: float,
    ) -> list[CrawledDocument]:
        if not self.config.allow_network:
            raise ProviderUnavailable("public web crawling is disabled; set data.allow_network=true explicitly")
        candidates: list[str] = []
        for index_url in index_urls:
            response = self._client().fetch(canonicalize_url(index_url))
            if response.status_code == 304 or not response.body:
                continue
            if response.status_code >= 400:
                continue
            candidates.extend(extract_article_links(response.body, response.url)[: self.config.max_links_per_index])
        return self.fetch_documents(
            list(dict.fromkeys(candidates)),
            as_of_date=as_of_date,
            source_type=source_type,
            source_reliability=source_reliability,
        )

    def fetch_one(
        self,
        url: str,
        *,
        as_of_date: str,
        source_type: str,
        source_reliability: float,
    ) -> CrawledDocument | None:
        canonical_url = canonicalize_url(url)
        response = self._client().fetch(
            canonical_url,
            etag=self.etag_cache.get(canonical_url, ""),
            last_modified=self.last_modified_cache.get(canonical_url, ""),
        )
        if response.status_code == 304:
            return None
        document = parse_document(
            response.body,
            canonical_url,
            as_of_date=as_of_date,
            source_type=source_type,
            source_reliability=source_reliability,
            status_code=response.status_code,
            headers=response.headers,
            config=self.config,
        )
        if document.etag:
            self.etag_cache[canonical_url] = document.etag
        if document.last_modified:
            self.last_modified_cache[canonical_url] = document.last_modified
        if document.blocked:
            self.metrics.blocked += 1
        return document

    def _client(self) -> CrawlerHttpClient:
        if self.http_client is None:
            self.http_client = CrawlerHttpClient(self.config, metrics=self.metrics)
        return self.http_client
