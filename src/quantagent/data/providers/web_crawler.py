from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import unescape
import re
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from quantagent.data.providers.base import ProviderResult, ProviderUnavailable


@dataclass(frozen=True)
class CrawledDocument:
    url: str
    title: str
    body: str
    published_at: str
    available_at: str
    ingested_at: str
    source: str
    source_type: str
    source_reliability: float
    content_hash: str


class PublicWebCrawler:
    """Small stdlib crawler for public policy, news, and disclosure pages."""

    def __init__(self, allow_network: bool = False, timeout_seconds: float = 10.0) -> None:
        self.allow_network = allow_network
        self.timeout_seconds = timeout_seconds

    def fetch_documents(
        self,
        urls: Iterable[str],
        *,
        as_of_date: str,
        source_type: str,
        source_reliability: float,
    ) -> list[CrawledDocument]:
        if not self.allow_network:
            raise ProviderUnavailable("public web crawling is disabled; set data.allow_network=true explicitly")
        documents: list[CrawledDocument] = []
        for url in urls:
            raw = self._fetch(url)
            title = _extract_title(raw) or url
            body = _extract_body(raw)
            published_at = _extract_date(raw) or as_of_date
            digest = sha256(f"{url}\n{title}\n{body}".encode("utf-8")).hexdigest()
            documents.append(
                CrawledDocument(
                    url=url,
                    title=title,
                    body=body,
                    published_at=published_at,
                    available_at=published_at,
                    ingested_at=as_of_date,
                    source=_host(url),
                    source_type=source_type,
                    source_reliability=source_reliability,
                    content_hash=digest,
                )
            )
        return documents

    def _fetch(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": "QuantAgent-V7-ResearchBot/0.1"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                content_type = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(content_type, errors="replace")
        except URLError as exc:  # pragma: no cover - network disabled in unit tests
            raise ProviderUnavailable(f"failed to fetch public page: {url}") from exc


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
        metadata={"document_count": len(documents)},
    )


def _extract_title(raw: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return _clean_html(match.group(1))[:240]


def _extract_body(raw: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    text = _clean_html(text)
    return text[:20_000]


def _extract_date(raw: str) -> str | None:
    patterns = (
        r"\b20[0-4][0-9]-[01][0-9]-[0-3][0-9]\b",
        r"\b20[0-4][0-9]/[01][0-9]/[0-3][0-9]\b",
    )
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            return match.group(0).replace("/", "-")
    return None


def _clean_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = unescape(text)
    return " ".join(text.split())


def _host(url: str) -> str:
    match = re.match(r"https?://([^/]+)", url)
    return match.group(1).lower() if match else "public_web"
