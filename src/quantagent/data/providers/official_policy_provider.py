from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable
from urllib.request import Request, urlopen

from quantagent.themes.policy_crawler import PolicyDocument


@dataclass(frozen=True)
class OfficialPolicySource:
    source: str
    url: str
    source_level: str


class OfficialPolicyCrawler:
    """Minimal official-policy crawler adapter. Network is opt-in and disabled in tests."""

    def __init__(self, allow_network: bool = False, timeout_seconds: float = 10.0) -> None:
        self.allow_network = allow_network
        self.timeout_seconds = timeout_seconds

    def crawl(self, sources: Iterable[OfficialPolicySource], as_of_date: str) -> list[PolicyDocument]:
        if not self.allow_network:
            return []
        documents: list[PolicyDocument] = []
        for index, source in enumerate(sources):
            html = self._fetch(source.url)
            title, body = extract_title_and_text(html)
            documents.append(
                PolicyDocument(
                    document_id=f"{source.source}-{index:04d}",
                    title=title or source.url,
                    body=body,
                    source=source.source,
                    source_level=source.source_level,
                    published_at=as_of_date,
                    raw_reference={"url": source.url},
                )
            )
        return documents

    def _fetch(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": "QuantAgentV7ResearchBot/0.1"})
        with urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310 - explicit opt-in research crawler
            return response.read().decode("utf-8", errors="ignore")


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.title: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        clean = " ".join(data.split())
        if not clean:
            return
        if self.in_title:
            self.title.append(clean)
        self.text.append(clean)


def extract_title_and_text(html: str) -> tuple[str, str]:
    parser = _TextParser()
    parser.feed(html)
    return " ".join(parser.title)[:240], " ".join(parser.text)[:8000]
