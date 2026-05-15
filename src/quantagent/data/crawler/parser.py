from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re

from quantagent.data.crawler.canonicalize import canonicalize_url, stable_content_hash, url_host
from quantagent.data.crawler.config import CrawlerConfig


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
    canonical_url: str
    raw_hash: str
    author: str = ""
    blocked: bool = False
    blocked_reason: str = ""
    status_code: int = 200
    etag: str = ""
    last_modified: str = ""


def parse_document(
    raw: str,
    url: str,
    *,
    as_of_date: str,
    source_type: str,
    source_reliability: float,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    config: CrawlerConfig | None = None,
) -> CrawledDocument:
    config = config or CrawlerConfig()
    headers = headers or {}
    canonical_url = _canonical_from_html(raw, url)
    title = _extract_title(raw) or canonical_url
    body = _extract_body(raw)
    published_at = _extract_date(raw, headers) or as_of_date
    author = _extract_author(raw)
    blocked, blocked_reason = _blocked_reason(raw, status_code, config)
    digest = stable_content_hash(canonical_url, title, body)
    return CrawledDocument(
        url=url,
        title=title,
        body=body,
        published_at=published_at,
        available_at=published_at,
        ingested_at=as_of_date,
        source=url_host(canonical_url) or "public_web",
        source_type=source_type,
        source_reliability=source_reliability,
        content_hash=digest,
        canonical_url=canonical_url,
        raw_hash=digest,
        author=author,
        blocked=blocked,
        blocked_reason=blocked_reason,
        status_code=status_code,
        etag=headers.get("ETag", headers.get("etag", "")),
        last_modified=headers.get("Last-Modified", headers.get("last-modified", "")),
    )


def extract_article_links(raw: str, base_url: str) -> list[str]:
    if not raw:
        return []
    sitemap_links = re.findall(r"<loc>\s*(https?://[^<]+)\s*</loc>", raw, flags=re.IGNORECASE)
    if sitemap_links:
        return [canonicalize_url(link.strip()) for link in sitemap_links if _looks_like_article(link)]
    rss_links = re.findall(r"<link[^>]*>\s*(https?://[^<\s]+)\s*</link>", raw, flags=re.IGNORECASE)
    if rss_links:
        return [canonicalize_url(link.strip()) for link in rss_links if _looks_like_article(link)]
    anchors = re.findall(r"<a[^>]+href=[\"']([^\"'#]+)[\"']", raw, flags=re.IGNORECASE)
    links: list[str] = []
    for href in anchors:
        lower = href.lower()
        if lower.startswith(("javascript:", "mailto:")):
            continue
        absolute = canonicalize_url(href, base_url=base_url)
        if _looks_like_article(absolute):
            links.append(absolute)
    return list(dict.fromkeys(links))


def _extract_title(raw: str) -> str:
    patterns = (
        r"<meta[^>]+property=[\"']og:title[\"'][^>]+content=[\"']([^\"']+)[\"']",
        r"<title[^>]*>(.*?)</title>",
        r"<h1[^>]*>(.*?)</h1>",
    )
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return _clean_html(match.group(1))[:240]
    return ""


def _extract_body(raw: str) -> str:
    text = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    return _clean_html(text)[:20_000]


def _extract_date(raw: str, headers: dict[str, str]) -> str | None:
    meta_patterns = (
        r"<meta[^>]+property=[\"']article:published_time[\"'][^>]+content=[\"']([^\"']+)[\"']",
        r"<meta[^>]+name=[\"']pubdate[\"'][^>]+content=[\"']([^\"']+)[\"']",
        r"<time[^>]+datetime=[\"']([^\"']+)[\"']",
    )
    for pattern in meta_patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)[:10].replace("/", "-")
    for pattern in (r"\b20[0-4][0-9]-[01][0-9]-[0-3][0-9]\b", r"\b20[0-4][0-9]/[01][0-9]/[0-3][0-9]\b"):
        match = re.search(pattern, raw)
        if match:
            return match.group(0).replace("/", "-")
    last_modified = headers.get("Last-Modified", headers.get("last-modified", ""))
    return last_modified[:10] if last_modified else None


def _extract_author(raw: str) -> str:
    for pattern in (
        r"<meta[^>]+name=[\"']author[\"'][^>]+content=[\"']([^\"']+)[\"']",
        r"<meta[^>]+property=[\"']article:author[\"'][^>]+content=[\"']([^\"']+)[\"']",
    ):
        match = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return _clean_html(match.group(1))[:120]
    return ""


def _canonical_from_html(raw: str, url: str) -> str:
    match = re.search(r"<link[^>]+rel=[\"']canonical[\"'][^>]+href=[\"']([^\"']+)[\"']", raw, flags=re.IGNORECASE)
    return canonicalize_url(match.group(1), base_url=url) if match else canonicalize_url(url)


def _blocked_reason(raw: str, status_code: int, config: CrawlerConfig) -> tuple[bool, str]:
    if status_code in set(config.blocked_status_codes):
        return True, f"blocked_status:{status_code}"
    lower = raw.lower()
    for marker in config.captcha_markers:
        if marker.lower() in lower:
            return True, f"blocked_marker:{marker.lower()}"
    return False, ""


def _clean_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    return " ".join(unescape(text).split())


def _looks_like_article(url: str) -> bool:
    lower = url.lower()
    if not lower.startswith(("http://", "https://")):
        return False
    if lower.endswith((".jpg", ".png", ".gif", ".css", ".js", ".ico", ".svg", ".pdf")):
        return False
    keywords = ("article", "news", "policy", "zhengce", "notice", "announcement", "content", "detail", "fagui", "zcfb", "xxgk")
    return any(keyword in lower for keyword in keywords) or bool(re.search(r"/\d{6,}", lower))
