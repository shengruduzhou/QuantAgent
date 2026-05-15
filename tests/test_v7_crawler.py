from quantagent.data.crawler.canonicalize import canonicalize_url
from quantagent.data.crawler.config import CrawlerConfig
from quantagent.data.crawler.fetcher import CrawlFetcher
from quantagent.data.crawler.http_client import CrawlerHttpClient, HttpResponse
from quantagent.data.crawler.parser import extract_article_links


def test_crawler_config_clamps_default_rate_limits():
    config = CrawlerConfig(global_rate_limit_per_second=50, per_domain_rate_limit_per_second=10)

    assert config.global_rate_limit_per_second <= 5.0
    assert config.per_domain_rate_limit_per_second <= 1.0


def test_crawler_canonicalizes_urls_and_extracts_links():
    url = canonicalize_url("HTTPS://Example.COM/news/123456?utm_source=x&b=2&a=1#frag")
    links = extract_article_links(
        '<a href="/news/123456?utm_campaign=x">A</a><a href="/image.png">B</a>',
        "https://example.com/index",
    )

    assert url == "https://example.com/news/123456?a=1&b=2"
    assert links == ["https://example.com/news/123456"]


def test_crawler_parser_marks_captcha_as_blocked_without_retry_storm():
    calls = []

    def transport(url, headers, timeout, proxy):
        calls.append(url)
        return HttpResponse(
            url=url,
            status_code=200,
            body="<html><title>Blocked</title>captcha verify you are human</html>",
            headers={},
        )

    config = CrawlerConfig(allow_network=True, max_retries=3)
    client = CrawlerHttpClient(config, transport=transport)
    fetcher = CrawlFetcher(config=config, http_client=client)

    document = fetcher.fetch_one(
        "https://example.com/news/123456",
        as_of_date="2026-05-14",
        source_type="news",
        source_reliability=0.62,
    )

    assert document is not None and document.blocked is True
    assert document.blocked_reason.startswith("blocked_marker")
    assert len(calls) == 1


def test_crawler_preserves_etag_and_last_modified():
    def transport(url, headers, timeout, proxy):
        return HttpResponse(
            url=url,
            status_code=200,
            body='<html><title>AI server order</title><meta name="author" content="desk">2026-05-13 body</html>',
            headers={"ETag": "abc", "Last-Modified": "2026-05-13"},
        )

    config = CrawlerConfig(allow_network=True)
    fetcher = CrawlFetcher(config=config, http_client=CrawlerHttpClient(config, transport=transport))
    document = fetcher.fetch_one(
        "https://example.com/news/123456",
        as_of_date="2026-05-14",
        source_type="news",
        source_reliability=0.62,
    )

    assert document is not None
    assert document.etag == "abc"
    assert document.last_modified == "2026-05-13"
    assert document.raw_hash
