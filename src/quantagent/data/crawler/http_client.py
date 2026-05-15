from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from quantagent.data.crawler.canonicalize import url_host
from quantagent.data.crawler.config import CrawlerConfig
from quantagent.data.crawler.metrics import CrawlerMetrics
from quantagent.data.crawler.proxy_pool import ProxyEndpoint, ProxyProvider
from quantagent.data.crawler.rate_limiter import TokenBucket
from quantagent.data.crawler.robots_policy import RobotsPolicy


@dataclass(frozen=True)
class HttpResponse:
    url: str
    status_code: int
    body: str
    headers: dict[str, str]
    proxy_url: str = ""


Transport = Callable[[str, dict[str, str], float, ProxyEndpoint | None], HttpResponse]


class CrawlerHttpClient:
    def __init__(
        self,
        config: CrawlerConfig,
        *,
        metrics: CrawlerMetrics | None = None,
        proxy_provider: ProxyProvider | None = None,
        robots_policy: RobotsPolicy | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.config = config
        self.metrics = metrics or CrawlerMetrics()
        self.proxy_provider = proxy_provider or ProxyProvider()
        self.robots_policy = robots_policy or RobotsPolicy(config)
        self.transport = transport
        self.global_bucket = TokenBucket(config.global_rate_limit_per_second)
        self.domain_buckets: dict[str, TokenBucket] = {}

    def fetch(self, url: str, *, etag: str = "", last_modified: str = "") -> HttpResponse:
        if not self.robots_policy.allowed(url):
            self.metrics.robots_blocked += 1
            return HttpResponse(url=url, status_code=999, body="", headers={"X-Robots-Blocked": "1"})
        headers = {"User-Agent": self.config.user_agent}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        proxy = self.proxy_provider.get_proxy()
        last_response: HttpResponse | None = None
        for attempt in range(self.config.max_retries + 1):
            self._wait_for_rate_limit(url)
            response = self._request_once(url, headers, proxy)
            self.metrics.record_status(response.status_code)
            if response.status_code in self.config.rotate_proxy_status_codes:
                self.proxy_provider.report_failure(proxy, f"status:{response.status_code}")
                proxy = self.proxy_provider.get_proxy()
            else:
                self.proxy_provider.report_success(proxy)
            last_response = response
            if response.status_code in self.config.blocked_status_codes:
                self.metrics.blocked += 1
                return response
            if response.status_code < 500 or attempt >= self.config.max_retries:
                return response
            self.metrics.retries += 1
            time.sleep(self._backoff(attempt))
        return last_response or HttpResponse(url=url, status_code=599, body="", headers={})

    def _request_once(self, url: str, headers: dict[str, str], proxy: ProxyEndpoint | None) -> HttpResponse:
        if self.transport is not None:
            return self.transport(url, headers, self.config.timeout_seconds, proxy)
        request = Request(url, headers=headers)
        try:
            if proxy is None:
                with urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310
                    charset = response.headers.get_content_charset() or "utf-8"
                    body = response.read().decode(charset, errors="replace")
                    return HttpResponse(url=url, status_code=int(response.status), body=body, headers=dict(response.headers))
            handler = ProxyHandler({"http": proxy.url, "https": proxy.url})
            opener = build_opener(handler)
            with opener.open(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310
                charset = response.headers.get_content_charset() or "utf-8"
                body = response.read().decode(charset, errors="replace")
                return HttpResponse(url=url, status_code=int(response.status), body=body, headers=dict(response.headers), proxy_url=proxy.url)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return HttpResponse(url=url, status_code=int(exc.code), body=body, headers=dict(exc.headers), proxy_url="" if proxy is None else proxy.url)
        except URLError:
            return HttpResponse(url=url, status_code=599, body="", headers={}, proxy_url="" if proxy is None else proxy.url)

    def _wait_for_rate_limit(self, url: str) -> None:
        waits = [self.global_bucket.consume()]
        host = url_host(url)
        bucket = self.domain_buckets.setdefault(host, TokenBucket(self.config.per_domain_rate_limit_per_second))
        waits.append(bucket.consume())
        wait_seconds = max(waits)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def _backoff(self, attempt: int) -> float:
        base = self.config.backoff_base_seconds * (2 ** max(0, attempt))
        jitter = random.uniform(0.0, self.config.backoff_jitter_seconds)
        return base + jitter
