from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from typing import Callable
from urllib import robotparser
from urllib.parse import urlsplit, urlunsplit

from quantagent.data.crawler.canonicalize import url_host
from quantagent.data.crawler.config import CrawlerConfig


@dataclass
class RobotsPolicy:
    config: CrawlerConfig
    fetch_text: Callable[[str], str] | None = None
    _parsers: dict[str, robotparser.RobotFileParser] = field(default_factory=dict, init=False)

    def allowed(self, url: str) -> bool:
        host = url_host(url)
        if self.config.domain_allowlist and host not in set(self.config.domain_allowlist):
            return False
        if not self.config.respect_robots_txt:
            return True
        parser = self._parser_for(url)
        if parser is None:
            return True
        return bool(parser.can_fetch(self.config.user_agent, url))

    def _parser_for(self, url: str) -> robotparser.RobotFileParser | None:
        host = url_host(url)
        if host in self._parsers:
            return self._parsers[host]
        robots_url = _robots_url(url)
        parser = robotparser.RobotFileParser()
        parser.set_url(robots_url)
        if self.fetch_text is None:
            return None
        try:
            raw = self.fetch_text(robots_url)
        except Exception:
            return None
        parser.parse(StringIO(raw).read().splitlines())
        self._parsers[host] = parser
        return parser


def _robots_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme or "https", parts.netloc, "/robots.txt", "", ""))
