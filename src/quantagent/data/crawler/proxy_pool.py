from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class ProxyEndpoint:
    url: str


class ProxyProvider:
    def get_proxy(self) -> ProxyEndpoint | None:
        return None

    def report_success(self, proxy: ProxyEndpoint | None) -> None:
        return None

    def report_failure(self, proxy: ProxyEndpoint | None, reason: str) -> None:
        return None


class StaticProxyPool(ProxyProvider):
    def __init__(self, proxies: tuple[str, ...] = ()) -> None:
        self._proxies = [ProxyEndpoint(url.strip()) for url in proxies if url.strip()]
        self._cursor = 0
        self.failures: dict[str, int] = {}

    def get_proxy(self) -> ProxyEndpoint | None:
        if not self._proxies:
            return None
        proxy = self._proxies[self._cursor % len(self._proxies)]
        self._cursor += 1
        return proxy

    def report_failure(self, proxy: ProxyEndpoint | None, reason: str) -> None:
        if proxy is not None:
            self.failures[proxy.url] = self.failures.get(proxy.url, 0) + 1


class EnvProxyProvider(StaticProxyPool):
    def __init__(self, env_var: str = "QUANTAGENT_PROXY_POOL") -> None:
        raw = os.getenv(env_var, "")
        proxies = tuple(item.strip() for item in raw.split(",") if item.strip())
        super().__init__(proxies)


class FileProxyProvider(StaticProxyPool):
    def __init__(self, path: str | Path) -> None:
        proxy_path = Path(path)
        if proxy_path.exists():
            proxies = tuple(line.strip() for line in proxy_path.read_text(encoding="utf-8").splitlines() if line.strip())
        else:
            proxies = ()
        super().__init__(proxies)
