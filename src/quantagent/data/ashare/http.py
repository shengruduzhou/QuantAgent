"""Shared HTTP transport for the public A-share sources.

Public Chinese market-data endpoints are rate-limited by IP and answer a burst
with a TCP reset rather than a 429, so a naive ``requests.get`` loop looks like
a network failure. This module centralises:

* a per-host minimum request interval (token-free pacing);
* bounded retries that distinguish *transient* from *permanent* failures;
* explicit failure objects — a failed fetch never degrades into empty data that
  a caller could mistake for "this security has no history".

``FetchOutcome`` is the only thing adapters return upward, so provenance and the
retry classification are always carried with the payload.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlparse

import requests

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Measured, conservative pacing per public host (seconds between requests).
HOST_MIN_INTERVAL: dict[str, float] = {
    "push2his.eastmoney.com": 0.6,
    "push2.eastmoney.com": 0.6,
    "datacenter-web.eastmoney.com": 0.8,
    "web.ifzq.gtimg.cn": 0.15,
    "ifzq.gtimg.cn": 0.15,
    "qt.gtimg.cn": 0.15,
    "stock.gtimg.cn": 0.25,
    "hq.sinajs.cn": 0.3,
    "finance.sina.com.cn": 0.3,
    "vip.stock.finance.sina.com.cn": 0.5,
    "query.sse.com.cn": 0.5,
    "www.szse.cn": 0.5,
}
DEFAULT_MIN_INTERVAL = 0.3

# Retry classification returned to the caller and recorded in ledgers.
RETRY_OK = "OK"
RETRY_TRANSIENT = "TRANSIENT"              # worth retrying later
RETRY_RATE_LIMITED = "RATE_LIMITED"        # back off, then retry
RETRY_PERMANENT = "PERMANENT"              # endpoint/symbol will not answer
RETRY_ENTITLEMENT = "ENTITLEMENT"          # authorisation/subscription blocker
RETRY_EMPTY = "EMPTY"                      # a valid answer carrying no rows


@dataclass
class FetchOutcome:
    """Result of one HTTP fetch, successful or not."""

    ok: bool
    endpoint: str
    retry_class: str
    status_code: int | None = None
    text: str | None = None
    payload: Any = None
    error: str | None = None
    latency_s: float = 0.0
    attempts: int = 0
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "endpoint": self.endpoint,
            "retry_class": self.retry_class,
            "status_code": self.status_code,
            "bytes": len(self.text or ""),
            "latency_s": round(self.latency_s, 3),
            "attempts": self.attempts,
            "error": self.error,
            "retrieved_at": self.retrieved_at,
        }


class _HostPacer:
    """Serialises requests per host to the configured minimum interval."""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def wait(self, host: str) -> float:
        with self._guard:
            lock = self._locks.setdefault(host, threading.Lock())
        interval = HOST_MIN_INTERVAL.get(host, DEFAULT_MIN_INTERVAL)
        with lock:
            now = time.monotonic()
            last = self._last.get(host, 0.0)
            sleep_for = max(0.0, interval - (now - last))
            if sleep_for:
                time.sleep(sleep_for)
            self._last[host] = time.monotonic()
        return sleep_for


_PACER = _HostPacer()


class HttpClient:
    """Paced, retrying HTTP client with explicit failure classification."""

    def __init__(self, timeout: float = 20.0, max_attempts: int = 3,
                 user_agent: str = DEFAULT_USER_AGENT) -> None:
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent, "Accept": "*/*"})
        self.rate_limit_waits = 0.0

    def get(self, url: str, params: Mapping[str, Any] | None = None,
            headers: Mapping[str, str] | None = None,
            encoding: str | None = None) -> FetchOutcome:
        host = urlparse(url).netloc
        started = time.monotonic()
        last_error: str | None = None
        status: int | None = None
        for attempt in range(1, self.max_attempts + 1):
            self.rate_limit_waits += _PACER.wait(host)
            try:
                resp = self._session.get(url, params=params, headers=dict(headers or {}),
                                         timeout=self.timeout)
                status = resp.status_code
                if encoding:
                    resp.encoding = encoding
                if status in (401, 403):
                    return FetchOutcome(False, url, RETRY_ENTITLEMENT, status,
                                        error=f"HTTP {status}", latency_s=time.monotonic() - started,
                                        attempts=attempt)
                # Tencent answers a throttled client with 501, not 429, so both
                # are treated as rate limiting and backed off hard. Running more
                # than one worker per public host reliably triggers this.
                if status in (429, 501):
                    last_error = f"HTTP {status}"
                    self.rate_limit_waits += min(60.0, 5.0 * attempt + random.random())
                    time.sleep(min(60.0, 5.0 * attempt + random.random()))
                    continue
                if status >= 500:
                    last_error = f"HTTP {status}"
                    time.sleep(min(15.0, 1.5 * attempt))
                    continue
                if status != 200:
                    return FetchOutcome(False, url, RETRY_PERMANENT, status,
                                        error=f"HTTP {status}", latency_s=time.monotonic() - started,
                                        attempts=attempt)
                return FetchOutcome(True, url, RETRY_OK, status, text=resp.text,
                                    latency_s=time.monotonic() - started, attempts=attempt)
            except requests.exceptions.RequestException as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                time.sleep(min(20.0, 1.5 * attempt + random.random()))
        klass = RETRY_RATE_LIMITED if last_error in ("HTTP 429", "HTTP 501") else RETRY_TRANSIENT
        return FetchOutcome(False, url, klass, status, error=last_error,
                            latency_s=time.monotonic() - started, attempts=self.max_attempts)

    def get_json(self, url: str, params: Mapping[str, Any] | None = None,
                 headers: Mapping[str, str] | None = None) -> FetchOutcome:
        outcome = self.get(url, params=params, headers=headers)
        if not outcome.ok:
            return outcome
        try:
            outcome.payload = requests.models.complexjson.loads(outcome.text or "")
        except Exception as exc:  # noqa: BLE001 - malformed body is a permanent failure
            return FetchOutcome(False, outcome.endpoint, RETRY_PERMANENT, outcome.status_code,
                                text=outcome.text, error=f"json decode: {str(exc)[:120]}",
                                latency_s=outcome.latency_s, attempts=outcome.attempts)
        return outcome


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
