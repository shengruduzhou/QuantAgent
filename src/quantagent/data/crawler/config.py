from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CrawlerConfig:
    allow_network: bool = False
    timeout_seconds: float = 10.0
    user_agent: str = "QuantAgent-V7-ResearchBot/0.2"
    max_links_per_index: int = 50
    max_retries: int = 2
    backoff_base_seconds: float = 0.25
    backoff_jitter_seconds: float = 0.15
    global_rate_limit_per_second: float = 5.0
    per_domain_rate_limit_per_second: float = 1.0
    respect_robots_txt: bool = True
    domain_allowlist: tuple[str, ...] = ()
    rotate_proxy_status_codes: tuple[int, ...] = (403, 407, 429, 500, 502, 503, 504)
    blocked_status_codes: tuple[int, ...] = (401, 403, 407, 429, 451)
    captcha_markers: tuple[str, ...] = (
        "captcha",
        "verify you are human",
        "access denied",
        "security check",
        "robot check",
        "too many requests",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_rate_limit_per_second", min(float(self.global_rate_limit_per_second), 5.0))
        object.__setattr__(self, "per_domain_rate_limit_per_second", min(float(self.per_domain_rate_limit_per_second), 1.0))
        object.__setattr__(self, "max_links_per_index", max(0, int(self.max_links_per_index)))
        object.__setattr__(self, "max_retries", max(0, int(self.max_retries)))

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "CrawlerConfig":
        if not value:
            return cls()
        allowed = {field for field in cls.__dataclass_fields__}
        return cls(**{key: data for key, data in value.items() if key in allowed})
