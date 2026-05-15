from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CrawlerMetrics:
    requests: int = 0
    retries: int = 0
    blocked: int = 0
    robots_blocked: int = 0
    deduplicated: int = 0
    status_counts: dict[int, int] = field(default_factory=dict)

    def record_status(self, status_code: int) -> None:
        self.requests += 1
        self.status_counts[status_code] = self.status_counts.get(status_code, 0) + 1
