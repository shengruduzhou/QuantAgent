from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import monotonic


@dataclass
class TokenBucket:
    rate_per_second: float
    capacity: float | None = None
    _tokens: float = field(init=False)
    _updated_at: float = field(init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.rate_per_second = max(0.001, float(self.rate_per_second))
        self.capacity = float(self.capacity or self.rate_per_second)
        self._tokens = self.capacity
        self._updated_at = monotonic()

    def consume(self, tokens: float = 1.0) -> float:
        with self._lock:
            now = monotonic()
            elapsed = max(0.0, now - self._updated_at)
            self._tokens = min(float(self.capacity), self._tokens + elapsed * self.rate_per_second)
            self._updated_at = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            missing = tokens - self._tokens
            self._tokens = 0.0
            return missing / self.rate_per_second
