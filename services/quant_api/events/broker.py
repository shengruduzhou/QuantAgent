from __future__ import annotations

from datetime import datetime, timezone
from queue import Empty, Full, Queue
from threading import RLock
from typing import Any, Iterable
from uuid import uuid4

from services.quant_api.events.contracts import EventEnvelope


class SubscriptionClosed(RuntimeError):
    pass


class EventSubscription:
    """Thread-safe bounded subscription used by async WebSocket adapters."""

    def __init__(self, topics: frozenset[str], queue_size: int) -> None:
        self.id = f"sub_{uuid4().hex}"
        self.topics = topics
        self._queue: Queue[EventEnvelope | None] = Queue(maxsize=queue_size)
        self._lock = RLock()
        self._closed = False
        self._dropped = 0

    def get(self, timeout: float) -> EventEnvelope:
        item = self._queue.get(timeout=timeout)
        if item is None:
            raise SubscriptionClosed("event subscription is closed")
        return item

    def take_dropped(self) -> int:
        with self._lock:
            dropped = self._dropped
            self._dropped = 0
            return dropped

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            while True:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
            self._queue.put_nowait(None)

    def offer(self, event: EventEnvelope) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._queue.put_nowait(event)
            except Full:
                self._discard_oldest()
                self._dropped += 1
                self._queue.put_nowait(event)

    def matches(self, topic: str) -> bool:
        return "*" in self.topics or any(
            topic == subscribed or topic.startswith(f"{subscribed}:")
            for subscribed in self.topics
        )

    def _discard_oldest(self) -> None:
        try:
            self._queue.get_nowait()
        except Empty:
            pass


class EventBroker:
    """Small in-process event bridge; domain state remains in existing services."""

    def __init__(self, *, default_queue_size: int = 256) -> None:
        if default_queue_size < 1:
            raise ValueError("default_queue_size must be positive")
        self.default_queue_size = default_queue_size
        self._lock = RLock()
        self._subscriptions: dict[str, EventSubscription] = {}
        self._sequence = 0
        self._running = False

    def start(self) -> None:
        with self._lock:
            self._running = True

    def close(self) -> None:
        with self._lock:
            self._running = False
            subscriptions = list(self._subscriptions.values())
            self._subscriptions.clear()
        for subscription in subscriptions:
            subscription.close()

    def subscribe(
        self,
        topics: Iterable[str],
        *,
        queue_size: int | None = None,
    ) -> EventSubscription:
        normalized = frozenset(topic.strip() for topic in topics if topic.strip())
        if not normalized:
            raise ValueError("at least one event topic is required")
        subscription = EventSubscription(normalized, queue_size or self.default_queue_size)
        with self._lock:
            if not self._running:
                raise RuntimeError("event broker is not running")
            self._subscriptions[subscription.id] = subscription
        return subscription

    def unsubscribe(self, subscription: EventSubscription) -> None:
        with self._lock:
            self._subscriptions.pop(subscription.id, None)
        subscription.close()

    def publish(
        self,
        *,
        topic: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        source: str,
        correlation_id: str | None = None,
    ) -> EventEnvelope:
        event = self.create_event(
            topic=topic,
            event_type=event_type,
            payload=payload,
            source=source,
            correlation_id=correlation_id,
        )
        with self._lock:
            subscriptions = list(self._subscriptions.values()) if self._running else []
        for subscription in subscriptions:
            if subscription.matches(topic):
                subscription.offer(event)
        return event

    def create_event(
        self,
        *,
        topic: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        source: str,
        correlation_id: str | None = None,
    ) -> EventEnvelope:
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        return EventEnvelope(
            eventType=event_type,
            topic=topic,
            occurredAt=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            source=source,
            sequence=sequence,
            correlationId=correlation_id,
            payload=payload or {},
        )

    def stats(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "running": self._running,
                "subscribers": len(self._subscriptions),
                "sequence": self._sequence,
            }


__all__ = ["EventBroker", "EventSubscription", "SubscriptionClosed"]
