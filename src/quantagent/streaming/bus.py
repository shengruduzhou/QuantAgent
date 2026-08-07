"""The event bus: one total order, one frontier, and no way to see the future.

An event-driven engine is only as trustworthy as the order it processes things
in. Three failure modes make a streaming backtest quietly wrong, and each has a
countermeasure here rather than a convention:

* **Arrival order leaking into results.** Feeds deliver out of order; a bus that
  emitted in arrival order would make the answer depend on network timing. Events
  are buffered in a heap and emitted in `MarketEvent.sort_key` order, so the same
  events shuffled any way produce the same run.
* **The past being rewritten.** Once an event has been emitted, consumers have
  decided on it. An event that sorts *before* the last emitted one is refused as a
  `LateArrival` — never silently reordered (which would make the emitted sequence
  a lie) and never silently dropped (which would lose a fill).
* **The future leaking in.** Draining advances a `DecisionFrontier`, and consumers
  read facts through it. A consumer that reaches for a bar not yet available gets
  a `LookAheadViolation`, not a number.

One more rule, which is a modelling statement rather than a mechanism: a consumer
may publish reactions — signals, orders, fills — but never market data. Injecting
a quote in response to having seen a quote is how a backtest invents the tape it
wanted, so `MARKET_KINDS` is refused from inside a drain.

Checkpoints are a running digest of what has been emitted, not a position index.
Resuming from a position alone would happily continue a *different* stream; the
digest makes a resumed run provably the same run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
import itertools
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from quantagent.domain.timeline import DecisionFrontier, EventTime, ensure_aware
from quantagent.streaming.events import MARKET_KINDS, EventKind, MarketEvent

GENESIS_DIGEST = "0" * 64


class LateArrival(RuntimeError):
    """An event arrived that sorts before one already emitted.

    Not recoverable by reordering: consumers have already decided on the events
    emitted so far, so slotting this one in behind them would make the emitted
    sequence a description of something that never happened. Refused loudly so the
    upstream ordering problem is fixed rather than absorbed.
    """

    def __init__(self, event: MarketEvent, last_emitted: MarketEvent) -> None:
        super().__init__(
            f"{event.kind.value} at {event.times.event_time.isoformat()} "
            f"({event.event_id}) sorts before the last emitted event "
            f"{last_emitted.kind.value} at {last_emitted.times.event_time.isoformat()} "
            f"({last_emitted.event_id}). Consumers have already decided on that one."
        )
        self.event = event
        self.last_emitted = last_emitted


class MarketDataFromConsumer(RuntimeError):
    """A consumer tried to publish market data while reacting to market data."""

    def __init__(self, event: MarketEvent) -> None:
        super().__init__(
            f"a consumer published {event.kind.value}, which is market data. A run may "
            "react to the tape and may not extend it; publish market data before the "
            "drain begins, from the source that observed it."
        )
        self.event = event


class DuplicateEvent(RuntimeError):
    """The same content-addressed event was published twice.

    Two identical events are one event delivered twice, and admitting both would
    double whatever they cause — a bar counted twice, a fill booked twice.
    """

    def __init__(self, event: MarketEvent) -> None:
        super().__init__(
            f"{event.kind.value} {event.event_id} was already published; an identical "
            "event is a redelivery, not a second occurrence"
        )
        self.event = event


@dataclass(frozen=True, slots=True)
class BusCheckpoint:
    """Enough to prove a resumed run is the same run.

    `digest` chains every emitted event id. A checkpoint carrying only a position
    would let a resume continue a different stream that happened to be the same
    length, which is the failure a checkpoint exists to prevent.
    """

    emitted: int
    digest: str
    frontier: str
    last_event_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "emitted": self.emitted,
            "digest": self.digest,
            "frontier": self.frontier,
            "lastEventId": self.last_event_id,
        }


class EventBus:
    """A deterministic, strictly ordered event queue with a moving time frontier."""

    def __init__(self, *, allow_duplicates: bool = False) -> None:
        self._heap: list[tuple[Any, ...]] = []
        #: Breaks heap ties for events whose sort keys are genuinely equal, which
        #: can only happen for events that are identical — so it never affects the
        #: emitted order, it only keeps `heapq` from comparing `MarketEvent`s.
        self._counter = itertools.count()
        self._emitted = 0
        self._digest = GENESIS_DIGEST
        self._last: MarketEvent | None = None
        self._seen: set[str] = set()
        self._allow_duplicates = allow_duplicates
        self._draining = False
        self._frontier: DecisionFrontier | None = None

    # -- publishing ---------------------------------------------------------
    def publish(self, event: MarketEvent) -> MarketEvent:
        """Buffer an event. Ordering happens on the way out, not on the way in."""
        if not self._allow_duplicates and event.event_id in self._seen:
            raise DuplicateEvent(event)
        if self._draining and event.kind in MARKET_KINDS:
            raise MarketDataFromConsumer(event)
        if self._last is not None and event.sort_key() < self._last.sort_key():
            raise LateArrival(event, self._last)
        self._seen.add(event.event_id)
        heapq.heappush(self._heap, (event.sort_key(), next(self._counter), event))
        return event

    def publish_all(self, events: Iterable[MarketEvent]) -> list[MarketEvent]:
        return [self.publish(event) for event in events]

    # -- draining -----------------------------------------------------------
    def drain(self, *, until: Any = None) -> Iterator[MarketEvent]:
        """Emit events in total order, advancing the frontier as it goes.

        Re-entrant by design: a consumer reacting to an event may publish more,
        and those are picked up by this same loop if they sort after the current
        position. That is what makes it an event-driven engine rather than a
        two-phase batch.
        """
        limit = ensure_aware(until, field="until") if until is not None else None
        # Clear any flag left set by a drain the caller abandoned mid-loop. The
        # generator's own cleanup runs at collection time, which is not a moment
        # this class can depend on.
        self._draining = False
        while self._heap:
            key, _, event = self._heap[0]
            if limit is not None and event.times.event_time > limit:
                break
            heapq.heappop(self._heap)
            self._emit(event)
            # True only while the consumer holds control, so "is a consumer
            # publishing this?" is answered by where execution actually is rather
            # than by a flag spanning the whole loop — which would keep refusing
            # market data after a caller broke out early.
            self._draining = True
            try:
                yield event
            finally:
                self._draining = False

    def _emit(self, event: MarketEvent) -> None:
        self._last = event
        self._emitted += 1
        self._digest = hashlib.sha256(
            f"{self._digest}{event.event_id}".encode("utf-8")
        ).hexdigest()
        frontier_time = event.times.event_time
        self._frontier = (
            DecisionFrontier(decision_time=frontier_time)
            if self._frontier is None
            else self._frontier.advanced_to(frontier_time)
        )

    def run(self, handler: Callable[[MarketEvent, DecisionFrontier], None], *, until: Any = None) -> int:
        """Drive `handler` over every event. Returns how many were emitted.

        The handler is given the frontier rather than being trusted to track time
        itself: a consumer that kept its own clock could drift from the bus, and
        the drift would look like a strategy that is slightly early.
        """
        count = 0
        for event in self.drain(until=until):
            handler(event, self.frontier)
            count += 1
        return count

    # -- state --------------------------------------------------------------
    @property
    def frontier(self) -> DecisionFrontier:
        if self._frontier is None:
            raise RuntimeError(
                "the frontier does not exist until the first event is emitted: before "
                "that there is no instant the run is deciding at"
            )
        return self._frontier

    @property
    def emitted(self) -> int:
        return self._emitted

    @property
    def digest(self) -> str:
        """Running hash of every emitted event id, in emission order."""
        return self._digest

    @property
    def pending(self) -> int:
        return len(self._heap)

    def peek(self) -> MarketEvent | None:
        return self._heap[0][2] if self._heap else None

    def checkpoint(self) -> BusCheckpoint:
        return BusCheckpoint(
            emitted=self._emitted,
            digest=self._digest,
            frontier=self.frontier.decision_time.isoformat() if self._frontier else "",
            last_event_id=self._last.event_id if self._last else None,
        )

    def resume_from(self, checkpoint: BusCheckpoint) -> None:
        """Adopt a checkpoint's position so continued draining extends the same run.

        Deliberately does *not* replay the events behind the checkpoint: they were
        already processed, and re-emitting them would double their effects. What
        it adopts is the digest, so the resumed run's digest can be compared with
        an uninterrupted one and prove they are the same run.
        """
        if self._emitted:
            raise RuntimeError(
                "resume_from must be called on a fresh bus: adopting a checkpoint "
                "after emitting would splice two runs together"
            )
        self._emitted = checkpoint.emitted
        self._digest = checkpoint.digest
        if checkpoint.frontier:
            self._frontier = DecisionFrontier.at(checkpoint.frontier)


def replay(events: Sequence[MarketEvent]) -> tuple[list[MarketEvent], str]:
    """Order a fixed set of events and digest the result.

    The comparison primitive for determinism testing: two shuffles of one input
    must produce the same list and the same digest.
    """
    bus = EventBus()
    # Published as given. Sorting first would make this function prove that
    # `sorted` works rather than that the bus orders what it is handed — nothing
    # has been emitted yet, so any arrival order is legitimate here.
    bus.publish_all(events)
    emitted = list(bus.drain())
    return emitted, bus.digest


__all__ = [
    "BusCheckpoint",
    "DuplicateEvent",
    "EventBus",
    "GENESIS_DIGEST",
    "LateArrival",
    "MarketDataFromConsumer",
    "replay",
]
