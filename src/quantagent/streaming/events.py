"""The canonical event taxonomy: everything a streaming run can be told.

One envelope, one vocabulary. The fast engine's dicts, the paper broker's order
states and a market data row are all *the same kind of thing* to a streaming
engine — something that happened, at a time, that something else must react to —
and until they share a type they cannot be put in one ordered queue.

Two properties do the work:

* **Every event carries `EventTime`.** Not a timestamp: the whole nine-stamp model,
  so the queue can order by when a thing *happened* while the look-ahead guard
  reads when it became *knowable*. An event that cannot say when it became
  knowable cannot be checked for look-ahead, which is why there is no constructor
  that takes a bare instant.
* **Every kind has a declared causal priority.** Two events at one instant are
  ordered by what causes what — market observed, then risk decides, then an order
  is sent, then the venue answers, then the session settles. `EventKind` and
  `timeline.EVENT_PRIORITY` are checked against each other at import, so adding a
  kind without deciding where it sits in that chain fails immediately rather than
  silently sorting last.

Event ids are content-addressed. Two runs given the same inputs produce the same
ids, which is what lets a replay be compared to the run it replays rather than
merely resembling it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from quantagent.domain.lineage import Lineage, content_id
from quantagent.domain.timeline import EVENT_PRIORITY, EventTime, ordering_key


class EventKind(str, Enum):
    """Every event a streaming run can carry.

    Ordered here as they occur in a session, which is also their causal order.
    """

    #: Trading calendar facts: which sessions exist, holidays, half days.
    CALENDAR = "CALENDAR"
    SESSION_OPEN = "SESSION_OPEN"
    #: Splits, dividends, share-ratio changes. Before any price is read, because
    #: they reprice everything that follows.
    CORPORATE_ACTION = "CORPORATE_ACTION"
    #: ST/*ST designation, delisting risk, listing status.
    SECURITY_STATE = "SECURITY_STATE"
    #: Trading halted for this symbol. Distinct from SECURITY_STATE: a suspension
    #: is a tradability fact for one session, not a change in what the security is.
    SUSPENSION = "SUSPENSION"
    #: The session's ceiling and floor, which depend on the previous close and on
    #: SECURITY_STATE, so it is ordered after both.
    PRICE_LIMIT = "PRICE_LIMIT"
    QUOTE = "QUOTE"
    TRADE = "TRADE"
    BAR = "BAR"
    #: A model's view. Ordered after all market data at the same instant, because a
    #: signal that preceded the data it read would be exactly the look-ahead the
    #: time model exists to catch.
    SIGNAL = "SIGNAL"
    TARGET = "TARGET"
    RISK_DECISION = "RISK_DECISION"
    ORDER = "ORDER"
    CANCEL = "CANCEL"
    #: The venue answering: acceptance, rejection, acknowledgement.
    VENUE_CALLBACK = "VENUE_CALLBACK"
    FILL = "FILL"
    EXPIRY = "EXPIRY"
    MARK_TO_MARKET = "MARK_TO_MARKET"
    #: T+1 settlement promoting the session's purchases to sellable.
    SETTLEMENT = "SETTLEMENT"
    SESSION_CLOSE = "SESSION_CLOSE"
    #: After the close: accounting is final for the day.
    END_OF_DAY = "END_OF_DAY"


#: Kinds that describe the market rather than our reaction to it. A consumer may
#: never publish one of these: injecting market data in response to having seen
#: market data is how a backtest invents the tape it wanted.
MARKET_KINDS: frozenset[EventKind] = frozenset(
    {
        EventKind.CALENDAR,
        EventKind.SESSION_OPEN,
        EventKind.CORPORATE_ACTION,
        EventKind.SECURITY_STATE,
        EventKind.SUSPENSION,
        EventKind.PRICE_LIMIT,
        EventKind.QUOTE,
        EventKind.TRADE,
        EventKind.BAR,
        EventKind.SESSION_CLOSE,
    }
)

#: Kinds a strategy, risk layer or venue produces while reacting.
REACTION_KINDS: frozenset[EventKind] = frozenset(EventKind) - MARKET_KINDS


def _assert_priorities_cover_every_kind() -> None:
    """Fail at import if a kind has no declared place in the causal order.

    A missing kind would sort after everything via `UNKNOWN_PRIORITY` — a silent
    default that puts, say, a new market-data kind *after* the orders that should
    have reacted to it. Checking here means adding a kind and forgetting its
    priority is a startup error, not a subtly wrong backtest.
    """
    missing = sorted(kind.value for kind in EventKind if kind.value not in EVENT_PRIORITY)
    if missing:
        raise RuntimeError(
            f"EventKind members with no entry in timeline.EVENT_PRIORITY: {missing}. "
            "Decide where each sits in the causal order rather than letting it "
            "default to last."
        )


_assert_priorities_cover_every_kind()


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """One thing that happened, with everything needed to order and audit it."""

    kind: EventKind
    times: EventTime
    #: Absent for account- or calendar-level events (settlement, end of day).
    symbol: str | None = None
    #: The source's own ordering within one instant. Two prints in the same
    #: millisecond are distinguished by this, not by arrival order.
    sequence: int = 0
    payload: Mapping[str, Any] = field(default_factory=dict)
    lineage: Lineage = field(default_factory=Lineage)
    event_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EventKind):
            object.__setattr__(self, "kind", EventKind(str(self.kind)))
        if not self.event_id:
            object.__setattr__(self, "event_id", self._derive_id())

    def _derive_id(self) -> str:
        """Content-addressed, so a replay produces the same ids as the run it replays."""
        return content_id(
            "evt",
            kind=self.kind.value,
            event_time=self.times.event_time.isoformat(),
            available_time=self.times.available_time.isoformat(),
            symbol=self.symbol or "",
            sequence=int(self.sequence),
            payload=sorted((str(k), str(v)) for k, v in dict(self.payload).items()),
        )

    @property
    def is_market_data(self) -> bool:
        return self.kind in MARKET_KINDS

    def sort_key(self) -> tuple[Any, ...]:
        """The bus's total order. Identical for identical events, by construction."""
        return ordering_key(
            self.times, self.kind.value, sequence=self.sequence, identity=self.event_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "eventId": self.event_id,
            "kind": self.kind.value,
            "symbol": self.symbol,
            "sequence": self.sequence,
            "times": self.times.to_dict(),
            "payload": dict(self.payload),
            "lineage": self.lineage.as_dict(),
        }


__all__ = [
    "EventKind",
    "MARKET_KINDS",
    "MarketEvent",
    "REACTION_KINDS",
]
