"""The explicit time model: when a thing happened, and when it became knowable.

A backtest that carries one timestamp per row cannot answer the only question that
matters for look-ahead — *could the strategy have known this yet?* — because "when
it happened" and "when it was knowable" are different instants, and for the most
dangerous data they differ by a whole session. A daily bar's close happens at
15:00; it becomes knowable at 15:00. A strategy deciding at 09:30 that reads that
close is not slightly optimistic, it is reading the future, and no single-timestamp
model can detect it.

So every fact carries `EventTime`, which separates:

* `source_time` — when the venue or provider stamped it.
* `event_time` — when it happened in the market.
* `available_time` — the first instant a strategy could legitimately have known
  it. This is the field the look-ahead guard reads, and the only one that can
  refuse a decision.
* `ingestion_time` / `processing_time` — when *our* system received and handled
  it. Operational, never economic: a slow pipeline must not change what a
  strategy was entitled to know.

and, for the action side of an order's life, `decision_time`, `submission_time`,
`venue_receive_time` and `fill_time`.

Two rules are enforced rather than documented:

* **Nothing is available before it happened.** `available_time < event_time` is
  rejected at construction, because that ordering is not a modelling choice — it
  is a claim to have known something in advance.
* **A decision may only read facts already available.** `DecisionFrontier.admit`
  raises rather than returning a boolean, so a caller cannot ignore the answer by
  forgetting to check it.

Timestamps must be timezone-aware. A naive timestamp compared against an aware one
raises in Python, and worse, two naive timestamps from different zones compare
successfully and wrongly — which is exactly how an off-by-one-session look-ahead
survives review.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

#: Every A-share instant is anchored here. Sessions are defined in exchange local
#: time, so deriving availability in UTC would put the boundary in the wrong place
#: for half the year.
EXCHANGE_TZ = ZoneInfo("Asia/Shanghai")

#: A-share continuous trading, in exchange local time.
MORNING_OPEN = time(9, 30)
MORNING_CLOSE = time(11, 30)
AFTERNOON_OPEN = time(13, 0)
#: The closing call auction prints at 15:00:00; a daily bar's close is not
#: knowable before it.
SESSION_CLOSE = time(15, 0)
#: The opening call auction collects from 09:15 and prints at 09:25.
OPENING_AUCTION_PRINT = time(9, 25)


class NaiveTimestamp(ValueError):
    """A timestamp arrived without a timezone.

    Refused rather than assumed. Two naive timestamps from different zones compare
    successfully and wrongly, which is how a look-ahead of exactly one session
    passes review: every assertion holds, in the wrong frame.
    """


class ImpossibleAvailability(ValueError):
    """A fact claimed to be knowable before it happened."""


class LookAheadViolation(RuntimeError):
    """A decision tried to consume a fact that was not yet available.

    Carries both instants and the gap, because "look-ahead detected" is not
    actionable and "available at 15:00, decided at 09:30, 5h30m early" is.
    """

    def __init__(self, description: str, available_time: datetime, decision_time: datetime) -> None:
        gap = available_time - decision_time
        super().__init__(
            f"{description} becomes available at {available_time.isoformat()} but the "
            f"decision frontier is {decision_time.isoformat()} — {gap} in the future"
        )
        self.description = description
        self.available_time = available_time
        self.decision_time = decision_time
        self.gap = gap


def ensure_aware(moment: datetime | str, *, field: str = "timestamp") -> datetime:
    """Parse to an aware `datetime`, refusing anything without a zone."""
    parsed = datetime.fromisoformat(moment) if isinstance(moment, str) else moment
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise NaiveTimestamp(
            f"{field} {parsed.isoformat()} has no timezone. An aware timestamp is "
            "required: naive values from different zones compare without error and "
            "with the wrong answer."
        )
    return parsed


def _optional(moment: datetime | str | None, *, field: str) -> datetime | None:
    return None if moment is None else ensure_aware(moment, field=field)


@dataclass(frozen=True, slots=True)
class EventTime:
    """When something happened, when it became knowable, and what we did about it."""

    #: When it happened in the market. Required: an event with no event time cannot
    #: be ordered against anything.
    event_time: datetime
    #: The first instant a strategy could legitimately have known it. Defaults to
    #: `event_time` only for facts that are knowable the moment they occur — a
    #: trade print, a quote. Bar and fundamental data must set it explicitly.
    available_time: datetime
    source_time: datetime | None = None
    ingestion_time: datetime | None = None
    processing_time: datetime | None = None
    decision_time: datetime | None = None
    submission_time: datetime | None = None
    venue_receive_time: datetime | None = None
    fill_time: datetime | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "event_time", "available_time", "source_time", "ingestion_time",
            "processing_time", "decision_time", "submission_time",
            "venue_receive_time", "fill_time",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            object.__setattr__(self, field_name, ensure_aware(value, field=field_name))
        if self.available_time < self.event_time:
            raise ImpossibleAvailability(
                f"available_time {self.available_time.isoformat()} precedes event_time "
                f"{self.event_time.isoformat()}: a fact cannot be knowable before it "
                "happens, and treating this as a tolerable rounding error is how "
                "look-ahead enters a backtest"
            )

    # -- construction -------------------------------------------------------
    @classmethod
    def immediate(cls, moment: datetime | str, **stamps: Any) -> "EventTime":
        """A fact knowable the instant it occurs: a print, a quote, an order event."""
        aware = ensure_aware(moment, field="event_time")
        return cls(event_time=aware, available_time=aware, **stamps)

    @classmethod
    def bar(
        cls,
        *,
        session: date | str,
        interval: str = "1d",
        source_time: datetime | str | None = None,
        **stamps: Any,
    ) -> "EventTime":
        """A bar: it *happens* over a window and is *knowable* when the window closes.

        This is the constructor that prevents the common defect. A daily bar keyed
        only by its session date invites code to treat it as available at the start
        of that session, which makes the close — and therefore that day's return —
        readable before it exists.
        """
        closes_at = bar_close(session, interval=interval)
        return cls(
            event_time=closes_at,
            available_time=closes_at,
            source_time=_optional(source_time, field="source_time"),
            **stamps,
        )

    def with_decision(self, moment: datetime | str) -> "EventTime":
        return replace(self, decision_time=ensure_aware(moment, field="decision_time"))

    def with_submission(self, moment: datetime | str) -> "EventTime":
        return replace(self, submission_time=ensure_aware(moment, field="submission_time"))

    def with_fill(
        self, moment: datetime | str, *, venue_receive_time: datetime | str | None = None
    ) -> "EventTime":
        return replace(
            self,
            fill_time=ensure_aware(moment, field="fill_time"),
            venue_receive_time=_optional(venue_receive_time, field="venue_receive_time"),
        )

    # -- queries ------------------------------------------------------------
    @property
    def availability_lag(self) -> timedelta:
        """How long after happening the fact became knowable."""
        return self.available_time - self.event_time

    def available_at(self, moment: datetime | str) -> bool:
        return self.available_time <= ensure_aware(moment, field="moment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "eventTime": self.event_time.isoformat(),
            "availableTime": self.available_time.isoformat(),
            "sourceTime": self.source_time.isoformat() if self.source_time else None,
            "ingestionTime": (
                self.ingestion_time.isoformat() if self.ingestion_time else None
            ),
            "processingTime": (
                self.processing_time.isoformat() if self.processing_time else None
            ),
            "decisionTime": self.decision_time.isoformat() if self.decision_time else None,
            "submissionTime": (
                self.submission_time.isoformat() if self.submission_time else None
            ),
            "venueReceiveTime": (
                self.venue_receive_time.isoformat() if self.venue_receive_time else None
            ),
            "fillTime": self.fill_time.isoformat() if self.fill_time else None,
            "availabilityLagSeconds": self.availability_lag.total_seconds(),
        }


# ---------------------------------------------------------------------------
# Session arithmetic
# ---------------------------------------------------------------------------
def _as_date(session: date | str) -> date:
    return session if isinstance(session, date) else date.fromisoformat(str(session))


def exchange_moment(session: date | str, clock: time) -> datetime:
    """An exchange-local instant on a session."""
    return datetime.combine(_as_date(session), clock, tzinfo=EXCHANGE_TZ)


def session_close(session: date | str) -> datetime:
    return exchange_moment(session, SESSION_CLOSE)


def session_open(session: date | str) -> datetime:
    return exchange_moment(session, MORNING_OPEN)


def bar_close(session: date | str, *, interval: str = "1d") -> datetime:
    """When a bar's window ends, which is when it becomes knowable.

    Only whole-session bars are resolved here. An intraday interval needs the bar's
    start to know its end, so asking for one by session alone is a question with no
    answer, and returning the session close would be a plausible wrong number.
    """
    if interval in {"1d", "d", "day", "daily"}:
        return session_close(session)
    raise ValueError(
        f"interval {interval!r} needs the bar's own window, not just a session: pass "
        "EventTime(event_time=..., available_time=...) with the bar's end instant"
    )


def intraday_bar(start: datetime | str, *, minutes: int) -> EventTime:
    """An intraday bar, knowable when its window closes."""
    begins = ensure_aware(start, field="start")
    ends = begins + timedelta(minutes=minutes)
    return EventTime(event_time=ends, available_time=ends)


#: When each column of a daily A-share bar becomes knowable. The whole bar is not
#: one fact: `pre_close` was settled yesterday, `open` prints in the opening
#: auction, and `high`/`low`/`close`/`volume` are only final at the close. Treating
#: a bar row as a single timestamped fact is what lets a strategy that decides at
#: the open read a close it cannot know — the panel column is right there in the
#: same row, and nothing about the row's shape says otherwise.
_BAR_FIELD_CLOCK: Mapping[str, time] = {
    "open": OPENING_AUCTION_PRINT,
    "high": SESSION_CLOSE,
    "low": SESSION_CLOSE,
    "close": SESSION_CLOSE,
    "volume": SESSION_CLOSE,
    "amount": SESSION_CLOSE,
    "vwap": SESSION_CLOSE,
}

#: Columns settled by an *earlier* session. `pre_close` is yesterday's close, so
#: reading it at today's open is legitimate and is the correct basis for a
#: price-limit calculation.
_PRIOR_SESSION_FIELDS: frozenset[str] = frozenset({"pre_close", "previous_close"})


def bar_field_availability(field: str, session: date | str) -> EventTime:
    """When one column of a daily bar became knowable.

    Raises on an unknown column rather than guessing. A default of "the close"
    would be safe-by-accident for most fields and wrong for the ones that matter,
    and a default of "the open" would invent look-ahead silently.
    """
    normalised = field.strip().lower()
    if normalised in _PRIOR_SESSION_FIELDS:
        # Available from the moment this session can be traded at all. Its own
        # event time is the previous close, which we do not need a calendar to
        # place: what matters is that it is knowable before this session opens.
        opens_at = session_open(session)
        return EventTime(event_time=opens_at - timedelta(days=1), available_time=opens_at)
    clock = _BAR_FIELD_CLOCK.get(normalised)
    if clock is None:
        raise ValueError(
            f"unknown bar column {field!r}: state when it becomes knowable rather than "
            f"letting it default. Known: {sorted(_BAR_FIELD_CLOCK) + sorted(_PRIOR_SESSION_FIELDS)}"
        )
    moment = exchange_moment(session, clock)
    return EventTime(event_time=moment, available_time=moment)


# ---------------------------------------------------------------------------
# The look-ahead guard
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DecisionFrontier:
    """The instant a strategy is deciding at. Facts after it do not exist yet.

    `admit` raises instead of returning a boolean on purpose: a guard whose result
    can be ignored by forgetting to read it is not a guard. Callers that genuinely
    want to filter use `available`, which is explicit about discarding.
    """

    decision_time: datetime

    @classmethod
    def at(cls, moment: datetime | str) -> "DecisionFrontier":
        return cls(decision_time=ensure_aware(moment, field="decision_time"))

    @classmethod
    def at_session_open(cls, session: date | str) -> "DecisionFrontier":
        """Deciding at the open: nothing from this session's bar is knowable yet."""
        return cls(decision_time=session_open(session))

    @classmethod
    def at_session_close(cls, session: date | str) -> "DecisionFrontier":
        return cls(decision_time=session_close(session))

    def admit(self, stamps: EventTime, *, description: str = "fact") -> EventTime:
        """Return `stamps` if it was knowable, otherwise raise."""
        if stamps.available_time > self.decision_time:
            raise LookAheadViolation(description, stamps.available_time, self.decision_time)
        return stamps

    def available(self, facts: Iterable[tuple[Any, EventTime]]) -> list[tuple[Any, EventTime]]:
        """Filter to what was knowable. Discards, and says so by name."""
        return [(item, stamps) for item, stamps in facts if stamps.available_time <= self.decision_time]

    def advanced_to(self, moment: datetime | str) -> "DecisionFrontier":
        """Move the frontier forward. It never moves back.

        A frontier that could retreat would let a replay re-decide a moment it had
        already passed, with information it had already seen.
        """
        target = ensure_aware(moment, field="moment")
        if target < self.decision_time:
            raise ValueError(
                f"the decision frontier cannot move back from "
                f"{self.decision_time.isoformat()} to {target.isoformat()}"
            )
        return DecisionFrontier(decision_time=target)


# ---------------------------------------------------------------------------
# Deterministic ordering
# ---------------------------------------------------------------------------
#: Priority for events sharing an instant, lowest first. The order encodes
#: causality: the market is observed, then risk decides, then an order is sent,
#: then the venue answers, then the session settles. Two events at the same
#: timestamp in the wrong order can make a fill precede the bar that caused it.
EVENT_PRIORITY: Mapping[str, int] = {
    "CALENDAR": 0,
    "SESSION_OPEN": 1,
    "CORPORATE_ACTION": 2,
    "SECURITY_STATE": 3,
    #: A halt is a tradability fact for one session, not a change in what the
    #: security is — so it follows SECURITY_STATE and precedes the price band,
    #: which is computed for a symbol that may not trade at all.
    "SUSPENSION": 4,
    "PRICE_LIMIT": 5,
    "QUOTE": 6,
    "TRADE": 7,
    "BAR": 8,
    "SIGNAL": 9,
    "TARGET": 10,
    "RISK_DECISION": 11,
    "ORDER": 12,
    "CANCEL": 13,
    "VENUE_CALLBACK": 14,
    "FILL": 15,
    "EXPIRY": 16,
    "MARK_TO_MARKET": 17,
    "SETTLEMENT": 18,
    "SESSION_CLOSE": 19,
    #: After the close: the day's accounting is final.
    "END_OF_DAY": 20,
}

#: An unknown kind sorts *after* every known one rather than sharing a bucket with
#: calendar events. A new event type that silently sorted first could preempt the
#: market data it is supposed to react to.
UNKNOWN_PRIORITY = max(EVENT_PRIORITY.values()) + 1


def ordering_key(
    stamps: EventTime, kind: str, *, sequence: int = 0, identity: str = ""
) -> tuple[Any, ...]:
    """A total order over events, deterministic for equal timestamps.

    Ties are broken by causal priority, then by the source's own sequence, then by
    a stable identity. The last term matters more than it looks: without it two
    events identical in every other field order by dict or set iteration, and a
    replay stops being reproducible in a way that only shows up as a flaky
    reconciliation.
    """
    return (
        stamps.event_time,
        EVENT_PRIORITY.get(kind.upper(), UNKNOWN_PRIORITY),
        int(sequence),
        str(identity),
    )


def in_order(
    events: Sequence[tuple[EventTime, str, int, str]]
) -> list[tuple[EventTime, str, int, str]]:
    """Sort events into their canonical processing order."""
    return sorted(
        events,
        key=lambda item: ordering_key(item[0], item[1], sequence=item[2], identity=item[3]),
    )


__all__ = [
    "AFTERNOON_OPEN",
    "DecisionFrontier",
    "EVENT_PRIORITY",
    "EXCHANGE_TZ",
    "EventTime",
    "ImpossibleAvailability",
    "LookAheadViolation",
    "MORNING_CLOSE",
    "MORNING_OPEN",
    "NaiveTimestamp",
    "OPENING_AUCTION_PRINT",
    "SESSION_CLOSE",
    "UNKNOWN_PRIORITY",
    "bar_close",
    "bar_field_availability",
    "ensure_aware",
    "exchange_moment",
    "in_order",
    "intraday_bar",
    "ordering_key",
    "session_close",
    "session_open",
]
