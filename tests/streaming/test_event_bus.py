"""M2-01/M2-02: the event taxonomy and the strictly ordered bus.

The properties worth testing are the ones that make a streaming backtest wrong
when they fail, and each is failable here: arrival order must not reach the
results, the past must not be rewritable, market data must not be publishable by
a consumer, and a resumed run must be provably the same run.
"""

from __future__ import annotations

from datetime import timedelta
import random

import pytest

from quantagent.domain.timeline import (
    EVENT_PRIORITY,
    EventTime,
    LookAheadViolation,
    UNKNOWN_PRIORITY,
    bar_field_availability,
    exchange_moment,
    session_close,
    session_open,
)
from quantagent.streaming.bus import (
    BusCheckpoint,
    DuplicateEvent,
    EventBus,
    GENESIS_DIGEST,
    LateArrival,
    MarketDataFromConsumer,
    replay,
)
from quantagent.streaming.events import (
    MARKET_KINDS,
    REACTION_KINDS,
    EventKind,
    MarketEvent,
)

SESSION = "2026-08-04"
NEXT_SESSION = "2026-08-05"
SYMBOL = "600000.SH"


def event(kind: EventKind, moment, *, sequence: int = 0, **payload) -> MarketEvent:
    return MarketEvent(
        kind=kind,
        times=EventTime.immediate(moment),
        symbol=SYMBOL,
        sequence=sequence,
        payload=payload,
    )


# -- taxonomy ----------------------------------------------------------------
def test_every_event_kind_the_programme_names_exists():
    """The mandate's list, checked by name rather than by count."""
    required = {
        "CALENDAR", "SESSION_OPEN", "SESSION_CLOSE", "QUOTE", "TRADE", "BAR",
        "CORPORATE_ACTION", "SECURITY_STATE", "SUSPENSION", "PRICE_LIMIT",
        "SIGNAL", "TARGET", "RISK_DECISION", "ORDER", "VENUE_CALLBACK", "FILL",
        "CANCEL", "EXPIRY", "SETTLEMENT", "MARK_TO_MARKET", "END_OF_DAY",
    }
    assert {kind.value for kind in EventKind} == required


def test_every_kind_has_a_declared_causal_priority():
    """A kind with no priority sorts last, silently, after what should react to it."""
    unplaced = [k.value for k in EventKind if k.value not in EVENT_PRIORITY]
    assert unplaced == []
    assert len({EVENT_PRIORITY[k.value] for k in EventKind}) == len(EventKind), (
        "two kinds share a priority, so their relative order is undefined"
    )


def test_market_and_reaction_kinds_partition_the_taxonomy():
    assert MARKET_KINDS | REACTION_KINDS == set(EventKind)
    assert MARKET_KINDS & REACTION_KINDS == set()
    assert EventKind.BAR in MARKET_KINDS
    assert EventKind.ORDER in REACTION_KINDS


def test_market_data_sorts_before_the_reactions_to_it():
    """A signal that preceded the data it read is the look-ahead in queue form."""
    for market in (EventKind.QUOTE, EventKind.TRADE, EventKind.BAR):
        for reaction in (EventKind.SIGNAL, EventKind.ORDER, EventKind.FILL):
            assert EVENT_PRIORITY[market.value] < EVENT_PRIORITY[reaction.value]


def test_a_suspension_is_ordered_between_security_state_and_the_price_band():
    assert (
        EVENT_PRIORITY["SECURITY_STATE"]
        < EVENT_PRIORITY["SUSPENSION"]
        < EVENT_PRIORITY["PRICE_LIMIT"]
    )


def test_an_event_id_is_content_addressed():
    """A replay must produce the same ids as the run it replays."""
    first = event(EventKind.BAR, session_close(SESSION), close=10.0)
    second = event(EventKind.BAR, session_close(SESSION), close=10.0)
    different = event(EventKind.BAR, session_close(SESSION), close=10.5)

    assert first.event_id == second.event_id
    assert first.event_id != different.event_id


def test_an_event_carries_the_whole_time_model_not_a_bare_instant():
    bar = MarketEvent(
        kind=EventKind.BAR, times=EventTime.bar(session=SESSION), symbol=SYMBOL
    )
    assert bar.times.available_time == session_close(SESSION)
    assert set(bar.to_dict()["times"]) >= {"eventTime", "availableTime"}


# -- ordering ----------------------------------------------------------------
def test_arrival_order_does_not_reach_the_results():
    """The property that makes a streaming run reproducible at all."""
    instant = session_close(SESSION)
    events = [
        event(EventKind.FILL, instant, sequence=1),
        event(EventKind.BAR, instant, sequence=2),
        event(EventKind.ORDER, instant, sequence=3),
        event(EventKind.QUOTE, instant, sequence=4),
        event(EventKind.SIGNAL, instant, sequence=5),
    ]

    digests = set()
    orders = set()
    for seed in range(8):
        shuffled = list(events)
        random.Random(seed).shuffle(shuffled)
        # Published in the shuffled order, deliberately not sorted first: sorting
        # here would test `sorted`, not the bus.
        bus = EventBus()
        bus.publish_all(shuffled)
        emitted = list(bus.drain())
        orders.add(tuple(e.kind.value for e in emitted))
        digests.add(bus.digest)

    assert len(digests) == 1, "arrival order changed the run"
    assert orders == {("QUOTE", "BAR", "SIGNAL", "ORDER", "FILL")}


def test_events_emit_in_time_order_across_instants():
    bus = EventBus()
    bus.publish(event(EventKind.BAR, session_close(SESSION)))
    bus.publish(event(EventKind.BAR, session_open(SESSION)))

    emitted = list(bus.drain())

    assert [e.times.event_time for e in emitted] == [
        session_open(SESSION), session_close(SESSION)
    ]


def test_draining_up_to_a_limit_leaves_the_rest_pending():
    bus = EventBus()
    bus.publish(event(EventKind.BAR, session_open(SESSION)))
    bus.publish(event(EventKind.BAR, session_close(SESSION)))
    bus.publish(event(EventKind.BAR, session_open(NEXT_SESSION)))

    emitted = list(bus.drain(until=session_close(SESSION)))

    assert len(emitted) == 2
    assert bus.pending == 1
    assert bus.peek().times.event_time == session_open(NEXT_SESSION)


# -- the past is not rewritable ----------------------------------------------
def test_an_event_before_the_last_emitted_one_is_refused():
    bus = EventBus()
    bus.publish(event(EventKind.BAR, session_close(SESSION)))
    list(bus.drain())

    with pytest.raises(LateArrival, match="already decided"):
        bus.publish(event(EventKind.BAR, session_open(SESSION)))


def test_a_late_arrival_is_neither_dropped_nor_reordered():
    """Both silent options are wrong; only refusing is honest."""
    bus = EventBus()
    bus.publish(event(EventKind.FILL, session_close(SESSION)))
    emitted_before = list(bus.drain())

    with pytest.raises(LateArrival):
        bus.publish(event(EventKind.QUOTE, session_close(SESSION)))

    assert bus.pending == 0, "the refused event must not sit in the queue"
    assert bus.emitted == len(emitted_before)


def test_a_reaction_at_the_same_instant_is_accepted():
    """Reacting to a bar by placing an order is the normal case, not a late arrival."""
    bus = EventBus()
    instant = session_close(SESSION)
    bus.publish(event(EventKind.BAR, instant))

    seen = []
    for emitted in bus.drain():
        seen.append(emitted.kind)
        if emitted.kind is EventKind.BAR:
            bus.publish(event(EventKind.SIGNAL, instant))
            bus.publish(event(EventKind.ORDER, instant))

    assert seen == [EventKind.BAR, EventKind.SIGNAL, EventKind.ORDER]


def test_a_consumer_cannot_publish_market_data():
    """Extending the tape you are reacting to invents the market you wanted."""
    bus = EventBus()
    bus.publish(event(EventKind.BAR, session_close(SESSION)))

    with pytest.raises(MarketDataFromConsumer, match="may not extend it"):
        for _ in bus.drain():
            bus.publish(event(EventKind.QUOTE, session_close(SESSION) + timedelta(minutes=1)))


def test_abandoning_a_drain_does_not_leave_the_bus_refusing_market_data():
    """The flag must track where execution is, not span the whole loop.

    A caller that breaks out of a drain has stopped consuming; the generator's own
    cleanup runs at collection time, which is not a moment this class can depend
    on. Leaving the flag set would make every later market-data publish fail with
    a message about consumers.
    """
    bus = EventBus()
    bus.publish(event(EventKind.BAR, session_open(SESSION)))
    bus.publish(event(EventKind.BAR, session_close(SESSION)))

    for _ in bus.drain():
        break

    bus.publish(event(EventKind.QUOTE, session_open(NEXT_SESSION)))
    assert bus.pending == 2


def test_market_data_may_be_published_before_the_drain_begins():
    bus = EventBus()
    bus.publish(event(EventKind.QUOTE, session_open(SESSION)))
    bus.publish(event(EventKind.BAR, session_close(SESSION)))
    assert len(list(bus.drain())) == 2


def test_a_redelivered_event_is_refused():
    bus = EventBus()
    bar = event(EventKind.BAR, session_close(SESSION), close=10.0)
    bus.publish(bar)

    with pytest.raises(DuplicateEvent, match="redelivery"):
        bus.publish(event(EventKind.BAR, session_close(SESSION), close=10.0))

    assert bus.pending == 1


# -- the frontier ------------------------------------------------------------
def test_the_frontier_advances_with_the_bus():
    bus = EventBus()
    bus.publish(event(EventKind.BAR, session_open(SESSION)))
    bus.publish(event(EventKind.BAR, session_close(SESSION)))

    stepped = []
    for _ in bus.drain():
        stepped.append(bus.frontier.decision_time)

    assert stepped == [session_open(SESSION), session_close(SESSION)]


def test_there_is_no_frontier_before_the_first_event():
    """Before anything is emitted there is no instant the run is deciding at."""
    with pytest.raises(RuntimeError, match="does not exist until"):
        EventBus().frontier


def test_a_consumer_reaching_for_an_unavailable_fact_is_refused():
    """The bus and the time model together: the look-ahead is an exception."""
    bus = EventBus()
    bus.publish(
        MarketEvent(
            kind=EventKind.SESSION_OPEN,
            times=EventTime.immediate(session_open(SESSION)),
            symbol=SYMBOL,
        )
    )

    for _ in bus.drain():
        with pytest.raises(LookAheadViolation) as caught:
            bus.frontier.admit(
                bar_field_availability("close", SESSION), description="today's close"
            )
        assert caught.value.gap == timedelta(hours=5, minutes=30)


def test_the_handler_is_given_the_frontier_rather_than_trusted_to_track_time():
    bus = EventBus()
    bus.publish(event(EventKind.BAR, session_open(SESSION)))
    bus.publish(event(EventKind.BAR, session_close(SESSION)))
    seen: list[tuple[str, str]] = []

    count = bus.run(
        lambda evt, frontier: seen.append(
            (evt.kind.value, frontier.decision_time.isoformat())
        )
    )

    assert count == 2
    assert [moment for _, moment in seen] == [
        session_open(SESSION).isoformat(), session_close(SESSION).isoformat()
    ]


# -- checkpoints -------------------------------------------------------------
def _three_sessions() -> list[MarketEvent]:
    return [
        event(EventKind.BAR, exchange_moment(session, session_close(SESSION).time()), sequence=i)
        for i, session in enumerate((SESSION, NEXT_SESSION, "2026-08-06"))
    ]


def test_an_interrupted_run_resumes_into_the_same_run():
    events = _three_sessions()

    uninterrupted = EventBus()
    uninterrupted.publish_all(events)
    list(uninterrupted.drain())

    first = EventBus()
    first.publish_all(events)
    consumed = list(first.drain(until=session_close(SESSION)))
    checkpoint = first.checkpoint()

    second = EventBus()
    second.resume_from(checkpoint)
    second.publish_all(events[len(consumed):])
    list(second.drain())

    assert second.digest == uninterrupted.digest, "the resumed run is a different run"
    assert second.emitted == uninterrupted.emitted


def test_a_checkpoint_carries_a_digest_not_just_a_position():
    """A position alone would happily resume a different stream of the same length."""
    bus = EventBus()
    bus.publish_all(_three_sessions())
    list(bus.drain(until=session_close(SESSION)))
    checkpoint = bus.checkpoint()

    assert checkpoint.emitted == 1
    assert checkpoint.digest != GENESIS_DIGEST
    assert checkpoint.last_event_id
    assert set(checkpoint.to_dict()) == {"emitted", "digest", "frontier", "lastEventId"}


def test_resuming_a_bus_that_has_already_emitted_is_refused():
    bus = EventBus()
    bus.publish_all(_three_sessions())
    list(bus.drain(until=session_close(SESSION)))

    with pytest.raises(RuntimeError, match="fresh bus"):
        bus.resume_from(bus.checkpoint())


def test_a_resumed_run_does_not_re_emit_what_it_already_processed():
    """Re-emitting would double whatever those events caused."""
    events = _three_sessions()
    first = EventBus()
    first.publish_all(events)
    consumed = list(first.drain(until=session_close(SESSION)))

    second = EventBus()
    second.resume_from(first.checkpoint())
    second.publish_all(events[len(consumed):])
    re_emitted = [e.event_id for e in second.drain()]

    assert consumed[0].event_id not in re_emitted


def test_two_different_streams_do_not_share_a_digest():
    """The check that makes the digest worth carrying."""
    _, first = replay(_three_sessions())
    altered = _three_sessions()[:-1] + [
        event(EventKind.BAR, session_close("2026-08-06"), sequence=99, close=11.0)
    ]
    _, second = replay(altered)

    assert first != second
