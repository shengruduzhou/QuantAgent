"""M2-03: the explicit time model, and the look-ahead it makes detectable.

The point of separating `event_time` from `available_time` is that a single
timestamp cannot express the one fact that matters: a daily bar's close happens and
becomes knowable at 15:00, so a strategy deciding at 09:30 that reads it is reading
the future. Every test here is ultimately about that distinction being enforced
rather than described.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from quantagent.domain.timeline import (
    DecisionFrontier,
    EVENT_PRIORITY,
    EXCHANGE_TZ,
    EventTime,
    ImpossibleAvailability,
    LookAheadViolation,
    NaiveTimestamp,
    UNKNOWN_PRIORITY,
    bar_close,
    ensure_aware,
    exchange_moment,
    in_order,
    intraday_bar,
    ordering_key,
    session_close,
    session_open,
)

SESSION = "2026-08-04"
NEXT_SESSION = "2026-08-05"


# -- timezone discipline -----------------------------------------------------
def test_a_naive_timestamp_is_refused():
    """Two naive timestamps from different zones compare wrongly and silently."""
    with pytest.raises(NaiveTimestamp, match="no timezone"):
        ensure_aware(datetime(2026, 8, 4, 9, 30))
    with pytest.raises(NaiveTimestamp):
        EventTime.immediate("2026-08-04T09:30:00")


def test_an_aware_timestamp_is_accepted_in_either_form():
    as_string = ensure_aware("2026-08-04T09:30:00+08:00")
    as_object = ensure_aware(datetime(2026, 8, 4, 9, 30, tzinfo=EXCHANGE_TZ))
    assert as_string == as_object


def test_session_boundaries_are_exchange_local_not_utc():
    """Deriving availability in UTC puts the boundary in the wrong place."""
    assert session_open(SESSION).utcoffset() == timedelta(hours=8)
    assert session_close(SESSION).hour == 15
    assert session_close(SESSION) > session_open(SESSION)


# -- the core distinction ----------------------------------------------------
def test_a_daily_bar_becomes_knowable_at_the_close_not_the_open():
    stamps = EventTime.bar(session=SESSION)

    assert stamps.available_time == session_close(SESSION)
    assert stamps.available_at(session_close(SESSION))
    assert not stamps.available_at(session_open(SESSION))


def test_a_strategy_deciding_at_the_open_cannot_read_that_sessions_bar():
    """The defect this whole module exists to make impossible."""
    frontier = DecisionFrontier.at_session_open(SESSION)
    todays_bar = EventTime.bar(session=SESSION)

    with pytest.raises(LookAheadViolation) as caught:
        frontier.admit(todays_bar, description=f"daily bar for {SESSION}")

    assert caught.value.gap == timedelta(hours=5, minutes=30)
    assert "daily bar" in str(caught.value)


def test_a_strategy_deciding_at_the_open_may_read_the_previous_sessions_bar():
    """The guard must not be so strict that nothing is usable."""
    frontier = DecisionFrontier.at_session_open(NEXT_SESSION)
    yesterdays_bar = EventTime.bar(session=SESSION)

    assert frontier.admit(yesterdays_bar) is yesterdays_bar


def test_a_fact_cannot_claim_to_be_knowable_before_it_happened():
    with pytest.raises(ImpossibleAvailability, match="precedes event_time"):
        EventTime(
            event_time=session_close(SESSION),
            available_time=session_open(SESSION),
        )


def test_an_immediate_fact_is_knowable_the_instant_it_occurs():
    moment = exchange_moment(SESSION, session_open(SESSION).time())
    stamps = EventTime.immediate(moment)

    assert stamps.availability_lag == timedelta(0)
    assert stamps.available_at(moment)


def test_an_intraday_bar_is_knowable_when_its_window_closes():
    stamps = intraday_bar("2026-08-04T09:30:00+08:00", minutes=5)

    assert stamps.available_time == ensure_aware("2026-08-04T09:35:00+08:00")
    assert not stamps.available_at("2026-08-04T09:34:59+08:00")


def test_an_intraday_interval_cannot_be_resolved_from_a_session_alone():
    """A plausible wrong number is worse than a refusal."""
    with pytest.raises(ValueError, match="needs the bar's own window"):
        bar_close(SESSION, interval="5m")


def test_a_slow_pipeline_does_not_change_what_was_knowable():
    """Ingestion and processing are operational. They must not move availability."""
    prompt = EventTime.bar(session=SESSION, ingestion_time=session_close(SESSION))
    slow = EventTime.bar(
        session=SESSION, ingestion_time=exchange_moment(NEXT_SESSION, session_open(SESSION).time())
    )

    assert prompt.available_time == slow.available_time
    frontier = DecisionFrontier.at_session_close(SESSION)
    assert frontier.admit(prompt) and frontier.admit(slow)


# -- the guard's shape -------------------------------------------------------
def test_the_guard_raises_rather_than_returning_a_boolean():
    """A result a caller can ignore by forgetting to read it is not a guard."""
    frontier = DecisionFrontier.at_session_open(SESSION)
    with pytest.raises(LookAheadViolation):
        frontier.admit(EventTime.bar(session=SESSION))


def test_filtering_is_available_but_has_to_be_asked_for_by_name():
    frontier = DecisionFrontier.at_session_open(NEXT_SESSION)
    facts = [
        ("yesterday", EventTime.bar(session=SESSION)),
        ("today", EventTime.bar(session=NEXT_SESSION)),
    ]

    kept = frontier.available(facts)

    assert [name for name, _ in kept] == ["yesterday"]


def test_the_frontier_cannot_move_backwards():
    """A retreating frontier lets a replay re-decide with what it already saw."""
    frontier = DecisionFrontier.at_session_close(SESSION)
    assert frontier.advanced_to(session_open(NEXT_SESSION)).decision_time == session_open(
        NEXT_SESSION
    )
    with pytest.raises(ValueError, match="cannot move back"):
        frontier.advanced_to(session_open(SESSION))


def test_a_fact_available_exactly_at_the_frontier_is_admitted():
    """The boundary is inclusive: knowable *at* the decision instant counts."""
    frontier = DecisionFrontier.at_session_close(SESSION)
    assert frontier.admit(EventTime.bar(session=SESSION))


# -- order lifecycle stamps --------------------------------------------------
def test_an_order_carries_decision_submission_venue_and_fill_times():
    decided = exchange_moment(SESSION, session_open(SESSION).time())
    stamps = (
        EventTime.immediate(decided)
        .with_decision(decided)
        .with_submission(decided + timedelta(seconds=1))
        .with_fill(decided + timedelta(seconds=3), venue_receive_time=decided + timedelta(seconds=2))
    )

    assert stamps.decision_time == decided
    assert stamps.submission_time == decided + timedelta(seconds=1)
    assert stamps.venue_receive_time == decided + timedelta(seconds=2)
    assert stamps.fill_time == decided + timedelta(seconds=3)
    assert stamps.to_dict()["fillTime"].endswith("+08:00")


def test_every_stamp_survives_serialisation():
    stamps = EventTime.bar(session=SESSION, source_time=session_close(SESSION))
    payload = stamps.to_dict()

    assert set(payload) == {
        "eventTime", "availableTime", "sourceTime", "ingestionTime", "processingTime",
        "decisionTime", "submissionTime", "venueReceiveTime", "fillTime",
        "availabilityLagSeconds",
    }
    assert payload["availabilityLagSeconds"] == 0.0


# -- deterministic ordering --------------------------------------------------
def test_events_at_one_instant_order_by_causality():
    """A fill must not precede the bar that caused it."""
    instant = session_close(SESSION)
    stamps = EventTime.immediate(instant)
    shuffled = [
        (stamps, "FILL", 0, "f1"),
        (stamps, "BAR", 0, "b1"),
        (stamps, "ORDER", 0, "o1"),
        (stamps, "RISK_DECISION", 0, "r1"),
        (stamps, "QUOTE", 0, "q1"),
    ]

    assert [kind for _, kind, _, _ in in_order(shuffled)] == [
        "QUOTE", "BAR", "RISK_DECISION", "ORDER", "FILL",
    ]


def test_an_unknown_event_kind_sorts_after_every_known_one():
    """A new type that silently sorted first would preempt its own inputs."""
    instant = session_close(SESSION)
    stamps = EventTime.immediate(instant)
    assert ordering_key(stamps, "SOMETHING_NEW")[1] == UNKNOWN_PRIORITY
    assert UNKNOWN_PRIORITY > max(EVENT_PRIORITY.values())

    ordered = in_order([(stamps, "SOMETHING_NEW", 0, "x"), (stamps, "CALENDAR", 0, "c")])
    assert [kind for _, kind, _, _ in ordered] == ["CALENDAR", "SOMETHING_NEW"]


def test_identical_events_order_reproducibly():
    """Without a stable final term, equal events order by set iteration."""
    instant = session_close(SESSION)
    stamps = EventTime.immediate(instant)
    events = [(stamps, "BAR", 0, identity) for identity in ("c", "a", "b")]

    runs = {tuple(identity for _, _, _, identity in in_order(events)) for _ in range(5)}

    assert runs == {("a", "b", "c")}


def test_a_sources_own_sequence_breaks_ties_before_identity():
    instant = session_close(SESSION)
    stamps = EventTime.immediate(instant)
    events = [(stamps, "TRADE", 2, "z"), (stamps, "TRADE", 1, "a")]

    assert [seq for _, _, seq, _ in in_order(events)] == [1, 2]


def test_event_time_orders_before_any_tie_break():
    early = EventTime.immediate(session_open(SESSION))
    late = EventTime.immediate(session_close(SESSION))
    # A FILL at the open must still precede a QUOTE at the close, even though
    # QUOTE outranks FILL on the tie-break.
    ordered = in_order([(late, "QUOTE", 0, "q"), (early, "FILL", 0, "f")])

    assert [kind for _, kind, _, _ in ordered] == ["FILL", "QUOTE"]


# -- a bar row is not one fact ----------------------------------------------
def test_a_strategy_deciding_at_the_open_may_read_pre_close_and_open_only():
    """The panel column is right there in the same row; the row's shape says nothing.

    This is the concrete form the look-ahead takes in this repository: a daily
    panel is one row per (symbol, session), so `close` is as easy to reach as
    `pre_close` from a decision made at the open.
    """
    from quantagent.domain.timeline import bar_field_availability

    frontier = DecisionFrontier.at_session_open(SESSION)

    for readable in ("pre_close", "open"):
        assert frontier.admit(bar_field_availability(readable, SESSION))

    for unknowable in ("close", "high", "low", "volume", "amount"):
        with pytest.raises(LookAheadViolation) as caught:
            frontier.admit(
                bar_field_availability(unknowable, SESSION), description=unknowable
            )
        assert caught.value.description == unknowable
        assert caught.value.gap == timedelta(hours=5, minutes=30)


def test_every_bar_column_is_readable_after_the_close():
    from quantagent.domain.timeline import bar_field_availability

    frontier = DecisionFrontier.at_session_close(SESSION)
    for field in ("pre_close", "open", "high", "low", "close", "volume", "amount"):
        assert frontier.admit(bar_field_availability(field, SESSION))


def test_the_open_is_knowable_from_the_opening_auction_print():
    from quantagent.domain.timeline import OPENING_AUCTION_PRINT, bar_field_availability

    stamps = bar_field_availability("open", SESSION)
    assert stamps.available_time == exchange_moment(SESSION, OPENING_AUCTION_PRINT)
    assert stamps.available_time < session_open(SESSION), (
        "the auction prints at 09:25, before continuous trading opens"
    )


def test_an_unknown_bar_column_is_refused_rather_than_defaulted():
    """A default would be safe by accident for most columns and wrong for the rest."""
    from quantagent.domain.timeline import bar_field_availability

    with pytest.raises(ValueError, match="unknown bar column"):
        bar_field_availability("turnover_rate", SESSION)


def test_yesterdays_close_is_readable_at_todays_open():
    """The correct basis for a price-limit calculation, and it must not be blocked."""
    from quantagent.domain.timeline import bar_field_availability

    frontier = DecisionFrontier.at_session_open(NEXT_SESSION)
    assert frontier.admit(bar_field_availability("pre_close", NEXT_SESSION))
    assert frontier.admit(bar_field_availability("close", SESSION))
