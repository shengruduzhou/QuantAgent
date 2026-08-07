"""M2-04/M2-05: the streaming lifecycle, and the same-bar ambiguity rule.

The lifecycle tests are mostly about what is *absent*: the streaming engine keeps
no order state, so the interesting assertions are that its figures come from a
replay and that it inherits Module One's refusals rather than restating them.

The ambiguity tests are about never silently taking the favourable branch — the
error that is invisible in any single trade and decisive over a backtest.
"""

from __future__ import annotations

from datetime import time

import pytest

from quantagent.domain.ledger import CanonicalLedger
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import (
    DuplicateExecution,
    IllegalTransition,
    OrderBook,
    OrderStatus,
    Side,
)
from quantagent.domain.timeline import EventTime, exchange_moment
from quantagent.reconciliation.snapshot import EconomicSnapshot
from quantagent.streaming.ambiguity import (
    AmbiguityPolicy,
    Bar,
    InvalidBracket,
    PathResolution,
    UNAMBIGUOUS,
    ambiguity_report,
    resolve_same_bar,
)
from quantagent.streaming.bus import EventBus
from quantagent.streaming.events import EventKind, MarketEvent
from quantagent.streaming.lifecycle import (
    MissingEventField,
    OrderLifecycle,
    UnroutableEvent,
)

SESSION = "2026-08-04"
NEXT_SESSION = "2026-08-05"
SYMBOL = "600000.SH"
INITIAL = 1_000_000.0
RUN = Lineage(research_id="r", strategy_version_id="sv", run_id="stream_run")


def evt(kind: EventKind, clock: time, *, session: str = SESSION, **payload) -> MarketEvent:
    return MarketEvent(
        kind=kind,
        times=EventTime.immediate(exchange_moment(session, clock)),
        symbol=SYMBOL,
        payload=payload,
    )


@pytest.fixture
def lifecycle(tmp_path) -> OrderLifecycle:
    return OrderLifecycle(
        ledger=CanonicalLedger(tmp_path / "stream.jsonl"),
        lineage=RUN,
        initial_cash=INITIAL,
    )


def _buy_and_fill(lifecycle: OrderLifecycle, *, quantity: int = 1_000) -> EventBus:
    bus = EventBus()
    bus.publish_all(
        [
            evt(EventKind.ORDER, time(9, 31), clientOrderId="c1", side="BUY",
                quantity=quantity, limitPrice=10.05),
            evt(EventKind.VENUE_CALLBACK, time(9, 32), clientOrderId="c1", status="ACCEPTED"),
            evt(EventKind.FILL, time(9, 33), clientOrderId="c1", executionId="X1",
                quantity=quantity, price=10.02, commission=5.0),
        ]
    )
    bus.run(lifecycle)
    return bus


# -- the lifecycle keeps no state of its own ---------------------------------
def test_the_streaming_engine_reports_figures_from_a_replay(lifecycle):
    _buy_and_fill(lifecycle)

    account = lifecycle.account()
    from_file = CanonicalLedger(lifecycle.ledger.path).replay(initial_cash=INITIAL)[1]

    assert account.content_hash() == from_file.content_hash()
    assert account.position(SYMBOL) == 1_000
    assert account.identity_residual({SYMBOL: 10.02}) == pytest.approx(0.0, abs=1e-9)


def test_no_economic_figure_on_the_lifecycle_evolves_as_events_are_handled(lifecycle):
    """The third parallel order model is the one that gets built by accident.

    Checked by what *changes* rather than by attribute names: `initial_cash` is an
    immutable replay input — the ledger records events, not an opening balance — so
    a name-based check would flag it and prove nothing. A cash or position field
    that moved when a fill arrived would be the real defect, and it would show up
    here as a changed value.
    """
    from dataclasses import fields

    # Declared fields, not `vars()`: an `init=False` field with a plain default
    # stays a class attribute until first assigned, so a `vars()` snapshot would
    # silently omit exactly the fields that only appear once they start changing.
    scalar_fields = [
        f.name for f in fields(lifecycle)
        if not isinstance(getattr(lifecycle, f.name), (dict, OrderBook, CanonicalLedger))
    ]
    before = {name: getattr(lifecycle, name) for name in scalar_fields}

    _buy_and_fill(lifecycle)

    changed = {name for name in scalar_fields if getattr(lifecycle, name) != before[name]}

    assert changed == {"handled"}, (
        f"the streaming engine mutated its own state while handling events: {changed}. "
        "Only the events-seen counter may move; anything economic must come from a replay."
    )
    assert lifecycle.handled == 3
    assert lifecycle.initial_cash == INITIAL, "the opening balance is an input, not state"
    # The only per-order things it keeps are translation tables, which carry no
    # quantity, price or balance.
    assert set(lifecycle._canonical_ids) == {"c1"}
    assert set(lifecycle._sessions) == {"c1"}
    assert lifecycle._sessions["c1"] == SESSION


def test_an_order_reaches_the_shared_ledger_with_its_intent(lifecycle):
    _buy_and_fill(lifecycle)

    records = CanonicalLedger(lifecycle.ledger.path).read()
    created = [r for r in records if r.event.event_type.value == "CREATED"]
    assert len(created) == 1
    assert created[0].intent is not None, "an order without a recorded intent"
    assert created[0].intent["quantity"] == 1_000


def test_the_full_lifecycle_sequence_is_recorded(lifecycle):
    _buy_and_fill(lifecycle)

    book = lifecycle.order_book()
    order = book.orders()[0]
    sequence = [e.event_type.value for e in book.history_of(order.order_id)]

    assert sequence == ["CREATED", "RISK_APPROVED", "SUBMITTED", "ACCEPTED", "FILL"]
    assert order.status is OrderStatus.FILLED
    assert order.leaves_quantity == 0


def test_a_partial_fill_leaves_the_order_working(lifecycle):
    bus = EventBus()
    bus.publish_all(
        [
            evt(EventKind.ORDER, time(9, 31), clientOrderId="c1", side="BUY",
                quantity=1_000, limitPrice=10.05),
            evt(EventKind.VENUE_CALLBACK, time(9, 32), clientOrderId="c1", status="ACCEPTED"),
            evt(EventKind.FILL, time(9, 33), clientOrderId="c1", executionId="X1",
                quantity=400, price=10.02, commission=5.0),
        ]
    )
    bus.run(lifecycle)

    assert lifecycle.status_of("c1") is OrderStatus.PARTIALLY_FILLED
    assert lifecycle.order_book().orders()[0].leaves_quantity == 600


def test_the_venue_does_not_decide_whether_a_fill_is_final(lifecycle):
    """A venue mislabelling a partial as final would close a working order."""
    bus = EventBus()
    bus.publish_all(
        [
            evt(EventKind.ORDER, time(9, 31), clientOrderId="c1", side="BUY",
                quantity=1_000, limitPrice=10.05),
            evt(EventKind.VENUE_CALLBACK, time(9, 32), clientOrderId="c1", status="ACCEPTED"),
            # The payload says nothing about finality; the remaining quantity decides.
            evt(EventKind.FILL, time(9, 33), clientOrderId="c1", executionId="X1",
                quantity=1_000, price=10.02, commission=5.0),
        ]
    )
    bus.run(lifecycle)

    assert lifecycle.status_of("c1") is OrderStatus.FILLED


# -- Module One's refusals are inherited, not restated -----------------------
def test_a_redelivered_execution_moves_no_money(lifecycle):
    bus = _buy_and_fill(lifecycle)
    before = lifecycle.account()

    bus.publish(
        evt(EventKind.FILL, time(9, 40), clientOrderId="c1", executionId="X1",
            quantity=1_000, price=10.02, commission=5.0)
    )
    bus.run(lifecycle)

    assert lifecycle.account().content_hash() == before.content_hash()


def test_reusing_an_execution_id_with_different_economics_is_refused(lifecycle):
    bus = _buy_and_fill(lifecycle)

    bus.publish(
        evt(EventKind.FILL, time(9, 41), clientOrderId="c1", executionId="X1",
            quantity=900, price=10.02, commission=5.0)
    )
    with pytest.raises(DuplicateExecution):
        bus.run(lifecycle)


def test_a_fill_after_a_cancel_is_refused(lifecycle):
    bus = EventBus()
    bus.publish_all(
        [
            evt(EventKind.ORDER, time(9, 31), clientOrderId="c1", side="BUY",
                quantity=1_000, limitPrice=10.05),
            evt(EventKind.VENUE_CALLBACK, time(9, 32), clientOrderId="c1", status="ACCEPTED"),
            evt(EventKind.CANCEL, time(9, 33), clientOrderId="c1"),
        ]
    )
    bus.run(lifecycle)
    assert lifecycle.status_of("c1") is OrderStatus.CANCELLED

    bus.publish(
        evt(EventKind.FILL, time(9, 34), clientOrderId="c1", executionId="X9",
            quantity=100, price=10.0)
    )
    with pytest.raises(IllegalTransition):
        bus.run(lifecycle)


def test_a_cancel_that_lost_a_race_to_a_fill_is_absorbed(lifecycle):
    """Normal, and not the same as absorbing a duplicate fill."""
    bus = _buy_and_fill(lifecycle)
    before = lifecycle.account()

    bus.publish(evt(EventKind.CANCEL, time(9, 45), clientOrderId="c1"))
    bus.run(lifecycle)

    assert lifecycle.status_of("c1") is OrderStatus.FILLED
    assert lifecycle.account().content_hash() == before.content_hash()


def test_a_rejection_after_a_fill_is_refused(lifecycle):
    """A rejection must not erase quantity that really traded."""
    bus = _buy_and_fill(lifecycle)

    bus.publish(
        evt(EventKind.VENUE_CALLBACK, time(9, 46), clientOrderId="c1",
            status="REJECTED", reason="too late")
    )
    with pytest.raises(IllegalTransition):
        bus.run(lifecycle)


def test_a_settlement_date_comes_from_the_fill_not_a_later_event(lifecycle):
    """DEF-016, inherited: a cancel must not re-date an earlier fill."""
    bus = EventBus()
    bus.publish_all(
        [
            evt(EventKind.ORDER, time(9, 31), clientOrderId="c1", side="BUY",
                quantity=1_000, limitPrice=10.05),
            evt(EventKind.VENUE_CALLBACK, time(9, 32), clientOrderId="c1", status="ACCEPTED"),
            evt(EventKind.FILL, time(9, 33), clientOrderId="c1", executionId="X1",
                quantity=400, price=10.02, commission=5.0),
            evt(EventKind.CANCEL, time(9, 34), clientOrderId="c1", session=NEXT_SESSION),
        ]
    )
    bus.run(lifecycle)

    account = lifecycle.account()
    assert account.lots[SYMBOL][0].acquired_on == SESSION
    assert account.sellable(SYMBOL, NEXT_SESSION) == 400


# -- routing failures fail closed --------------------------------------------
def test_a_callback_for_an_order_this_run_never_opened_is_refused(lifecycle):
    bus = EventBus()
    bus.publish(evt(EventKind.VENUE_CALLBACK, time(9, 32), clientOrderId="ghost",
                    status="ACCEPTED"))
    with pytest.raises(UnroutableEvent, match="never opened"):
        bus.run(lifecycle)


def test_a_missing_required_field_is_refused_rather_than_defaulted(lifecycle):
    bus = EventBus()
    bus.publish(evt(EventKind.ORDER, time(9, 31), clientOrderId="c1", side="BUY"))
    with pytest.raises(MissingEventField, match="quantity"):
        bus.run(lifecycle)


def test_an_unknown_venue_reply_is_not_treated_as_an_acceptance(lifecycle):
    """"Unknown" must not become "accepted", or a refused order starts trading."""
    bus = EventBus()
    bus.publish_all(
        [
            evt(EventKind.ORDER, time(9, 31), clientOrderId="c1", side="BUY",
                quantity=1_000, limitPrice=10.05),
            evt(EventKind.VENUE_CALLBACK, time(9, 32), clientOrderId="c1", status="MAYBE"),
        ]
    )
    with pytest.raises(MissingEventField, match="no canonical"):
        bus.run(lifecycle)


# -- the streaming lifecycle and paper agree ---------------------------------
def test_streaming_and_paper_agree_on_the_same_economic_events(tmp_path):
    """The claim behind "shared lifecycle": same events, same figures.

    Both sides are compared through `EconomicSnapshot`, which reads only the
    canonical replay — so this compares records of account, not two engines'
    opinions of themselves.
    """
    from quantagent.paper import ledger as paper_ledger
    from quantagent.paper.broker import BrokerConfig, MarketSnapshot, PaperBroker
    from quantagent.paper.orders import BUY, Order as PaperOrder
    from quantagent.paper.portfolio import Portfolio

    market = MarketSnapshot(
        symbol=SYMBOL, trade_date=SESSION, last_price=10.00, previous_close=10.00,
        session_volume=1e8, board="SH_Main",
    )
    paper = PaperBroker(
        Portfolio(portfolio_id="p", cash=INITIAL, initial_cash=INITIAL),
        paper_ledger.EventLedger(tmp_path / "op.jsonl"),
        run_id="paper_run",
        config=BrokerConfig(slippage_bps=0.0, impact_coefficient=0.0),
        canonical_ledger_path=str(tmp_path / "paper.jsonl"),
        lineage=Lineage(research_id="r", strategy_version_id="sv", run_id="paper_run"),
    )
    order = PaperOrder(symbol=SYMBOL, side=BUY, quantity=1_000.0, limit_price=10.05)
    paper.submit(order, market)
    paper_fill = paper.fills[0]

    streaming = OrderLifecycle(
        ledger=CanonicalLedger(tmp_path / "stream.jsonl"), lineage=RUN, initial_cash=INITIAL
    )
    bus = EventBus()
    bus.publish_all(
        [
            evt(EventKind.ORDER, time(10, 0), clientOrderId="c1", side="BUY",
                quantity=1_000, limitPrice=10.05),
            evt(EventKind.VENUE_CALLBACK, time(10, 0, 1), clientOrderId="c1",
                status="ACCEPTED"),
            evt(EventKind.FILL, time(10, 0, 2), clientOrderId="c1", executionId="X1",
                quantity=1_000, price=paper_fill.price,
                commission=paper_fill.commission, stampDuty=paper_fill.stamp_duty,
                transferFee=paper_fill.transfer_fee),
        ]
    )
    bus.run(streaming)

    prices = {SYMBOL: 10.00}
    paper_book, paper_account = CanonicalLedger(tmp_path / "paper.jsonl").replay(
        initial_cash=INITIAL
    )
    stream_book, stream_account = CanonicalLedger(tmp_path / "stream.jsonl").replay(
        initial_cash=INITIAL
    )
    left = EconomicSnapshot.from_replay(
        "paper", paper_book, paper_account, session=SESSION, prices=prices
    )
    right = EconomicSnapshot.from_replay(
        "streaming", stream_book, stream_account, session=SESSION, prices=prices
    )

    assert left.cash == right.cash
    assert left.positions == right.positions
    assert left.fees_total == right.fees_total
    assert left.realised_pnl == right.realised_pnl
    assert left.lots == right.lots
    assert left.identity_residual == pytest.approx(right.identity_residual, abs=1e-9)
    facts = next(iter(left.orders.values())), next(iter(right.orders.values()))
    assert facts[0].event_sequence == facts[1].event_sequence
    assert facts[0].status == facts[1].status


# -- M2-05: same-bar ambiguity ----------------------------------------------
LONG = Side.BUY
SHORT = Side.SELL


def test_no_policy_resolves_ambiguity_favourably():
    """The structural guarantee: there is no favourable branch to select."""
    assert {p.value for p in AmbiguityPolicy} == {"CONSERVATIVE", "MARK_AMBIGUOUS"}
    assert not any(
        token in p.value.lower()
        for p in AmbiguityPolicy
        for token in ("favour", "favor", "optimistic", "target_first", "best")
    )


def test_a_bar_reaching_only_the_target_is_unambiguous():
    outcome = resolve_same_bar(
        Bar(open=10.0, high=10.6, low=9.9, close=10.5), side=LONG, stop=9.5, target=10.5
    )
    assert outcome.resolution is PathResolution.TARGET_ONLY
    assert outcome.ambiguous is False
    assert outcome.resolution in UNAMBIGUOUS
    assert outcome.trigger_price == 10.5


def test_a_bar_reaching_only_the_stop_is_unambiguous():
    outcome = resolve_same_bar(
        Bar(open=10.0, high=10.1, low=9.4, close=9.6), side=LONG, stop=9.5, target=10.5
    )
    assert outcome.resolution is PathResolution.STOP_ONLY
    assert outcome.ambiguous is False
    assert outcome.trigger_price == 9.5


def test_a_bar_reaching_neither_resolves_to_neither():
    outcome = resolve_same_bar(
        Bar(open=10.0, high=10.2, low=9.8, close=10.1), side=LONG, stop=9.5, target=10.5
    )
    assert outcome.resolution is PathResolution.NEITHER
    assert outcome.triggered is None


def test_a_bar_touching_both_resolves_against_the_position_by_default():
    """The whole point: the default assumption is adverse, and says so."""
    outcome = resolve_same_bar(
        Bar(open=10.0, high=10.6, low=9.4, close=10.2), side=LONG, stop=9.5, target=10.5
    )
    assert outcome.resolution is PathResolution.AMBIGUOUS_RESOLVED_CONSERVATIVELY
    assert outcome.triggered == "stop"
    assert outcome.ambiguous is True
    assert outcome.is_assumption is True
    assert "assumed" in outcome.rule


def test_a_short_bracket_resolves_against_the_short():
    """Adverse means adverse *to the position*, not "lower price"."""
    outcome = resolve_same_bar(
        Bar(open=10.0, high=10.6, low=9.4, close=10.2), side=SHORT, stop=10.5, target=9.5
    )
    assert outcome.resolution is PathResolution.AMBIGUOUS_RESOLVED_CONSERVATIVELY
    assert outcome.triggered == "stop"
    assert outcome.trigger_price == 10.5


def test_marking_ambiguous_resolves_nothing_and_says_so():
    outcome = resolve_same_bar(
        Bar(open=10.0, high=10.6, low=9.4, close=10.2), side=LONG, stop=9.5, target=10.5,
        policy=AmbiguityPolicy.MARK_AMBIGUOUS,
    )
    assert outcome.resolution is PathResolution.AMBIGUOUS_UNRESOLVED
    assert outcome.triggered is None
    assert outcome.trigger_price is None
    assert outcome.ambiguous is True
    assert outcome.is_assumption is False


def test_intrabar_data_settles_the_question_as_a_measurement():
    """The only resolution that is not an assumption, and it is labelled apart."""
    bar = Bar(open=10.0, high=10.6, low=9.4, close=10.2)
    rose_first = resolve_same_bar(
        bar, side=LONG, stop=9.5, target=10.5, intrabar=[10.0, 10.3, 10.6, 9.4]
    )
    fell_first = resolve_same_bar(
        bar, side=LONG, stop=9.5, target=10.5, intrabar=[10.0, 9.7, 9.4, 10.6]
    )

    assert rose_first.resolution is PathResolution.AMBIGUOUS_RESOLVED_BY_INTRABAR
    assert rose_first.triggered == "target"
    assert rose_first.is_assumption is False
    assert fell_first.triggered == "stop"
    assert fell_first.resolution is PathResolution.AMBIGUOUS_RESOLVED_BY_INTRABAR


def test_a_gap_through_a_level_fills_at_the_open_not_the_level():
    """Filling at the level credits a price the market never offered."""
    outcome = resolve_same_bar(
        Bar(open=9.0, high=9.2, low=8.8, close=9.1), side=LONG, stop=9.5, target=10.5
    )
    assert outcome.resolution is PathResolution.STOP_ONLY
    assert outcome.trigger_price == 9.0, "the first available price was the open"


def test_an_inverted_bracket_is_refused():
    with pytest.raises(InvalidBracket, match="long bracket needs stop < target"):
        resolve_same_bar(
            Bar(open=10.0, high=10.6, low=9.4, close=10.2),
            side=LONG, stop=10.5, target=9.5,
        )
    with pytest.raises(InvalidBracket, match="short bracket needs stop > target"):
        resolve_same_bar(
            Bar(open=10.0, high=10.6, low=9.4, close=10.2),
            side=SHORT, stop=9.5, target=10.5,
        )


def test_an_inconsistent_bar_is_refused():
    """An inconsistent bar can be made to resolve either way."""
    with pytest.raises(ValueError, match="exceeds high"):
        Bar(open=10.0, high=9.0, low=10.5, close=10.0)
    with pytest.raises(ValueError, match="lies outside"):
        Bar(open=12.0, high=10.6, low=9.4, close=10.2)


def test_the_report_separates_assumptions_from_measurements():
    """"3% ambiguous" and "3% priced by assumption" are different confidences."""
    bar = Bar(open=10.0, high=10.6, low=9.4, close=10.2)
    outcomes = [
        resolve_same_bar(Bar(open=10.0, high=10.6, low=9.9, close=10.5),
                         side=LONG, stop=9.5, target=10.5),
        resolve_same_bar(bar, side=LONG, stop=9.5, target=10.5),
        resolve_same_bar(bar, side=LONG, stop=9.5, target=10.5,
                         intrabar=[10.0, 10.6, 9.4]),
        resolve_same_bar(bar, side=LONG, stop=9.5, target=10.5,
                         policy=AmbiguityPolicy.MARK_AMBIGUOUS),
    ]

    report = ambiguity_report(outcomes)

    assert report["total"] == 4
    assert report["ambiguous"] == 3
    assert report["resolvedByAssumption"] == 1
    assert report["resolvedByIntrabar"] == 1
    assert report["leftUnresolved"] == 1
    assert report["byResolution"]["TARGET_ONLY"] == 1
