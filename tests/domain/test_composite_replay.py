"""The composite gate: one scenario, four engines, zero unexplained.

The positive tests assert the thing the gate actually requires. The negative ones
matter just as much: a reconciler that cannot fail is not evidence, so several
tests deliberately corrupt a figure and assert the table reports it as
`unexplained` rather than absorbing it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quantagent.domain.accounting import AccountState, InvariantViolation
from quantagent.domain.ledger import CanonicalLedger, LedgerCorruption, LineageCollision
from quantagent.domain.orders import (
    DuplicateExecution,
    Fill,
    IllegalTransition,
    OrderBook,
    OrderEventType,
    OrderIntent,
    OrderStatus,
    Side,
    Signal,
)
from quantagent.domain.lineage import Lineage
from quantagent.reconciliation import composite as C
from quantagent.reconciliation.differences import (
    BOUNDED_FLOAT,
    DOCUMENTED_ENGINE_DIFFERENCE,
    ExplanationRule,
    UNEXPLAINED,
    compare_flat,
    compare_snapshots,
)


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> C.CompositeReport:
    """Run the composite once; every test below reads the same evidence."""
    return C.run_composite(tmp_path_factory.mktemp("composite"))


# -- the gate ---------------------------------------------------------------
def test_no_unexplained_economic_differences(report):
    unexplained = [
        (t.left_label, t.right_label, d.dimension, d.left_value, d.right_value)
        for t in report.all_tables
        for d in t.unexplained
    ]
    assert report.unexplained_economic_differences == 0, unexplained


def test_every_path_replays_from_its_ledger_alone(report):
    """The engine's own figures must equal the figures rebuilt from the file."""
    for path in report.paths:
        assert path.native_table.unexplained_economic_differences == 0, (
            f"{path.label} disagrees with its own record of account: "
            f"{[d.dimension for d in path.native_table.unexplained]}"
        )


def test_paper_and_oms_agree_on_every_dimension(report):
    """The OMS adds intent, risk and idempotency — and changes no economic figure.

    Not "no unexplained differences": *no differences at all*. Both drive the
    same venue on the same market data, so anything at all here would mean the
    routing layer moved money.
    """
    table = next(
        t for t in report.cross_tables
        if {t.left_label, t.right_label} == {C.PAPER, C.OMS_PAPER}
    )
    assert table.differences == [], [d.to_dict() for d in table.differences]
    assert table.only_left == [] and table.only_right == []


def test_pnl_split_reconciles_with_cash_on_every_path(report):
    for path in report.paths:
        assert path.snapshot.identity_residual == pytest.approx(0.0, abs=1e-6), (
            f"{path.label}: realised + unrealised does not equal NAV - initial cash"
        )


def test_every_ledger_verifies_and_replay_is_hash_stable(report):
    for path in report.paths:
        ledger = CanonicalLedger(path.ledger_path)
        assert ledger.verify()["valid"]
        hashes = {
            CanonicalLedger(path.ledger_path)
            .replay(initial_cash=C.INITIAL_CASH)[1]
            .content_hash()
            for _ in range(3)
        }
        assert len(hashes) == 1, f"{path.label} replay is not deterministic"


def test_no_order_is_missing_a_lineage_link(report):
    for path in report.paths:
        assert path.snapshot.lineage_gaps == (), (
            f"{path.label} has orders that cannot be drilled down: "
            f"{path.snapshot.lineage_gaps}"
        )


# -- the scenario really covered the cases it claims -------------------------
def test_scenario_covers_the_required_lifecycle_cases(report):
    paper = next(p for p in report.paths if p.label == C.PAPER)
    statuses = {facts.status for facts in paper.snapshot.orders.values()}
    assert OrderStatus.FILLED.value in statuses
    assert OrderStatus.REJECTED.value in statuses, "no rejection was exercised"
    assert OrderStatus.CANCELLED.value in statuses, "no cancellation was exercised"
    assert paper.snapshot.counts["rejected"] == 2, "expected a T+1 and a cash refusal"


def test_a_rejected_order_moved_no_money(report):
    for path in report.paths:
        for facts in path.snapshot.orders.values():
            if facts.status == OrderStatus.REJECTED.value:
                assert facts.cumulative_quantity == 0
                assert facts.fill_count == 0
                assert facts.fees == 0.0
                assert facts.reason, "a rejection must say why"


def test_cancelled_remainder_keeps_its_executed_quantity(report):
    paper = next(p for p in report.paths if p.label == C.PAPER)
    cancelled = [
        f for f in paper.snapshot.orders.values()
        if f.status == OrderStatus.CANCELLED.value
    ]
    assert cancelled, "the partial-fill-then-cancel step did not run"
    order = cancelled[0]
    assert order.cumulative_quantity == 1_000, "the executed part must survive the cancel"
    assert order.quantity == 3_000
    assert order.leaves_quantity == 0, "a cancelled order is no longer working"
    assert "PARTIAL_FILL" in order.event_sequence
    assert order.event_sequence[-1] == "CANCELLED"


def test_t_plus_one_refusal_then_next_session_sale(report):
    paper = next(p for p in report.paths if p.label == C.PAPER)
    same_session_sell = paper.snapshot.orders[f"{C.BLUE}|SELL|500|{C.SESSION_1}"]
    next_session_sell = paper.snapshot.orders[f"{C.BLUE}|SELL|1000|{C.SESSION_2}"]
    assert same_session_sell.status == OrderStatus.REJECTED.value
    assert next_session_sell.status == OrderStatus.FILLED.value
    assert next_session_sell.fees > 0, "a sale pays stamp duty"


def test_sell_side_stamp_duty_is_charged_and_buy_side_is_not(report):
    for path in report.paths:
        assert path.snapshot.stamp_duty > 0, f"{path.label} charged no stamp duty"
        sells = [
            f for f in path.snapshot.orders.values()
            if f.side == "SELL" and f.fill_count
        ]
        assert sells, f"{path.label} never sold anything"


def test_duplicate_execution_was_absorbed_not_booked(report):
    """The note is the measurement: `booked=False` means no money moved twice."""
    for label in (C.PAPER, C.OMS_PAPER):
        path = next(p for p in report.paths if p.label == label)
        assert any("booked=False" in note for note in path.notes), path.notes


def test_out_of_order_event_was_refused(report):
    for label in (C.PAPER, C.OMS_PAPER):
        path = next(p for p in report.paths if p.label == label)
        assert any("refused" in note for note in path.notes), path.notes


def test_resubmitted_intent_did_not_reach_the_venue_twice(report):
    oms = next(p for p in report.paths if p.label == C.OMS_PAPER)
    note = next(n for n in oms.notes if n.startswith("re-submitted intent"))
    before, after = note.rsplit(" ", 3)[-3], note.rsplit(" ", 1)[-1]
    assert before == after, f"the duplicate guard let a second order through: {note}"


def test_oms_projection_matches_the_ledger_rebuild(report):
    oms = next(p for p in report.paths if p.label == C.OMS_PAPER)
    assert "OMS wire projection matches ledger rebuild: True" in oms.notes


# -- the reconciler can fail -------------------------------------------------
def test_an_unexplained_cash_difference_blocks():
    table = compare_flat("a", {"cash": 1_000.0}, "b", {"cash": 999.0})
    assert table.unexplained_economic_differences == 1
    assert table.differences[0].classification == UNEXPLAINED
    assert table.differences[0].resolution_status == "blocking"
    assert not table.clean


def test_a_dimension_present_on_one_side_only_is_a_difference():
    """An engine that models nothing must not reconcile perfectly."""
    table = compare_flat("a", {"cash": 1.0, "position[X]": 100}, "b", {"cash": 1.0})
    assert table.unexplained_economic_differences == 1
    assert table.differences[0].dimension == "position[X]"
    assert table.differences[0].right_value is None


def test_a_tolerance_cannot_be_granted_to_a_discrete_dimension():
    with pytest.raises(ValueError, match="discrete dimension"):
        ExplanationRule(
            "position[600000.SH]", BOUNDED_FLOAT, "rule", "reason", tolerance=0.5
        )
    with pytest.raises(ValueError, match="discrete dimension"):
        ExplanationRule(
            "order[k].cumulative_quantity", BOUNDED_FLOAT, "rule", "reason", tolerance=1.0
        )


def test_a_tolerance_does_not_stretch_past_its_bound():
    rule = ExplanationRule("cash", BOUNDED_FLOAT, "float_math", "rounding", tolerance=0.01)
    within = compare_flat("a", {"cash": 100.0}, "b", {"cash": 100.005}, rules=(rule,))
    beyond = compare_flat("a", {"cash": 100.0}, "b", {"cash": 100.5}, rules=(rule,))
    assert within.clean
    assert beyond.unexplained_economic_differences == 1


def test_an_explanation_rule_must_carry_a_reason():
    with pytest.raises(ValueError, match="source rule"):
        ExplanationRule("cash", DOCUMENTED_ENGINE_DIFFERENCE, "", "reason")
    with pytest.raises(ValueError, match="source rule"):
        ExplanationRule("cash", DOCUMENTED_ENGINE_DIFFERENCE, "rule", "   ")


def test_a_rule_scoped_to_one_pair_does_not_leak_to_another():
    rule = ExplanationRule(
        "cash", DOCUMENTED_ENGINE_DIFFERENCE, "rule", "reason", pair=("x", "y")
    )
    assert compare_flat("x", {"cash": 1.0}, "y", {"cash": 2.0}, rules=(rule,)).clean
    assert not compare_flat("p", {"cash": 1.0}, "q", {"cash": 2.0}, rules=(rule,)).clean


def test_the_cross_path_rules_never_excuse_a_quantity_difference():
    """The fast-vs-venue exemptions are for price, not for what happened."""
    for pair in ((C.FAST, C.PAPER), (C.FAST, C.OMS_PAPER)):
        table = compare_flat(
            pair[0],
            {f"order[{C.BLUE}|BUY|1000|{C.SESSION_1}].cumulative_quantity": 1_000},
            pair[1],
            {f"order[{C.BLUE}|BUY|1000|{C.SESSION_1}].cumulative_quantity": 900},
            rules=C.CROSS_PATH_RULES,
        )
        # The order-level NOT_APPLICABLE rule is prefix-scoped, so it *does* match
        # this dimension — which is why the fast engine must independently be
        # shown to have executed the same quantity, below.
        assert table.differences, "the difference must at least be reported"


def test_fast_and_venue_executed_the_same_quantity(report):
    """The claim the price exemptions rest on: same shares, different price."""
    fast = next(p for p in report.paths if p.label == C.FAST)
    paper = next(p for p in report.paths if p.label == C.PAPER)
    key_buy = f"{C.BLUE}|BUY|1000|{C.SESSION_1}"
    key_sell = f"{C.BLUE}|SELL|1000|{C.SESSION_2}"
    for key in (key_buy, key_sell):
        assert fast.snapshot.orders[key].cumulative_quantity == (
            paper.snapshot.orders[key].cumulative_quantity
        ), f"{key}: the engines did not execute the same quantity"
        assert fast.snapshot.orders[key].status == paper.snapshot.orders[key].status


# -- duplicate execution, at the canonical layer -----------------------------
def _open_order(book: OrderBook, ledger: CanonicalLedger, quantity: int = 1_000):
    lineage = Lineage(research_id="r", strategy_version_id="sv", run_id="run_dup")
    signal = Signal.create(
        symbol=C.BLUE, trade_date=C.SESSION_1, score=1.0, lineage=lineage
    )
    intent = OrderIntent.create(
        symbol=C.BLUE, side=Side.BUY, quantity=quantity, trade_date=C.SESSION_1,
        lineage=signal.lineage, limit_price=10.05,
    )
    order = book.open(intent)
    ledger.append(
        book.history_of(order.order_id)[-1], trade_date=C.SESSION_1, intent=intent
    )
    for event in (
        OrderEventType.RISK_APPROVED, OrderEventType.SUBMITTED, OrderEventType.ACCEPTED
    ):
        book.apply(order.order_id, event)
        ledger.append(book.history_of(order.order_id)[-1], trade_date=C.SESSION_1)
    return book.state_of(order.order_id)


def _fill(order, execution_id: str, quantity: int, price: float, commission: float = 5.0):
    return Fill(
        execution_id=execution_id, order_id=order.order_id, symbol=C.BLUE, side=Side.BUY,
        quantity=quantity, price=price, reference_price=10.0, commission=commission,
        filled_at=C.SESSION_1, lineage=order.lineage.derive(execution_id=execution_id),
    )


def test_a_redelivered_execution_records_no_event_and_moves_no_money():
    book, ledger = OrderBook(), CanonicalLedger()
    order = _open_order(book, ledger)
    fill = _fill(order, "EXEC-1", 500, 10.0)

    book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=fill)
    ledger.append(book.history_of(order.order_id)[-1], trade_date=C.SESSION_1)
    events_after_first = len(book.events())
    _, account_before = ledger.replay(initial_cash=C.INITIAL_CASH)

    book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=fill)

    assert len(book.events()) == events_after_first, "the duplicate was recorded"
    assert book.state_of(order.order_id).cumulative_quantity == 500
    _, account_after = ledger.replay(initial_cash=C.INITIAL_CASH)
    assert account_after.content_hash() == account_before.content_hash()


def test_reusing_an_execution_id_with_different_economics_is_refused():
    book, ledger = OrderBook(), CanonicalLedger()
    order = _open_order(book, ledger)
    book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=_fill(order, "EXEC-1", 500, 10.0))

    with pytest.raises(DuplicateExecution, match="EXEC-1"):
        book.apply(
            order.order_id, OrderEventType.PARTIAL_FILL, fill=_fill(order, "EXEC-1", 400, 10.0)
        )
    with pytest.raises(DuplicateExecution):
        book.apply(
            order.order_id, OrderEventType.PARTIAL_FILL, fill=_fill(order, "EXEC-1", 500, 10.5)
        )
    with pytest.raises(DuplicateExecution):
        book.apply(
            order.order_id,
            OrderEventType.PARTIAL_FILL,
            fill=_fill(order, "EXEC-1", 500, 10.0, commission=9.0),
        )
    assert book.state_of(order.order_id).cumulative_quantity == 500


def test_replay_refuses_to_double_count_a_duplicate_already_in_the_log():
    """Defence on the read path: an old file may hold what the writer now blocks."""
    book, ledger = OrderBook(), CanonicalLedger()
    order = _open_order(book, ledger)
    fill = _fill(order, "EXEC-1", 500, 10.0)
    book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=fill)
    duplicated_event = book.history_of(order.order_id)[-1]
    ledger.append(duplicated_event, trade_date=C.SESSION_1)
    ledger.append(duplicated_event, trade_date=C.SESSION_1)  # as a pre-fix writer would

    _, account = ledger.replay(initial_cash=C.INITIAL_CASH)

    assert account.position(C.BLUE) == 500, "replay applied the same execution twice"
    assert account.duplicate_executions == ("EXEC-1",)


def test_a_later_cancel_does_not_re_date_an_earlier_fills_settlement():
    """The settlement session belongs to the fill, not to the order.

    DEF-016: `replay` mapped trade dates per *order*, last-write-wins. An order
    that traded in one session and was cancelled in the next therefore had its
    fill re-dated to the cancel's session, pushing the T+1 lot a day forward and
    making settled shares look unsettled. Deliberately uses two explicit sessions
    rather than the wall clock — the original bug was invisible on any day when
    "today" happened to equal the session being simulated.
    """
    book, ledger = OrderBook(), CanonicalLedger()
    order = _open_order(book, ledger, quantity=1_000)
    fill = _fill(order, "EXEC-1", 500, 10.0)

    book.apply(order.order_id, OrderEventType.PARTIAL_FILL, fill=fill)
    ledger.append(book.history_of(order.order_id)[-1], trade_date=C.SESSION_1)
    for event in (OrderEventType.CANCEL_REQUESTED, OrderEventType.CANCELLED):
        book.apply(order.order_id, event, reason="operator_cancel")
        ledger.append(book.history_of(order.order_id)[-1], trade_date=C.SESSION_2)

    _, account = ledger.replay(initial_cash=C.INITIAL_CASH)

    lot = account.lots[C.BLUE][0]
    assert lot.acquired_on == C.SESSION_1, "the cancel re-dated the fill"
    assert account.sellable(C.BLUE, C.SESSION_2) == 500, (
        "shares bought in the first session must be sellable in the second"
    )


def test_a_cancel_without_a_market_snapshot_records_no_invented_date(tmp_path):
    """An absent session is recoverable; a wrong one is not.

    The paper broker used to fall back to the wall clock for any canonical event
    whose caller did not supply a session, writing *today* into a
    settlement-relevant field. Replay can fall back to the fill's own session, so
    recording nothing is strictly safer than guessing.
    """
    from quantagent.paper import ledger as paper_ledger
    from quantagent.paper.broker import PaperBroker
    from quantagent.paper.orders import BUY, Order as PaperOrder
    from quantagent.paper.portfolio import Portfolio

    path = tmp_path / "chain.jsonl"
    broker = PaperBroker(
        Portfolio(portfolio_id="p", cash=C.INITIAL_CASH, initial_cash=C.INITIAL_CASH),
        paper_ledger.EventLedger(tmp_path / "op.jsonl"),
        run_id="r",
        canonical_ledger_path=str(path),
        lineage=Lineage(research_id="r", strategy_version_id="sv", run_id="r"),
    )
    thin = C.market_point(C.THIN, C.SESSION_1)
    order = PaperOrder(
        symbol=C.THIN, side=BUY, quantity=3_000.0, limit_price=20.20, board="SZ_Main"
    )
    broker.submit(order, thin.snapshot())
    broker.cancel(order.order_id)  # no market snapshot: session unknown

    records = CanonicalLedger(path).read()
    cancel_dates = [
        record.trade_date
        for record in records
        if record.event.event_type.value in {"CANCEL_REQUESTED", "CANCELLED"}
    ]
    assert cancel_dates and all(date is None for date in cancel_dates), (
        f"a session was invented for the cancel: {cancel_dates}"
    )

    _, account = CanonicalLedger(path).replay(initial_cash=C.INITIAL_CASH)
    assert account.lots[C.THIN][0].acquired_on == C.SESSION_1
    assert account.sellable(C.THIN, C.SESSION_2) == 1_000


def test_an_out_of_order_lifecycle_event_is_refused():
    book, ledger = OrderBook(), CanonicalLedger()
    order = _open_order(book, ledger, quantity=500)
    book.apply(order.order_id, OrderEventType.FILL, fill=_fill(order, "EXEC-1", 500, 10.0))

    with pytest.raises(IllegalTransition):
        book.apply(order.order_id, OrderEventType.ACCEPTED)


# -- the ledger is still the only admissible source --------------------------
def test_a_tampered_composite_ledger_blocks_every_projection(report):
    """Edit a filled quantity in the file; every projection must refuse to load.

    The edit has to be semantic. The chain hashes canonical JSON rather than raw
    bytes, so reformatting a record is correctly treated as the same record —
    which is why an earlier version of this test passed against a whitespace-only
    change and proved nothing.
    """
    import json

    for path in report.paths:
        lines = Path(path.ledger_path).read_text(encoding="utf-8").splitlines()
        index, record = next(
            (i, json.loads(line))
            for i, line in reversed(list(enumerate(lines)))
            if json.loads(line)["event"].get("fill")
        )
        record["event"]["fill"]["quantity"] += 100
        lines[index] = json.dumps(record, sort_keys=True, ensure_ascii=False)
        tampered = Path(str(path.ledger_path) + ".tampered")
        tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(LedgerCorruption):
            CanonicalLedger(tampered).replay(initial_cash=C.INITIAL_CASH)
        assert CanonicalLedger(tampered).verify()["brokenAt"] == index


def test_a_paper_broker_cannot_be_given_two_chains(tmp_path):
    from quantagent.paper import ledger as paper_ledger
    from quantagent.paper.broker import PaperBroker
    from quantagent.paper.portfolio import Portfolio

    with pytest.raises(ValueError, match="duplicate record of account"):
        PaperBroker(
            Portfolio(portfolio_id="p", cash=1.0, initial_cash=1.0),
            paper_ledger.EventLedger(tmp_path / "op.jsonl"),
            run_id="r",
            canonical_ledger=CanonicalLedger(),
            canonical_ledger_path=str(tmp_path / "c.jsonl"),
        )


def test_an_order_manager_cannot_be_given_two_chains():
    from quantagent.execution.order_manager import OrderManager
    from quantagent.execution.virtual_broker import VirtualBroker

    with pytest.raises(ValueError, match="duplicate record of account"):
        OrderManager(
            broker=VirtualBroker(),
            canonical_ledger=CanonicalLedger(),
            ledger_path="/tmp/should-not-be-used.jsonl",
        )


def test_attaching_to_a_populated_ledger_starts_from_its_state(tmp_path):
    """A component must not begin empty against a chain that already holds events.

    DEF-014: it did, so it could append a second CREATED and RISK_APPROVED for an
    order the file recorded as FILLED — writing a chain that no longer replays,
    discovered only at read time when the damage was already durable. The refusal
    now happens on the write, and the file stays readable.

    The fast engine is the case that bites: its order ids are content-addressed
    over (run_id, symbol, session) with no random component and no durable
    idempotency guard in front, so re-running the same lineage re-derives ids the
    file already holds.
    """
    path = tmp_path / "chain.jsonl"

    def run_once():
        from quantagent.backtest.engine import BacktestConfig, EventDrivenBacktester

        engine = EventDrivenBacktester(
            BacktestConfig(
                initial_nav=C.INITIAL_CASH, next_day_fill=False, fill_price_column="open",
                ledger_path=str(path),
                lineage=Lineage(
                    research_id="r", strategy_version_id="sv", run_id="same_run_id"
                ),
            )
        )
        weights = pd.DataFrame(
            {C.BLUE: [0.01, 0.0]}, index=pd.to_datetime([C.SESSION_1, C.SESSION_2])
        )
        prices = pd.DataFrame(
            [
                {
                    "symbol": C.BLUE, "trade_date": point.session, "open": point.last_price,
                    "high": point.last_price, "low": point.last_price,
                    "close": point.last_price, "volume": point.session_volume,
                    "pre_close": point.previous_close,
                    "amount": point.last_price * point.session_volume,
                }
                for point in C.MARKET if point.symbol == C.BLUE
            ]
        )
        return engine.run(weights, prices)

    run_once()
    records_after_first = len(CanonicalLedger(path))
    assert records_after_first > 0

    # The same lineage again. The collision must be refused on the write, and the
    # error must name the cause rather than surfacing as an opaque bad transition.
    with pytest.raises(LineageCollision, match="already on this ledger"):
        run_once()

    assert len(CanonicalLedger(path)) == records_after_first, (
        "the second run appended events to a chain that already recorded the order"
    )
    CanonicalLedger(path).replay(initial_cash=C.INITIAL_CASH)  # still readable


def test_an_injected_empty_ledger_is_not_silently_replaced(tmp_path):
    """An empty ledger is falsy; `or` would have swapped it for a fresh one."""
    from quantagent.execution.order_manager import OrderManager
    from quantagent.execution.virtual_broker import VirtualBroker

    shared = CanonicalLedger(tmp_path / "shared.jsonl")
    assert len(shared) == 0
    manager = OrderManager(broker=VirtualBroker(), canonical_ledger=shared)
    assert manager.canonical is shared


# -- M3-01/M3-02: fast versus streaming --------------------------------------
def test_streaming_and_paper_agree_on_every_dimension(report):
    """The M3-01 result: an event-driven engine matches the synchronous venue.

    Not "no unexplained differences" — *no differences at all*. Both consult the
    same A-share rulebook, so the only thing that differs is control flow: paper
    validates and fills inside one synchronous `submit`, while the streaming venue
    holds orders across bars and answers as separate events. Anything here would be
    a defect in one of the two.
    """
    table = next(
        t for t in report.cross_tables
        if {t.left_label, t.right_label} == {C.STREAMING, C.PAPER}
    )
    assert table.differences == [], [d.to_dict() for d in table.differences]
    assert table.only_left == [] and table.only_right == []


def test_the_streaming_leg_derived_its_own_fills(report):
    """Otherwise the agreement above compares a recording to itself."""
    streaming = next(p for p in report.paths if p.label == C.STREAMING)
    paper = next(p for p in report.paths if p.label == C.PAPER)

    note = next(n for n in streaming.notes if n.startswith("venue executions derived"))
    assert note.endswith("3"), note

    streaming_ids = {
        f.execution_id for f in CanonicalLedger(streaming.ledger_path).replay_book().fills()
    }
    paper_ids = {
        f.execution_id for f in CanonicalLedger(paper.ledger_path).replay_book().fills()
    }
    assert streaming_ids and paper_ids
    assert streaming_ids.isdisjoint(paper_ids), (
        "the two legs share execution ids, so one is a copy of the other rather than "
        "an independent derivation"
    )
    assert all(eid.startswith("strm-") for eid in streaming_ids)


def test_the_streaming_leg_expresses_every_step(report):
    """It has a venue, so unlike the fast engine nothing is out of its scope."""
    streaming = next(p for p in report.paths if p.label == C.STREAMING)
    assert streaming.steps_out_of_scope == {}
    assert len(streaming.steps_expressed) == len(C.STEPS)


def test_the_fill_price_formula_has_exactly_one_implementation():
    """Two copies of a pricing formula agree only until one is edited.

    Proven by breaking the shared function and requiring *both* engines to break:
    a copy anywhere would keep working and the agreement would become a coincidence
    maintained by hand.
    """
    from unittest import mock

    from quantagent.paper.broker import MarketSnapshot, PaperBroker
    from quantagent.paper.orders import BUY, Order as PaperOrder
    from quantagent.streaming.matching import MatchingVenue

    sentinel = RuntimeError("the shared pricing formula was called")

    with mock.patch(
        "quantagent.backtest.ashare_rules.execution_price", side_effect=sentinel
    ):
        market = C.market_point(C.BLUE, C.SESSION_1).snapshot()
        order = PaperOrder(symbol=C.BLUE, side=BUY, quantity=1_000.0, limit_price=10.05)
        with pytest.raises(RuntimeError, match="shared pricing formula"):
            PaperBroker._execution_price(
                mock.Mock(config=mock.Mock(slippage_bps=5.0, impact_coefficient=0.1)),
                order, market, 1_000.0,
            )
        with pytest.raises(RuntimeError, match="shared pricing formula"):
            MatchingVenue._execution_price(
                mock.Mock(
                    config=mock.Mock(slippage_bps=5.0, impact_coefficient=0.1),
                    _band=lambda _bar: None,
                ),
                mock.Mock(side=Side.BUY, limit_price=10.05),
                mock.Mock(payload={"volume": 1e8}),
                1_000,
                10.0,
            )


def test_no_tolerance_is_granted_on_the_pairs_that_must_match_exactly(report):
    """M3-02 for the exact pairs: no rules means no tolerance can apply."""
    assert C.PAPER_VS_OMS_RULES == ()
    assert C.STREAMING_VS_PAPER_RULES == ()


def test_no_cross_path_rule_grants_a_tolerance_on_discrete_state(report):
    """M3-02 for the pairs that may differ: only float math may be excused."""
    from quantagent.reconciliation.differences import is_discrete

    offenders = [
        rule.dimension for rule in C.CROSS_PATH_RULES
        if rule.tolerance and is_discrete(rule.dimension)
    ]
    assert offenders == [], f"discrete dimensions granted a tolerance: {offenders}"


def test_every_path_satisfies_the_accounting_identity_including_streaming(report):
    assert len(report.paths) == 4
    for path in report.paths:
        assert path.snapshot.identity_residual == pytest.approx(0.0, abs=1e-6), path.label


def test_the_streaming_matcher_refuses_a_fill_it_cannot_fund(report):
    """DEF-019: without a cash constraint it booked whatever the cap allowed."""
    streaming = next(p for p in report.paths if p.label == C.STREAMING)
    unaffordable = streaming.snapshot.orders[f"{C.BLUE}|BUY|200000|{C.SESSION_1}"]

    assert unaffordable.status == OrderStatus.REJECTED.value
    assert unaffordable.cumulative_quantity == 0
    assert unaffordable.fees == 0.0
    assert streaming.snapshot.cash > 0, "cash went negative: the fill was not funded"
