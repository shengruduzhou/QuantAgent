"""Golden venue scenarios: the cases the fast engine cannot express.

`test_golden_backtest_scenarios.py` covers the weights-driven fast engine. It
cannot express a partial fill, a cancel, a re-delivered callback or a corporate
action, because it has no venue and no working orders — so the scenarios the
programme lists for those cases have to be driven through the paper broker and the
streaming engine.

The discipline is the same as the fast-engine file: every expected figure is
derived by hand in the comment above it from the shipped defaults, and asserted
exactly. Reading a number out of the engine and calling it the expectation proves
only that the engine is self-consistent.

    commission      2.5 bps on gross, minimum 5.00 CNY
    transfer fee    0.1 bps on gross (SH main board)
    stamp duty      5.0 bps on gross, SELL SIDE ONLY
    participation   10% of session volume
    slippage        set to 0 here, so prices are exact and hand-checkable
"""

from __future__ import annotations

from datetime import time

import pytest

from quantagent.domain.ledger import CanonicalLedger
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import (
    CorporateAction,
    IllegalTransition,
    OrderStatus,
    Side,
)
from quantagent.domain.timeline import EventTime, exchange_moment
from quantagent.paper import ledger as paper_ledger
from quantagent.paper.broker import BrokerConfig, MarketSnapshot, PaperBroker
from quantagent.paper.orders import BUY, SELL, Order as PaperOrder
from quantagent.paper.portfolio import Portfolio
from quantagent.streaming.ambiguity import (
    AmbiguityPolicy,
    Bar,
    PathResolution,
    resolve_same_bar,
)
from quantagent.streaming.bus import EventBus
from quantagent.streaming.events import EventKind, MarketEvent
from quantagent.streaming.lifecycle import OrderLifecycle
from quantagent.streaming.matching import MatcherConfig, MatchingVenue

SYMBOL = "600000.SH"
SESSION_1 = "2026-08-04"
SESSION_2 = "2026-08-05"
SESSION_3 = "2026-08-06"
INITIAL = 1_000_000.0
RUN = Lineage(research_id="golden", strategy_version_id="sv_golden", run_id="golden_run")

#: Zero slippage and zero impact, so every price below is exactly the last price
#: (or the order's limit) and can be checked with a calculator.
EXACT = dict(slippage_bps=0.0, impact_coefficient=0.0, participation_cap=0.10)


def _paper(tmp_path, *, name: str = "paper") -> tuple[PaperBroker, Portfolio]:
    portfolio = Portfolio(portfolio_id=name, cash=INITIAL, initial_cash=INITIAL)
    broker = PaperBroker(
        portfolio,
        paper_ledger.EventLedger(tmp_path / f"{name}_op.jsonl"),
        run_id=name,
        config=BrokerConfig(**EXACT),
        canonical_ledger_path=str(tmp_path / f"{name}.jsonl"),
        lineage=Lineage(research_id="golden", strategy_version_id="sv_golden", run_id=name),
    )
    return broker, portfolio


def _market(
    session: str, *, price: float, volume: float, previous_close: float | None = None,
    board: str = "SH_Main",
) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=SYMBOL, trade_date=session, last_price=price,
        previous_close=previous_close if previous_close is not None else price,
        session_volume=volume, board=board,
    )


def _replayed(path):
    return CanonicalLedger(path).replay(initial_cash=INITIAL)


# --------------------------------------------------------------------------
# 1. Partial fill: the participation cap, to the cent
# --------------------------------------------------------------------------
def test_a_partial_fill_charges_costs_only_on_what_traded(tmp_path):
    """Order 3,000 into a 10,000-share session at 20.00, cap 10%.

    available   = 10,000 x 0.10                = 1,000 shares
    lot-rounded = 1,000                        = 1,000 (10 lots)
    price       = 20.00 (zero slippage/impact, under the 20.20 limit)
    gross       = 1,000 x 20.00                = 20,000.00
    commission  = 20,000 x 2.5/10000 = 5.00    =      5.00  (at the floor exactly)
    transfer    = 20,000 x 0.1/10000           =      0.20
    stamp duty  = 0 (buy side)
    cash out    = 20,000 + 5.00 + 0.20         = 20,005.20
    cash left   = 1,000,000 - 20,005.20        = 979,994.80
    leaves      = 3,000 - 1,000                =  2,000 still working
    """
    broker, portfolio = _paper(tmp_path)
    order = PaperOrder(symbol=SYMBOL, side=BUY, quantity=3_000.0, limit_price=20.20)

    broker.submit(order, _market(SESSION_1, price=20.00, volume=10_000))

    assert order.filled_quantity == 1_000
    assert order.remaining == 2_000
    assert order.state == "PARTIALLY_FILLED"
    fill = broker.fills[0]
    assert fill.price == pytest.approx(20.00)
    assert fill.commission == pytest.approx(5.00)
    assert fill.transfer_fee == pytest.approx(0.20)
    assert fill.stamp_duty == pytest.approx(0.0)
    assert portfolio.cash == pytest.approx(979_994.80, abs=1e-6)

    book, account = _replayed(tmp_path / "paper.jsonl")
    assert account.cash == pytest.approx(979_994.80, abs=1e-6)
    assert account.position(SYMBOL) == 1_000
    assert book.orders()[0].leaves_quantity == 2_000


def test_cancelling_the_remainder_leaves_the_executed_part_intact(tmp_path):
    """Same numbers as above, then cancel. Nothing about the fill may change.

    cash left   = 979,994.80  (unchanged by the cancel)
    position    = 1,000       (unchanged)
    leaves      = 0           (the order can no longer trade)
    """
    broker, portfolio = _paper(tmp_path)
    order = PaperOrder(symbol=SYMBOL, side=BUY, quantity=3_000.0, limit_price=20.20)
    broker.submit(order, _market(SESSION_1, price=20.00, volume=10_000))

    broker.cancel(order.order_id, _market(SESSION_1, price=20.00, volume=10_000))

    assert order.state == "CANCELLED"
    assert order.filled_quantity == 1_000
    assert portfolio.cash == pytest.approx(979_994.80, abs=1e-6)

    book, account = _replayed(tmp_path / "paper.jsonl")
    cancelled = book.orders()[0]
    assert cancelled.status is OrderStatus.CANCELLED
    assert cancelled.cumulative_quantity == 1_000
    assert cancelled.leaves_quantity == 0
    assert account.cash == pytest.approx(979_994.80, abs=1e-6)


def test_a_cancel_before_any_fill_leaves_the_account_untouched(tmp_path):
    """A resting buy below the market never trades, then is cancelled.

    last price 20.00, limit 19.00 -> the market never reached the limit
    cash        = 1,000,000.00 (unchanged)
    position    = 0
    """
    broker, portfolio = _paper(tmp_path)
    order = PaperOrder(symbol=SYMBOL, side=BUY, quantity=1_000.0, limit_price=19.00)
    broker.submit(order, _market(SESSION_1, price=20.00, volume=1e8))
    assert order.filled_quantity == 0

    broker.cancel(order.order_id, _market(SESSION_1, price=20.00, volume=1e8))

    assert order.state == "CANCELLED"
    assert portfolio.cash == pytest.approx(INITIAL)
    book, account = _replayed(tmp_path / "paper.jsonl")
    assert account.cash == pytest.approx(INITIAL)
    assert account.position(SYMBOL) == 0
    assert book.orders()[0].cumulative_quantity == 0


def test_a_late_fill_after_a_cancel_is_refused_and_moves_nothing(tmp_path):
    """A fill arriving after the order is dead must not resurrect it."""
    broker, portfolio = _paper(tmp_path)
    order = PaperOrder(symbol=SYMBOL, side=BUY, quantity=1_000.0, limit_price=19.00)
    broker.submit(order, _market(SESSION_1, price=20.00, volume=1e8))
    broker.cancel(order.order_id, _market(SESSION_1, price=20.00, volume=1e8))
    _, before = _replayed(tmp_path / "paper.jsonl")

    from quantagent.domain.orders import Fill as CanonicalFill, OrderEventType

    canonical_id = broker._canonical_ids[order.order_id]
    late = CanonicalFill(
        execution_id="LATE-1", order_id=canonical_id, symbol=SYMBOL, side=Side.BUY,
        quantity=1_000, price=19.00, reference_price=19.00, filled_at=SESSION_1,
    )
    with pytest.raises(IllegalTransition):
        broker._canonical_event(order, OrderEventType.FILL, fill=late, trade_date=SESSION_1)

    _, after = _replayed(tmp_path / "paper.jsonl")
    assert after.content_hash() == before.content_hash()


# --------------------------------------------------------------------------
# 2. Corporate actions (DEF-020)
# --------------------------------------------------------------------------
def test_a_cash_dividend_pays_income_and_leaves_nav_unchanged(tmp_path):
    """Buy 1,000 at 10.00, then a 0.50/share dividend on the ex date.

    buy gross   = 1,000 x 10.00                = 10,000.00
    commission  = max(10,000 x 2.5/10000, 5)   =      5.00  (at the floor)
    transfer    = 10,000 x 0.1/10000           =      0.10
    cash after  = 1,000,000 - 10,005.10        = 989,994.90
    all-in cost = 10,005.10 / 1,000            =     10.0051 per share

    dividend    = 1,000 x 0.50                 =    500.00
    cash after  = 989,994.90 + 500.00          = 990,494.90
    realised    = 0 + 500.00                   =    500.00  (income, not free cash)

    On the ex date the mark drops by the dividend, 10.00 -> 9.50:
    NAV         = 990,494.90 + 1,000 x 9.50    = 999,994.90
    which is exactly the NAV before the dividend (989,994.90 + 10,000.00).
    """
    broker, portfolio = _paper(tmp_path)
    broker.submit(
        PaperOrder(symbol=SYMBOL, side=BUY, quantity=1_000.0, limit_price=10.05),
        _market(SESSION_1, price=10.00, volume=1e8),
    )
    assert portfolio.cash == pytest.approx(989_994.90, abs=1e-6)
    nav_before = portfolio.cash + 1_000 * 10.00

    broker.apply_corporate_action(SYMBOL, cash_per_share=0.50, ex_date=SESSION_2)

    assert portfolio.cash == pytest.approx(990_494.90, abs=1e-6)
    assert portfolio.realised_pnl == pytest.approx(500.00, abs=1e-6)
    assert portfolio.cash + 1_000 * 9.50 == pytest.approx(nav_before, abs=1e-6)

    _, account = _replayed(tmp_path / "paper.jsonl")
    assert account.cash == pytest.approx(990_494.90, abs=1e-6)
    assert account.realised_pnl == pytest.approx(500.00, abs=1e-6)
    assert account.identity_residual({SYMBOL: 9.50}) == pytest.approx(0.0, abs=1e-9)


def test_a_two_for_one_split_moves_no_money(tmp_path):
    """Buy 1,000 at 10.00, then a 2:1 split.

    shares      = 1,000 x 2                    =  2,000
    basis       = 10.0051 / 2                  =      5.00255
    cash        = 989,994.90                   (unchanged)
    realised    = 0                            (unchanged: a split is not income)

    The price halves too, 10.00 -> 5.00:
    NAV         = 989,994.90 + 2,000 x 5.00    = 999,994.90  (unchanged)
    """
    broker, portfolio = _paper(tmp_path)
    broker.submit(
        PaperOrder(symbol=SYMBOL, side=BUY, quantity=1_000.0, limit_price=10.05),
        _market(SESSION_1, price=10.00, volume=1e8),
    )
    nav_before = portfolio.cash + 1_000 * 10.00

    broker.apply_corporate_action(SYMBOL, share_ratio=2.0, ex_date=SESSION_2)

    assert portfolio.position(SYMBOL).total == 2_000
    assert portfolio.cash == pytest.approx(989_994.90, abs=1e-6)
    assert portfolio.realised_pnl == pytest.approx(0.0, abs=1e-9)
    # Cost basis halves with the share count, so book value is unchanged: 10.0051
    # on 1,000 shares becomes 5.00255 on 2,000.
    assert portfolio.position(SYMBOL).average_cost == pytest.approx(5.00255, abs=1e-9)
    assert portfolio.cash + 2_000 * 5.00 == pytest.approx(nav_before, abs=1e-6)

    _, account = _replayed(tmp_path / "paper.jsonl")
    assert account.position(SYMBOL) == 2_000
    assert account.cost_basis[SYMBOL] == pytest.approx(5.00255, abs=1e-9)
    assert account.identity_residual({SYMBOL: 5.00}) == pytest.approx(0.0, abs=1e-9)


def test_a_corporate_action_reaches_the_canonical_ledger(tmp_path):
    """DEF-020: it used to mutate the portfolio and write nothing canonical."""
    broker, portfolio = _paper(tmp_path)
    broker.submit(
        PaperOrder(symbol=SYMBOL, side=BUY, quantity=1_000.0, limit_price=10.05),
        _market(SESSION_1, price=10.00, volume=1e8),
    )
    records_before = len(CanonicalLedger(tmp_path / "paper.jsonl"))

    result = broker.apply_corporate_action(SYMBOL, cash_per_share=0.50, ex_date=SESSION_2)

    ledger = CanonicalLedger(tmp_path / "paper.jsonl")
    assert len(ledger) == records_before + 1
    assert ledger.verify()["valid"]
    actions = ledger.corporate_actions()
    assert len(actions) == 1
    position, action = actions[0]
    assert action.corporate_action_id == result["corporateActionId"]
    assert action.cash_per_share == pytest.approx(0.50)
    assert action.ex_date == SESSION_2
    assert position == len(ledger.events()), "the action must sit after the fill"


def test_holding_through_an_ex_date_earns_the_same_total_return_as_selling_first(tmp_path):
    """Two accounts, identical but for whether they held through the ex date.

    My first version of this test asserted the holder ended 500.00 ahead. It does
    not, and the engine said so: the two end within 0.26 of each other. That is the
    *correct* answer and the reason the ex-date adjustment exists — the holder
    receives 500.00 of income and then sells 500.00 lower, so total return is
    unchanged. The residual 0.26 is only the fee difference on a smaller notional:

    sold cum-dividend at 10.00: stamp 5.000 + transfer 0.100 + commission 5.00 = 10.100
    sold ex-dividend  at  9.50: stamp 4.750 + transfer 0.095 + commission 5.00 =  9.845
    difference                                                                =  0.255

    An engine that credited the dividend *without* dropping the mark would show the
    holder 500.00 ahead — free money on the ex date — which is what this pins down.
    """
    def run(*, sell_before_ex_date: bool, name: str):
        broker, portfolio = _paper(tmp_path, name=name)
        broker.submit(
            PaperOrder(symbol=SYMBOL, side=BUY, quantity=1_000.0, limit_price=10.05),
            _market(SESSION_1, price=10.00, volume=1e8),
        )
        broker.close_session(SESSION_1)
        if sell_before_ex_date:
            broker.submit(
                PaperOrder(symbol=SYMBOL, side=SELL, quantity=1_000.0, limit_price=9.95),
                _market(SESSION_2, price=10.00, volume=1e8),
            )
            broker.apply_corporate_action(SYMBOL, cash_per_share=0.50, ex_date=SESSION_3)
        else:
            broker.apply_corporate_action(SYMBOL, cash_per_share=0.50, ex_date=SESSION_2)
            broker.submit(
                PaperOrder(symbol=SYMBOL, side=SELL, quantity=1_000.0, limit_price=9.45),
                _market(SESSION_3, price=9.50, volume=1e8, previous_close=9.50),
            )
        _, account = _replayed(tmp_path / f"{name}.jsonl")
        return portfolio, account

    sold_first, sold_first_replay = run(sell_before_ex_date=True, name="sold_first")
    held_through, held_replay = run(sell_before_ex_date=False, name="held_through")

    for portfolio, account in ((sold_first, sold_first_replay), (held_through, held_replay)):
        assert account.position(SYMBOL) == 0
        assert account.cash == pytest.approx(portfolio.cash, abs=1e-6)
        assert account.realised_pnl == pytest.approx(portfolio.realised_pnl, abs=1e-6)

    difference = held_replay.realised_pnl - sold_first_replay.realised_pnl
    assert difference == pytest.approx(0.255, abs=1e-6), (
        f"the ex-date adjustment is wrong: holding through changed total return by "
        f"{difference:.4f}, which should be only the 0.255 fee difference"
    )
    # And the dividend really was received, rather than the two agreeing because
    # nothing happened.
    assert held_replay.cash - sold_first_replay.cash == pytest.approx(0.255, abs=1e-6)


def test_where_a_corporate_action_sits_in_the_stream_changes_the_answer(tmp_path):
    """The interleaving property, tested directly.

    A replay that applied every corporate action at the end of the stream, rather
    than at its recorded position, would give the same answer for both of these.
    """
    from quantagent.domain.accounting import AccountState, replay_account
    from quantagent.domain.orders import (
        Fill as CanonicalFill,
        OrderBook,
        OrderEvent,
        OrderEventType,
        OrderIntent,
        Signal,
    )

    book = OrderBook()
    signal = Signal.create(symbol=SYMBOL, trade_date=SESSION_1, score=1.0, lineage=RUN)
    buy = OrderIntent.create(
        symbol=SYMBOL, side=Side.BUY, quantity=1_000, trade_date=SESSION_1,
        lineage=signal.lineage, limit_price=10.05,
    )
    order = book.open(buy)
    for stage in (
        OrderEventType.RISK_APPROVED, OrderEventType.SUBMITTED, OrderEventType.ACCEPTED
    ):
        book.apply(order.order_id, stage)
    book.apply(
        order.order_id, OrderEventType.FILL,
        fill=CanonicalFill(
            execution_id="X1", order_id=order.order_id, symbol=SYMBOL, side=Side.BUY,
            quantity=1_000, price=10.00, reference_price=10.00, filled_at=SESSION_1,
        ),
    )
    events = book.events()
    dividend = CorporateAction.create(
        symbol=SYMBOL, ex_date=SESSION_2, lineage=RUN, cash_per_share=0.50
    )

    before_the_fill = replay_account(
        events, initial_cash=INITIAL, corporate_actions=((0, dividend),)
    )
    after_the_fill = replay_account(
        events, initial_cash=INITIAL, corporate_actions=((len(events), dividend),)
    )

    # Nothing was held before the fill, so that dividend paid nothing.
    assert before_the_fill.realised_pnl == pytest.approx(0.0, abs=1e-9)
    assert after_the_fill.realised_pnl == pytest.approx(500.0, abs=1e-9)
    assert after_the_fill.cash - before_the_fill.cash == pytest.approx(500.0, abs=1e-6)


def test_a_fractional_entitlement_is_refused_rather_than_invented(tmp_path):
    """A 1.5:1 ratio on 1,001 shares yields 1,501.5 — which cannot exist."""
    from quantagent.domain.accounting import AccountState, InvariantViolation
    from quantagent.domain.orders import PositionLot

    lot = PositionLot(
        position_lot_id="l1", symbol=SYMBOL, quantity=1_001, cost_price=10.0,
        acquired_on=SESSION_1,
    )
    account = AccountState(
        cash=INITIAL, initial_cash=INITIAL, lots={SYMBOL: (lot,)},
        cost_basis={SYMBOL: 10.0},
    )
    action = CorporateAction.create(
        symbol=SYMBOL, ex_date=SESSION_2, lineage=RUN, share_ratio=1.5
    )

    with pytest.raises(InvariantViolation, match="Fractional entitlements"):
        account.apply_corporate_action(action)


def test_a_corporate_action_on_a_position_never_held_has_no_effect(tmp_path):
    from quantagent.domain.accounting import AccountState

    account = AccountState.opening(INITIAL)
    action = CorporateAction.create(
        symbol=SYMBOL, ex_date=SESSION_2, lineage=RUN, cash_per_share=0.50
    )

    assert account.apply_corporate_action(action) is account


# --------------------------------------------------------------------------
# 3. Two strategies competing for one account
# --------------------------------------------------------------------------
def test_two_strategies_competing_for_cash_do_not_overdraw(tmp_path):
    """Two 600,000 orders against 1,000,000 of cash. Exactly one can fill.

    order one   = 60,000 shares x 10.00        = 600,000.00
    commission  = 600,000 x 2.5/10000          =     150.00
    transfer    = 600,000 x 0.1/10000          =       6.00
    cash after  = 1,000,000 - 600,156.00       = 399,844.00
    order two   needs 600,156.00 > 399,844.00  -> refused, zero fills
    """
    broker, portfolio = _paper(tmp_path)
    market = _market(SESSION_1, price=10.00, volume=1e8)

    first = PaperOrder(symbol=SYMBOL, side=BUY, quantity=60_000.0, limit_price=10.05)
    second = PaperOrder(symbol=SYMBOL, side=BUY, quantity=60_000.0, limit_price=10.05)
    broker.submit(first, market)
    broker.submit(second, market)

    assert first.state == "FILLED"
    assert second.state == "REJECTED"
    assert second.filled_quantity == 0
    assert portfolio.cash == pytest.approx(399_844.00, abs=1e-6)

    book, account = _replayed(tmp_path / "paper.jsonl")
    assert account.cash == pytest.approx(399_844.00, abs=1e-6)
    assert account.cash > 0, "the account was overdrawn"
    assert account.position(SYMBOL) == 60_000
    rejected = [o for o in book.orders() if o.status is OrderStatus.REJECTED]
    assert len(rejected) == 1 and rejected[0].reason


def test_two_strategies_competing_for_inventory_cannot_both_sell(tmp_path):
    """1,000 settled shares, two 1,000-share sells. Only one can be filled.

    The second must be refused for exceeding settled inventory, not filled short.
    """
    broker, portfolio = _paper(tmp_path)
    broker.submit(
        PaperOrder(symbol=SYMBOL, side=BUY, quantity=1_000.0, limit_price=10.05),
        _market(SESSION_1, price=10.00, volume=1e8),
    )
    broker.close_session(SESSION_1)
    market = _market(SESSION_2, price=10.00, volume=1e8)

    first = PaperOrder(symbol=SYMBOL, side=SELL, quantity=1_000.0, limit_price=9.95)
    second = PaperOrder(symbol=SYMBOL, side=SELL, quantity=1_000.0, limit_price=9.95)
    broker.submit(first, market)
    broker.submit(second, market)

    assert first.state == "FILLED"
    assert second.state == "REJECTED"
    assert "T+1-settled" in (second.reject_reason or "")

    _, account = _replayed(tmp_path / "paper.jsonl")
    assert account.position(SYMBOL) == 0, "the account went short"


# --------------------------------------------------------------------------
# 4. Same-bar stop and target (M2-05), with the arithmetic stated
# --------------------------------------------------------------------------
def test_a_bar_touching_both_levels_is_priced_at_the_stop(tmp_path):
    """Long 1,000 from 10.00, stop 9.50, target 10.50, bar 9.40-10.60.

    Both levels are inside the bar, so the bar is consistent with either outcome.
    The conservative rule takes the stop:
    exit price  = 9.50
    loss/share  = 9.50 - 10.00                 =     -0.50
    gross loss  = 1,000 x 0.50                 =    500.00
    Taking the target instead would report +500.00 on the same bar — a 1,000.00
    swing decided by which branch the code happened to check first.
    """
    outcome = resolve_same_bar(
        Bar(open=10.00, high=10.60, low=9.40, close=10.20),
        side=Side.BUY, stop=9.50, target=10.50,
    )

    assert outcome.resolution is PathResolution.AMBIGUOUS_RESOLVED_CONSERVATIVELY
    assert outcome.triggered == "stop"
    assert outcome.trigger_price == pytest.approx(9.50)
    assert outcome.is_assumption is True

    favourable = 1_000 * (10.50 - 10.00)
    conservative = 1_000 * (outcome.trigger_price - 10.00)
    assert conservative == pytest.approx(-500.00)
    assert favourable - conservative == pytest.approx(1_000.00), (
        "the swing this rule prevents"
    )


def test_the_same_bar_marked_ambiguous_prices_nothing(tmp_path):
    outcome = resolve_same_bar(
        Bar(open=10.00, high=10.60, low=9.40, close=10.20),
        side=Side.BUY, stop=9.50, target=10.50,
        policy=AmbiguityPolicy.MARK_AMBIGUOUS,
    )
    assert outcome.resolution is PathResolution.AMBIGUOUS_UNRESOLVED
    assert outcome.trigger_price is None


# --------------------------------------------------------------------------
# 5. The streaming engine agrees on the partial fill, to the cent
# --------------------------------------------------------------------------
def test_streaming_reproduces_the_partial_fill_arithmetic(tmp_path):
    """The same 3,000-into-10,000 order, derived from a bar by the matcher.

    Same expected figures as `test_a_partial_fill_charges_costs_only_on_what_traded`:
    1,000 shares at 20.00, commission 5.00, transfer 0.20, cash 979,994.80.
    """
    lifecycle = OrderLifecycle(
        ledger=CanonicalLedger(tmp_path / "stream.jsonl"), lineage=RUN,
        initial_cash=INITIAL,
    )
    bus = EventBus()
    venue = MatchingVenue(lifecycle=lifecycle, bus=bus, config=MatcherConfig(**EXACT))

    def handler(event, frontier):
        lifecycle.handle(event, frontier)
        venue.handle(event)

    bus.publish(
        MarketEvent(
            kind=EventKind.ORDER,
            times=EventTime.immediate(exchange_moment(SESSION_1, time(10, 0))),
            symbol=SYMBOL,
            payload={
                "clientOrderId": "c1", "side": "BUY", "quantity": 3_000,
                "limitPrice": 20.20, "board": "SH_Main", "previousClose": 20.00,
            },
        )
    )
    bus.publish(
        MarketEvent(
            kind=EventKind.BAR,
            times=EventTime.immediate(exchange_moment(SESSION_1, time(15, 0))),
            symbol=SYMBOL,
            payload={
                "close": 20.00, "previousClose": 20.00, "volume": 10_000,
                "board": "SH_Main",
            },
        )
    )
    bus.run(handler)

    account = lifecycle.account()
    order = lifecycle.order_book().orders()[0]
    assert order.cumulative_quantity == 1_000
    assert order.leaves_quantity == 2_000
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.total_fees == pytest.approx(5.20, abs=1e-6)
    assert account.cash == pytest.approx(979_994.80, abs=1e-6)
    assert account.position(SYMBOL) == 1_000


# --------------------------------------------------------------------------
# 6. Delisting and long suspension: the value is unknown, never zero (DEF-021)
# --------------------------------------------------------------------------
def test_a_delisted_holding_makes_nav_unknown_rather_than_zero(tmp_path):
    """Buy 1,000 at 10.00, then the symbol stops trading. What is it worth?

    Not zero, and not 10.00 carried forward. Both are specific claims and both are
    usually false. Measured cost of the zero default:

    cash        = 989,994.90
    all-in cost = 10,005.10  (1,000 x 10.0051)
    NAV at 10.00 = 989,994.90 + 10,000.00       = 999,994.90
    NAV at zero  = 989,994.90                   = 989,994.90
    understated by                              =  10,000.00
    fabricated loss                             =  10,005.10

    And the accounting identity held either way, because cash and the mark were
    consistently wrong together — which is why this needed an explicit refusal
    rather than a check.
    """
    from quantagent.domain.accounting import UnpriceablePosition

    broker, portfolio = _paper(tmp_path)
    broker.submit(
        PaperOrder(symbol=SYMBOL, side=BUY, quantity=1_000.0, limit_price=10.05),
        _market(SESSION_1, price=10.00, volume=1e8),
    )
    _, account = _replayed(tmp_path / "paper.jsonl")
    assert account.cash == pytest.approx(989_994.90, abs=1e-6)

    # While the symbol still trades, NAV is a number.
    assert account.nav({SYMBOL: 10.00}) == pytest.approx(999_994.90, abs=1e-6)

    # Delisted: no mark. Every valuation refuses rather than defaulting.
    assert account.unpriceable({}) == (SYMBOL,)
    for query in (account.market_value, account.unrealised_pnl, account.nav):
        with pytest.raises(UnpriceablePosition, match="unknown, not"):
            query({})

    # And the reportable form says unknown, explicitly.
    valuation = account.valuation({})
    assert valuation["nav"] is None
    assert valuation["unrealisedPnl"] is None
    assert valuation["unpriceableSymbols"] == [SYMBOL]
    assert "unknown rather than zero" in valuation["reason"]


def test_paper_and_the_replay_agree_that_a_delisted_holding_is_unpriceable(tmp_path):
    """Both records of account must refuse, or one of them invents a number."""
    from quantagent.domain.accounting import UnpriceablePosition

    broker, portfolio = _paper(tmp_path)
    broker.submit(
        PaperOrder(symbol=SYMBOL, side=BUY, quantity=1_000.0, limit_price=10.05),
        _market(SESSION_1, price=10.00, volume=1e8),
    )
    _, account = _replayed(tmp_path / "paper.jsonl")

    assert portfolio.unpriceable({}) == account.unpriceable({}) == (SYMBOL,)
    with pytest.raises(UnpriceablePosition):
        portfolio.equity({})
    snapshot = portfolio.to_dict({})
    assert snapshot["equity"] is None
    assert snapshot["unpriceable_symbols"] == [SYMBOL]
    # Cash is still knowable, and still agrees. Only valuation is unavailable.
    assert snapshot["cash"] == pytest.approx(account.cash, abs=1e-6)


def test_a_partially_priced_book_refuses_rather_than_valuing_what_it_can(tmp_path):
    """One tradable holding and one delisted: the *whole* NAV is unknown.

    Reporting NAV for the priceable part and quietly omitting the rest is the
    version of this defect that looks most reasonable and is still wrong — the
    number would be presented as the account's value while excluding some of it.
    """
    from quantagent.domain.accounting import UnpriceablePosition

    other = "000001.SZ"
    broker, portfolio = _paper(tmp_path)
    broker.submit(
        PaperOrder(symbol=SYMBOL, side=BUY, quantity=1_000.0, limit_price=10.05),
        _market(SESSION_1, price=10.00, volume=1e8),
    )
    broker.submit(
        PaperOrder(symbol=other, side=BUY, quantity=1_000.0, limit_price=20.05, board="SZ_Main"),
        MarketSnapshot(
            symbol=other, trade_date=SESSION_1, last_price=20.00, previous_close=20.00,
            session_volume=1e8, board="SZ_Main",
        ),
    )
    _, account = _replayed(tmp_path / "paper.jsonl")

    assert account.unpriceable({SYMBOL: 10.00}) == (other,)
    with pytest.raises(UnpriceablePosition, match=other):
        account.nav({SYMBOL: 10.00})
    assert account.valuation({SYMBOL: 10.00})["nav"] is None

    # With both marked, it is a number again.
    assert account.nav({SYMBOL: 10.00, other: 20.00}) == pytest.approx(
        account.cash + 10_000.00 + 20_000.00, abs=1e-6
    )


def test_a_flat_position_needs_no_mark(tmp_path):
    """Sold out of a symbol that then delisted: nothing held, nothing to value."""
    broker, portfolio = _paper(tmp_path)
    broker.submit(
        PaperOrder(symbol=SYMBOL, side=BUY, quantity=1_000.0, limit_price=10.05),
        _market(SESSION_1, price=10.00, volume=1e8),
    )
    broker.close_session(SESSION_1)
    broker.submit(
        PaperOrder(symbol=SYMBOL, side=SELL, quantity=1_000.0, limit_price=9.95),
        _market(SESSION_2, price=10.00, volume=1e8),
    )
    _, account = _replayed(tmp_path / "paper.jsonl")

    assert account.position(SYMBOL) == 0
    assert account.unpriceable({}) == ()
    assert account.nav({}) == pytest.approx(account.cash, abs=1e-6)
    assert portfolio.unpriceable({}) == ()


# --------------------------------------------------------------------------
# 7. Missing benchmark: excess return is unknown, never zero (DEF-022)
# --------------------------------------------------------------------------
def _flat_nav(sessions: int = 10):
    import pandas as pd

    dates = pd.date_range("2026-01-05", periods=sessions, freq="B")
    return dates, pd.Series([INITIAL] * sessions, index=dates)


def _benchmark_panel(dates, closes):
    import pandas as pd

    return pd.DataFrame(
        {"symbol": ["BENCH"] * len(closes), "trade_date": dates[: len(closes)],
         "close": list(closes)}
    )


def _report(nav, panel):
    import pandas as pd

    from quantagent.backtest.paper_report import _pnl_frame, _summary

    pnl = _pnl_frame(nav, INITIAL, panel, "BENCH" if panel is not None else None)
    empty = pd.DataFrame(columns=["status", "estimated_fee", "estimated_slippage"])
    return _summary(pnl, empty, pd.DataFrame(), pd.DataFrame(), INITIAL)


def test_a_complete_benchmark_gives_excess_return_to_the_basis_point():
    """Strategy exactly flat, benchmark 100 -> 120 over 10 sessions.

    strategy return  = 0.00%
    benchmark return = 120/100 - 1              = +20.00%
    excess return    = 0.00% - 20.00%           = -20.00%
    """
    dates, nav = _flat_nav()
    panel = _benchmark_panel(
        dates, [100.0, 100.0, 100.0, 100.0, 100.0, 104.0, 108.0, 112.0, 116.0, 120.0]
    )

    summary = _report(nav, panel)

    assert summary["benchmark_status"] == "complete"
    assert summary["benchmark_sessions_covered"] == 10
    assert summary["benchmark_return"] == pytest.approx(0.20, abs=1e-9)
    assert summary["excess_return"] == pytest.approx(-0.20, abs=1e-9)


def test_a_benchmark_gap_makes_excess_return_unknown_not_flat():
    """The same run, with the benchmark absent for the 5 sessions it rose in.

    Filling the gap with 0%-return days reported benchmark_return = +0.00% and
    excess_return = +0.00%, when the truth is +20% and -20% — excess return
    overstated by 20 percentage points, presented as a confident number with
    nothing in the output to say half the benchmark was missing (DEF-022).
    """
    dates, nav = _flat_nav()
    partial = _benchmark_panel(dates, [100.0, 100.0, 100.0, 100.0, 100.0])

    summary = _report(nav, partial)

    assert summary["benchmark_status"] == "incomplete"
    assert summary["benchmark_sessions_covered"] == 5
    assert summary["benchmark_sessions_missing"] == 5
    assert summary["benchmark_return"] is None
    assert summary["excess_return"] is None
    assert summary["excess_return_after_costs"] is None
    assert summary["information_ratio"] is None


def test_an_absent_benchmark_is_distinguishable_from_an_incomplete_one():
    """Different causes need different fixes, so they cannot share one answer."""
    dates, nav = _flat_nav()

    absent = _report(nav, None)
    incomplete = _report(nav, _benchmark_panel(dates, [100.0] * 5))

    assert absent["benchmark_status"] == "absent"
    assert absent["benchmark_sessions_covered"] == 0
    assert incomplete["benchmark_status"] == "incomplete"
    assert incomplete["benchmark_sessions_covered"] == 5
    # Both are unknown, and neither is zero.
    assert absent["excess_return"] is incomplete["excess_return"] is None


def test_the_gate_reports_unknown_and_names_the_cause():
    """An operator told only "no benchmark" would hunt a config problem that is not there."""
    from quantagent.data.v7_quality_gates import (
        GATE_UNKNOWN,
        V7ModelAcceptanceGateConfig,
        evaluate_model_acceptance_gates,
    )

    dates, nav = _flat_nav()
    incomplete = _report(nav, _benchmark_panel(dates, [100.0] * 5))
    metrics = {
        "rank_ic_mean": 0.05, "rank_ic_stability": 0.5,
        "turnover_adjusted_net_return": 0.08, "max_drawdown": -0.03,
        "single_factor_dominance": 0.4, "adverse_regime_passed": True,
        "uses_mock_or_synthetic": False, "pit_violation_count": 0,
        "selection_pressure_min": 8.0, "training_dataset_symbol_count": 400,
        "prediction_symbol_count": 356, "effective_universe_min": 120,
        **dict(incomplete),
    }

    report = evaluate_model_acceptance_gates(
        metrics,
        V7ModelAcceptanceGateConfig(
            require_paper_report=False, require_benchmark=False,
            min_training_symbols=2, min_prediction_symbols=1,
            min_effective_universe_by_date=1,
        ),
    )
    gate = next(g for g in report.gates if g["name"] == "excess_return_after_costs")

    assert gate["status"] == GATE_UNKNOWN
    assert gate["actual"] is None
    assert gate["passed"] is False
    # The machine-readable code stays stable; the cause is structured data.
    assert gate["reason"] == "excess_return_after_costs_unknown"
    assert gate["benchmark_status"] == "incomplete"
    assert gate["benchmark_sessions_missing"] == 5
    assert "不完整" in gate["detail"]
    assert report.passed is False


# --------------------------------------------------------------------------
# 8. Expiration: an unfilled order at the end of the session
# --------------------------------------------------------------------------
def test_an_order_left_working_at_the_close_expires_without_trading(tmp_path):
    """A resting buy 1.00 below the market, expired at the session close.

    cash        = 1,000,000.00 (unchanged: it never traded)
    cumulative  = 0
    leaves      = 0            (expired orders can no longer trade)
    status      = EXPIRED      (distinct from CANCELLED: nobody asked to cancel it)
    """
    lifecycle = OrderLifecycle(
        ledger=CanonicalLedger(tmp_path / "expiry.jsonl"), lineage=RUN,
        initial_cash=INITIAL,
    )
    bus = EventBus()
    venue = MatchingVenue(lifecycle=lifecycle, bus=bus, config=MatcherConfig(**EXACT))

    def handler(event, frontier):
        lifecycle.handle(event, frontier)
        venue.handle(event)

    bus.publish(
        MarketEvent(
            kind=EventKind.ORDER,
            times=EventTime.immediate(exchange_moment(SESSION_1, time(10, 0))),
            symbol=SYMBOL,
            payload={
                "clientOrderId": "resting", "side": "BUY", "quantity": 1_000,
                "limitPrice": 9.00, "board": "SH_Main", "previousClose": 10.00,
            },
        )
    )
    bus.publish(
        MarketEvent(
            kind=EventKind.BAR,
            times=EventTime.immediate(exchange_moment(SESSION_1, time(15, 0))),
            symbol=SYMBOL,
            payload={"close": 10.00, "previousClose": 10.00, "volume": 1e8},
        )
    )
    bus.run(handler)
    assert lifecycle.status_of("resting") is OrderStatus.ACCEPTED, "it must not have traded"

    bus.publish(
        MarketEvent(
            kind=EventKind.EXPIRY,
            times=EventTime.immediate(exchange_moment(SESSION_1, time(15, 1))),
            symbol=SYMBOL,
            payload={"clientOrderId": "resting", "reason": "end_of_session"},
        )
    )
    bus.run(handler)

    order = lifecycle.order_book().orders()[0]
    assert order.status is OrderStatus.EXPIRED
    assert order.cumulative_quantity == 0
    assert order.leaves_quantity == 0
    assert order.reason == "end_of_session"
    assert lifecycle.account().cash == pytest.approx(INITIAL)
    assert "EXPIRED" in [
        e.event_type.value for e in lifecycle.order_book().history_of(order.order_id)
    ]


def test_expiring_an_already_filled_order_is_absorbed(tmp_path):
    """The close arriving after a fill is normal, not a lifecycle violation."""
    lifecycle = OrderLifecycle(
        ledger=CanonicalLedger(tmp_path / "expiry2.jsonl"), lineage=RUN,
        initial_cash=INITIAL,
    )
    bus = EventBus()
    venue = MatchingVenue(lifecycle=lifecycle, bus=bus, config=MatcherConfig(**EXACT))

    def handler(event, frontier):
        lifecycle.handle(event, frontier)
        venue.handle(event)

    bus.publish(
        MarketEvent(
            kind=EventKind.ORDER,
            times=EventTime.immediate(exchange_moment(SESSION_1, time(10, 0))),
            symbol=SYMBOL,
            payload={
                "clientOrderId": "c1", "side": "BUY", "quantity": 1_000,
                "limitPrice": 10.05, "board": "SH_Main", "previousClose": 10.00,
            },
        )
    )
    bus.publish(
        MarketEvent(
            kind=EventKind.BAR,
            times=EventTime.immediate(exchange_moment(SESSION_1, time(15, 0))),
            symbol=SYMBOL,
            payload={"close": 10.00, "previousClose": 10.00, "volume": 1e8},
        )
    )
    bus.run(handler)
    before = lifecycle.account()

    bus.publish(
        MarketEvent(
            kind=EventKind.EXPIRY,
            times=EventTime.immediate(exchange_moment(SESSION_1, time(15, 1))),
            symbol=SYMBOL,
            payload={"clientOrderId": "c1"},
        )
    )
    bus.run(handler)

    assert lifecycle.status_of("c1") is OrderStatus.FILLED
    assert lifecycle.account().content_hash() == before.content_hash()
