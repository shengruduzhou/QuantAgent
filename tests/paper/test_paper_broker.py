"""Local paper broker: A-share rules, ledger integrity, recovery, determinism.

Each test names a way a paper book silently diverges from what the market would
actually have done -- filling a locked limit board, selling unsettled shares,
ignoring lot rules, or reporting a recovery that only proves a snapshot was
readable.
"""

from __future__ import annotations

import json

import pytest

from quantagent.paper import broker as bk
from quantagent.paper import ledger as lg
from quantagent.paper import recovery as rc
from quantagent.paper.orders import (
    BUY,
    CANCELLED,
    FILLED,
    LIMIT,
    MARKETABLE_LIMIT,
    POV,
    REJECTED,
    SELL,
    TWAP,
    VWAP,
    Order,
    OrderStateError,
    ParentOrder,
)
from quantagent.paper.portfolio import (
    InsufficientCash,
    InsufficientSellable,
    Portfolio,
)


@pytest.fixture
def setup(tmp_path):
    ledger = lg.EventLedger(tmp_path / "ledger.jsonl")
    portfolio = Portfolio(portfolio_id="P1", cash=1_000_000.0, initial_cash=1_000_000.0)
    broker = bk.PaperBroker(portfolio, ledger, run_id="R1",
                            config=bk.BrokerConfig(slippage_bps=0.0,
                                                   impact_coefficient=0.0))
    return broker, portfolio, ledger


def market(**kw):
    base = dict(symbol="600000.SH", trade_date="2026-07-24", last_price=9.04,
                previous_close=9.00, session_volume=50_000_000.0, board="SH_Main",
                clock="10:00:00")
    base.update(kw)
    return bk.MarketSnapshot(**base)


def buy(qty=10_000, price=9.50, **kw):
    return Order(symbol="600000.SH", side=BUY, quantity=qty,
                 order_type=MARKETABLE_LIMIT, limit_price=price, **kw)


class TestNoLiveConnection:
    def test_broker_declares_itself_local(self):
        assert bk.PaperBroker.is_local_simulation is True
        assert bk.PaperBroker.has_broker_connection is False

    def test_no_network_or_broker_import_in_the_package(self):
        import ast
        from pathlib import Path

        package = Path(bk.__file__).parent
        forbidden = {"requests", "httpx", "socket", "xtquant", "MetaTrader5", "urllib"}
        offenders = {}
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module.split(".")[0]]
                hit = set(names) & forbidden
                if hit:
                    offenders.setdefault(path.name, set()).update(hit)
        assert not offenders, f"paper package must not reach a network: {offenders}"

    def test_unconstrained_market_order_does_not_exist(self):
        with pytest.raises(ValueError, match="unconstrained market order"):
            Order(symbol="600000.SH", side=BUY, quantity=100,
                  order_type=MARKETABLE_LIMIT, limit_price=None)


class TestAShareRules:
    def test_t_plus_one_blocks_same_day_sell(self, setup):
        broker, portfolio, _ = setup
        broker.submit(buy(), market())
        assert portfolio.sellable("600000.SH") == 0.0
        order = broker.submit(
            Order(symbol="600000.SH", side=SELL, quantity=1_000,
                  order_type=LIMIT, limit_price=9.00), market())
        assert order.state == REJECTED
        assert "T+1" in order.reject_reason

    def test_settlement_makes_shares_sellable(self, setup):
        broker, portfolio, _ = setup
        broker.submit(buy(), market())
        broker.close_session("2026-07-24")
        assert portfolio.sellable("600000.SH") == 10_000.0
        order = broker.submit(
            Order(symbol="600000.SH", side=SELL, quantity=1_000,
                  order_type=LIMIT, limit_price=9.00), market())
        assert order.state == FILLED

    def test_cannot_buy_a_locked_limit_up(self, setup):
        broker, _, _ = setup
        order = broker.submit(buy(price=9.90), market(last_price=9.90))
        assert order.state == REJECTED
        assert "limit up" in order.reject_reason

    def test_cannot_sell_a_locked_limit_down(self, setup):
        broker, portfolio, _ = setup
        broker.submit(buy(), market())
        broker.close_session("2026-07-24")
        order = broker.submit(
            Order(symbol="600000.SH", side=SELL, quantity=1_000,
                  order_type=LIMIT, limit_price=8.10), market(last_price=8.10))
        assert order.state == REJECTED
        assert "limit down" in order.reject_reason

    def test_suspension_blocks_both_sides(self, setup):
        broker, _, _ = setup
        order = broker.submit(buy(), market(is_suspended=True))
        assert order.state == REJECTED
        assert "suspended" in order.reject_reason

    def test_st_narrower_band_rejects_an_out_of_band_limit(self, setup):
        broker, _, _ = setup
        broker.config.allow_st_buy = True
        # ST band is +/-5%: 9.00 * 1.05 = 9.45, so 9.60 is outside.
        order = broker.submit(buy(price=9.60), market(is_st=True))
        assert order.state == REJECTED
        assert "ceiling" in order.reject_reason

    def test_st_buy_blocked_by_policy_by_default(self, setup):
        broker, _, _ = setup
        order = broker.submit(buy(price=9.20), market(is_st=True))
        assert order.state == REJECTED
        assert "ST buy blocked" in order.reject_reason

    def test_sub_lot_order_rejected_not_rounded_up(self, setup):
        broker, _, _ = setup
        order = broker.submit(buy(qty=50), market())
        assert order.state == REJECTED
        assert "minimum lot" in order.reject_reason

    def test_star_board_requires_two_hundred_shares(self, setup):
        broker, _, _ = setup
        order = Order(symbol="688981.SH", side=BUY, quantity=150,
                      order_type=MARKETABLE_LIMIT, limit_price=60.0, board="STAR")
        result = broker.submit(order, market(symbol="688981.SH", board="STAR",
                                             last_price=55.0, previous_close=55.0))
        assert result.state == REJECTED

    def test_lunch_break_rejects_orders(self, setup):
        broker, _, _ = setup
        order = broker.submit(buy(), market(clock="12:00:00"))
        assert order.state == REJECTED
        assert "session phase" in order.reject_reason

    def test_closing_auction_is_tradable(self, setup):
        broker, _, _ = setup
        order = broker.submit(buy(), market(clock="14:58:00"))
        assert order.state == FILLED

    def test_sell_pays_stamp_duty_and_buy_does_not(self, setup):
        broker, portfolio, _ = setup
        broker.submit(buy(), market())
        assert all(f.stamp_duty == 0.0 for f in broker.fills)
        broker.close_session("2026-07-24")
        broker.submit(Order(symbol="600000.SH", side=SELL, quantity=1_000,
                            order_type=LIMIT, limit_price=9.00), market())
        assert broker.fills[-1].stamp_duty > 0

    def test_participation_cap_limits_the_fill(self, setup):
        broker, _, _ = setup
        order = broker.submit(buy(qty=1_000_000), market(session_volume=100_000))
        # 10% of 100,000 = 10,000 shares
        assert order.filled_quantity == pytest.approx(10_000.0)
        assert order.remaining > 0

    def test_resting_limit_below_market_does_not_fill(self, setup):
        broker, _, _ = setup
        order = broker.submit(
            Order(symbol="600000.SH", side=BUY, quantity=1_000,
                  order_type=LIMIT, limit_price=8.50), market(last_price=9.04))
        assert order.filled_quantity == 0.0
        assert order.is_open


class TestCashAndPositions:
    def test_insufficient_cash_rejects(self, setup):
        broker, portfolio, _ = setup
        portfolio.cash = 100.0
        order = broker.submit(buy(qty=10_000), market())
        assert order.state == REJECTED
        assert "available" in order.reject_reason

    def test_insufficient_sellable_rejects(self, setup):
        broker, portfolio, _ = setup
        broker.submit(buy(qty=1_000), market())
        broker.close_session("2026-07-24")
        order = broker.submit(
            Order(symbol="600000.SH", side=SELL, quantity=5_000,
                  order_type=LIMIT, limit_price=9.00), market())
        assert order.state == REJECTED

    def test_average_cost_includes_fees(self, setup):
        broker, portfolio, _ = setup
        broker.submit(buy(qty=10_000), market())
        position = portfolio.position("600000.SH")
        assert position.average_cost > 9.04

    def test_corporate_action_preserves_book_value(self, setup):
        broker, portfolio, _ = setup
        broker.submit(buy(qty=10_000), market())
        position = portfolio.position("600000.SH")
        book_before = position.total * position.average_cost
        broker.apply_corporate_action("600000.SH", share_ratio=2.0)
        assert position.total == 20_000.0
        assert position.total * position.average_cost == pytest.approx(book_before)

    def test_dividend_credits_cash(self, setup):
        broker, portfolio, _ = setup
        broker.submit(buy(qty=10_000), market())
        before = portfolio.cash
        broker.apply_corporate_action("600000.SH", cash_per_share=0.20)
        assert portfolio.cash == pytest.approx(before + 2_000.0)


class TestOrderStateMachine:
    def test_illegal_transition_refused(self):
        order = buy()
        order.transition("ACCEPTED")
        order.transition(FILLED)
        with pytest.raises(OrderStateError, match="illegal transition"):
            order.transition(CANCELLED)

    def test_cancel_marks_order_cancelled(self, setup):
        broker, _, _ = setup
        order = broker.submit(
            Order(symbol="600000.SH", side=BUY, quantity=1_000,
                  order_type=LIMIT, limit_price=8.50), market())
        broker.cancel(order.order_id)
        assert order.state == CANCELLED
        assert not order.is_open


class TestParentOrders:
    def test_twap_splits_evenly(self):
        parent = ParentOrder(symbol="600000.SH", side=BUY, quantity=10_000,
                             order_type=TWAP, slices=4)
        children = parent.schedule(reference_prices=[9.0, 9.1, 9.2, 9.3])
        assert len(children) == 4
        assert all(c.quantity == pytest.approx(2_500.0) for c in children)

    def test_vwap_weights_by_volume(self):
        parent = ParentOrder(symbol="600000.SH", side=BUY, quantity=10_000,
                             order_type=VWAP, slices=2)
        children = parent.schedule(reference_prices=[9.0, 9.1],
                                   volumes=[3_000.0, 1_000.0])
        assert children[0].quantity == pytest.approx(7_500.0)
        assert children[1].quantity == pytest.approx(2_500.0)

    def test_pov_underfills_a_thin_market(self):
        """POV legitimately under-fills when volume is thin; that is the policy."""
        parent = ParentOrder(symbol="600000.SH", side=BUY, quantity=100_000,
                             order_type=POV, slices=2, participation_rate=0.10)
        children = parent.schedule(reference_prices=[9.0, 9.1],
                                   volumes=[1_000.0, 1_000.0])
        assert sum(c.quantity for c in children) == pytest.approx(200.0)

    def test_children_are_bounded_limit_orders(self):
        parent = ParentOrder(symbol="600000.SH", side=BUY, quantity=1_000,
                             order_type=TWAP, slices=2)
        for child in parent.schedule(reference_prices=[9.0, 9.1]):
            assert child.order_type == MARKETABLE_LIMIT
            assert child.limit_price is not None


class TestLedgerIntegrity:
    def test_chain_verifies(self, setup):
        broker, _, ledger = setup
        broker.submit(buy(), market())
        assert ledger.verify()["valid"] is True

    def test_editing_an_event_breaks_the_chain(self, setup, tmp_path):
        broker, _, ledger = setup
        broker.submit(buy(), market())
        lines = ledger.path.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[1])
        record["payload"]["tampered"] = True
        lines[1] = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        ledger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert ledger.verify()["valid"] is False

    def test_deleting_an_event_breaks_the_chain(self, setup):
        broker, _, ledger = setup
        broker.submit(buy(), market())
        lines = ledger.path.read_text(encoding="utf-8").splitlines()
        del lines[1]
        ledger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert ledger.verify()["valid"] is False

    def test_unknown_event_type_refused(self, setup):
        _, _, ledger = setup
        with pytest.raises(lg.LedgerError, match="unknown event type"):
            ledger.append("ORDER_TELEPORTED", run_id="R1", portfolio_id="P1")

    def test_guarantee_is_stated_honestly(self, setup):
        _, _, ledger = setup
        assert "tamper-evident, not tamper-proof" in ledger.verify()["guarantee"]


class TestRecovery:
    def test_replay_reconstructs_cash_and_positions(self, setup):
        broker, portfolio, ledger = setup
        broker.submit(buy(qty=10_000), market())
        broker.close_session("2026-07-24")
        broker.submit(Order(symbol="600000.SH", side=SELL, quantity=2_000,
                            order_type=LIMIT, limit_price=9.00), market())

        recovered = rc.recover(ledger, portfolio_id="P1", initial_cash=1_000_000.0)
        assert recovered.portfolio.cash == pytest.approx(portfolio.cash)
        assert recovered.portfolio.position("600000.SH").total == pytest.approx(
            portfolio.position("600000.SH").total
        )

    def test_reconciliation_passes_after_replay(self, setup):
        broker, _, ledger = setup
        broker.submit(buy(), market())
        recovered = rc.recover(ledger, portfolio_id="P1", initial_cash=1_000_000.0)
        assert rc.reconcile(broker, recovered)["passed"] is True

    def test_reconciliation_detects_divergence(self, setup):
        broker, portfolio, ledger = setup
        broker.submit(buy(), market())
        recovered = rc.recover(ledger, portfolio_id="P1", initial_cash=1_000_000.0)
        portfolio.cash += 1_000.0  # an action that bypassed the ledger
        result = rc.reconcile(broker, recovered)
        assert result["passed"] is False
        assert any("cash mismatch" in p for p in result["problems"])

    def test_recovery_refuses_a_broken_chain(self, setup):
        broker, _, ledger = setup
        broker.submit(buy(), market())
        lines = ledger.path.read_text(encoding="utf-8").splitlines()
        del lines[0]
        ledger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(rc.RecoveryRefused, match="chain verification failed"):
            rc.recover(ledger, portfolio_id="P1", initial_cash=1_000_000.0)

    def test_kill_switch_state_survives_restart(self, setup):
        broker, _, ledger = setup
        broker.trigger_kill_switch("daily loss limit breached")
        recovered = rc.recover(ledger, portfolio_id="P1", initial_cash=1_000_000.0)
        assert recovered.killed is True
        assert "daily loss" in recovered.kill_reason

    def test_settlement_state_survives_restart(self, setup):
        broker, _, ledger = setup
        broker.submit(buy(qty=10_000), market())
        broker.close_session("2026-07-24")
        recovered = rc.recover(ledger, portfolio_id="P1", initial_cash=1_000_000.0)
        assert recovered.portfolio.sellable("600000.SH") == 10_000.0
        assert recovered.sessions_closed == 1

    def test_kill_switch_blocks_new_orders(self, setup):
        broker, _, _ = setup
        broker.trigger_kill_switch("test")
        order = broker.submit(buy(), market())
        assert order.state == REJECTED
        assert "kill switch" in order.reject_reason


class TestDeterminism:
    def test_identical_inputs_produce_identical_fills(self, tmp_path):
        def run(path):
            ledger = lg.EventLedger(path)
            portfolio = Portfolio(portfolio_id="P1", cash=1_000_000.0,
                                  initial_cash=1_000_000.0)
            broker = bk.PaperBroker(portfolio, ledger, run_id="R1")
            for _ in range(3):
                broker.submit(buy(qty=5_000), market())
            return [(f.symbol, f.quantity, f.price, f.notional) for f in broker.fills]

        first = run(tmp_path / "a.jsonl")
        second = run(tmp_path / "b.jsonl")
        assert first == second
        assert len(first) == 3

    def test_impact_grows_with_participation(self, tmp_path):
        def average_price(volume):
            ledger = lg.EventLedger(tmp_path / f"{volume}.jsonl")
            portfolio = Portfolio(portfolio_id="P", cash=10_000_000.0,
                                  initial_cash=10_000_000.0)
            broker = bk.PaperBroker(portfolio, ledger, run_id="R")
            order = broker.submit(buy(qty=10_000), market(session_volume=volume))
            return order.average_price

        thin = average_price(200_000.0)
        deep = average_price(50_000_000.0)
        assert thin > deep, "a larger participation fraction must cost more"
