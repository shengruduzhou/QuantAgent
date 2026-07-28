"""Risk engine: pre-trade, portfolio, operational, and latching kill switches.

The property that matters most is that a rejection is final. Every other test
here names a specific limit that, unchecked, turns a paper book into a
misleading one.
"""

from __future__ import annotations

import pytest

from quantagent.paper import ledger as lg
from quantagent.paper import risk as rk
from quantagent.paper.orders import BUY, MARKETABLE_LIMIT, SELL, Order
from quantagent.paper.portfolio import Portfolio


@pytest.fixture
def engine():
    return rk.RiskEngine(rk.RiskLimits())


@pytest.fixture
def portfolio():
    return Portfolio(portfolio_id="P1", cash=1_000_000.0, initial_cash=1_000_000.0)


def order(qty=10_000, price=9.00, side=BUY, **kw):
    return Order(symbol="600000.SH", side=side, quantity=qty,
                 order_type=MARKETABLE_LIMIT, limit_price=price, **kw)


class TestRejectionIsFinal:
    def test_enforce_raises_and_offers_no_override(self, engine, portfolio):
        decision = engine.check_order(order(qty=1_000_000), portfolio,
                                      reference_price=9.00)
        assert not decision.approved
        with pytest.raises(rk.RiskRejection, match="cannot be overridden"):
            engine.enforce(decision)

    def test_decision_advertises_no_override_path(self, engine, portfolio):
        decision = engine.check_order(order(), portfolio, reference_price=9.00)
        assert decision.to_dict()["override_available"] is False

    def test_enforce_signature_has_no_force_parameter(self):
        import inspect
        params = set(inspect.signature(rk.RiskEngine.enforce).parameters)
        assert params == {"self", "decision"}


class TestPreTradeRisk:
    def test_order_notional_limit(self, engine, portfolio):
        decision = engine.check_order(order(qty=100_000), portfolio,
                                      reference_price=9.00)
        assert "order_notional" in decision.failed

    def test_order_share_limit(self, portfolio):
        engine = rk.RiskEngine(rk.RiskLimits(max_order_shares=1_000,
                                             max_order_notional=1e12))
        decision = engine.check_order(order(qty=10_000), portfolio,
                                      reference_price=9.00)
        assert "order_shares" in decision.failed

    def test_fat_finger_price_rejected(self, engine, portfolio):
        decision = engine.check_order(order(qty=100, price=20.0), portfolio,
                                      reference_price=9.00)
        assert "fat_finger" in decision.failed

    def test_stale_quote_rejected(self, engine, portfolio):
        decision = engine.check_order(order(qty=100), portfolio,
                                      reference_price=9.00,
                                      quote_age_seconds=10_000)
        assert "stale_data" in decision.failed

    def test_participation_limit(self, engine, portfolio):
        decision = engine.check_order(order(qty=10_000), portfolio,
                                      reference_price=9.00, session_volume=20_000)
        assert "participation" in decision.failed

    def test_single_name_concentration(self, portfolio):
        engine = rk.RiskEngine(rk.RiskLimits(max_single_name_weight=0.01,
                                             max_order_notional=1e12))
        decision = engine.check_order(order(qty=10_000), portfolio,
                                      reference_price=9.00)
        assert "single_name_weight" in decision.failed

    def test_industry_concentration(self, portfolio):
        engine = rk.RiskEngine(rk.RiskLimits(max_industry_weight=0.05,
                                             max_order_notional=1e12,
                                             max_single_name_weight=1.0))
        decision = engine.check_order(
            order(qty=10_000), portfolio, reference_price=9.00,
            industry="银行", industry_weights={"银行": 0.04})
        assert "industry_weight" in decision.failed

    def test_insufficient_cash(self, engine):
        poor = Portfolio(portfolio_id="P", cash=100.0, initial_cash=100.0)
        decision = engine.check_order(order(qty=1_000), poor, reference_price=9.00)
        assert "cash_available" in decision.failed

    def test_sell_without_settled_shares(self, engine, portfolio):
        decision = engine.check_order(order(qty=1_000, side=SELL), portfolio,
                                      reference_price=9.00)
        assert "position_available" in decision.failed

    def test_duplicate_order_id_rejected(self, engine, portfolio):
        first = order(qty=100)
        assert engine.check_order(first, portfolio, reference_price=9.00).approved
        again = engine.check_order(first, portfolio, reference_price=9.00)
        assert "duplicate_order" in again.failed

    def test_unapproved_model_rejected(self, engine, portfolio):
        decision = engine.check_order(order(qty=100), portfolio,
                                      reference_price=9.00, model_approved=False)
        assert "model_approved" in decision.failed

    def test_unapproved_dataset_rejected(self, engine, portfolio):
        decision = engine.check_order(order(qty=100), portfolio,
                                      reference_price=9.00, dataset_approved=False)
        assert "dataset_approved" in decision.failed

    def test_daily_turnover_limit(self, portfolio):
        engine = rk.RiskEngine(rk.RiskLimits(max_daily_turnover=0.01,
                                             max_order_notional=1e12,
                                             max_single_name_weight=1.0))
        decision = engine.check_order(order(qty=10_000), portfolio,
                                      reference_price=9.00)
        assert "daily_turnover" in decision.failed

    def test_clean_order_approved(self, engine, portfolio):
        decision = engine.check_order(order(qty=1_000), portfolio,
                                      reference_price=9.00, session_volume=10_000_000)
        assert decision.approved, decision.failed


class TestPortfolioRisk:
    def test_daily_loss_triggers_kill_switch(self, portfolio):
        engine = rk.RiskEngine(rk.RiskLimits(max_daily_loss=1_000.0))
        engine.check_portfolio(portfolio, {})
        portfolio.cash -= 50_000.0
        decision = engine.check_portfolio(portfolio, {})
        assert "daily_loss" in decision.failed
        assert engine.kill_switch.is_triggered(rk.SCOPE_PORTFOLIO)

    def test_drawdown_triggers_kill_switch(self, portfolio):
        engine = rk.RiskEngine(rk.RiskLimits(max_drawdown=0.05,
                                             max_daily_loss=1e12))
        engine.check_portfolio(portfolio, {})
        portfolio.cash *= 0.5
        decision = engine.check_portfolio(portfolio, {})
        assert "drawdown" in decision.failed
        assert engine.kill_switch.is_triggered(rk.SCOPE_PORTFOLIO)

    def test_gross_exposure_limit(self, portfolio):
        engine = rk.RiskEngine(rk.RiskLimits(max_gross_exposure=0.05))
        position = portfolio.position("600000.SH")
        position.total = 100_000.0
        position.average_cost = 9.0
        decision = engine.check_portfolio(portfolio, {"600000.SH": 9.0})
        assert "gross_exposure" in decision.failed

    def test_industry_exposure_limit(self, portfolio):
        engine = rk.RiskEngine(rk.RiskLimits(max_industry_weight=0.01,
                                             max_gross_exposure=10.0,
                                             max_single_name_weight=10.0))
        position = portfolio.position("600000.SH")
        position.total = 10_000.0
        position.average_cost = 9.0
        decision = engine.check_portfolio(
            portfolio, {"600000.SH": 9.0}, industry_map={"600000.SH": "银行"})
        assert any(c.startswith("industry:") for c in decision.failed)


class TestOperationalRisk:
    def test_stale_heartbeat(self, engine, portfolio):
        decision = engine.check_operational(portfolio, heartbeat_age_seconds=9_999)
        assert "heartbeat" in decision.failed

    def test_broken_ledger_triggers_global_kill(self, engine, portfolio):
        decision = engine.check_operational(portfolio, ledger_valid=False)
        assert "ledger_chain" in decision.failed
        assert engine.kill_switch.is_triggered(rk.SCOPE_GLOBAL)

    def test_reconciliation_failure_triggers_global_kill(self, engine, portfolio):
        decision = engine.check_operational(portfolio, reconciliation_passed=False)
        assert "reconciliation" in decision.failed
        assert engine.kill_switch.is_triggered(rk.SCOPE_GLOBAL)

    def test_insufficient_disk(self, engine, portfolio):
        decision = engine.check_operational(portfolio, disk_free_bytes=1024)
        assert "disk_space" in decision.failed

    def test_clock_drift(self, engine, portfolio):
        decision = engine.check_operational(portfolio, clock_drift_seconds=60.0)
        assert "clock_drift" in decision.failed

    def test_schema_mismatch(self, engine, portfolio):
        decision = engine.check_operational(portfolio, schema_matches=False)
        assert "schema" in decision.failed

    def test_repeated_rejections(self, engine, portfolio):
        decision = engine.check_operational(portfolio, consecutive_rejections=99)
        assert "repeated_rejections" in decision.failed

    def test_negative_position_detected(self, engine, portfolio):
        portfolio.position("600000.SH").total = -100.0
        decision = engine.check_operational(portfolio)
        assert "no_negative_positions" in decision.failed

    def test_healthy_system_approved(self, engine, portfolio):
        decision = engine.check_operational(portfolio, disk_free_bytes=1 << 40)
        assert decision.approved, decision.failed


class TestKillSwitch:
    def test_global_halts_every_scope(self):
        switch = rk.KillSwitch()
        switch.trigger(rk.SCOPE_GLOBAL, "emergency")
        assert switch.is_triggered(rk.SCOPE_ORDER)
        assert switch.is_triggered(rk.SCOPE_STRATEGY, "alpha")
        assert switch.is_triggered(rk.SCOPE_PORTFOLIO)

    def test_strategy_scope_is_isolated(self):
        switch = rk.KillSwitch()
        switch.trigger(rk.SCOPE_STRATEGY, "bad signals", key="alpha")
        assert switch.is_triggered(rk.SCOPE_STRATEGY, "alpha")
        assert not switch.is_triggered(rk.SCOPE_STRATEGY, "beta")

    def test_unknown_scope_rejected(self):
        with pytest.raises(ValueError, match="unknown kill-switch scope"):
            rk.KillSwitch().trigger("EVERYTHING", "why not")

    def test_clearing_requires_human_confirmation(self):
        switch = rk.KillSwitch()
        switch.trigger(rk.SCOPE_GLOBAL, "test")
        with pytest.raises(rk.RiskRejection, match="human confirmation"):
            switch.clear(rk.SCOPE_GLOBAL)
        assert switch.is_triggered(rk.SCOPE_GLOBAL)
        assert switch.clear(rk.SCOPE_GLOBAL, human_confirmation=True) is True
        assert not switch.is_triggered(rk.SCOPE_GLOBAL)

    def test_triggered_switch_blocks_new_orders(self, portfolio):
        engine = rk.RiskEngine(rk.RiskLimits())
        engine.kill_switch.trigger(rk.SCOPE_GLOBAL, "halt")
        decision = engine.check_order(order(qty=100), portfolio, reference_price=9.00)
        assert "kill_switch" in decision.failed

    def test_decisions_are_written_to_the_ledger(self, tmp_path, portfolio):
        ledger = lg.EventLedger(tmp_path / "l.jsonl")
        engine = rk.RiskEngine(rk.RiskLimits(), event_ledger=ledger, run_id="R")
        engine.check_order(order(qty=1_000), portfolio, reference_price=9.00)
        engine.check_order(order(qty=999_999), portfolio, reference_price=9.00)
        kinds = [e.event_type for e in ledger.read()]
        assert lg.RISK_APPROVED in kinds
        assert lg.RISK_REJECTED in kinds
        assert ledger.verify()["valid"] is True
