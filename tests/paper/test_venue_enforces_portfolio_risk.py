"""The paper venue must enforce portfolio-level limits, not just instrument rules.

Round 21 / R2 (risk) finding.  `quantagent.paper.risk` — the most complete risk
engine in the repository (single-name weight, industry concentration, gross
exposure, daily turnover, daily loss, drawdown, participation, fat-finger,
scoped kill switch) — was imported by nothing except its own test file, while
three production sites constructed `PaperBroker`.  `PaperBroker._validate`
covered only instrument-level rules: kill switch, session phase, tradability,
lot size, price limits, T+1 sellability.

So a 50% single-name paper position executed with nothing objecting.  The
limits existed, were deterministic, carried no override path — and were never
called.
"""

from __future__ import annotations

import quantagent.paper.orders as po
from quantagent.paper import ledger as lg
from quantagent.paper.broker import BrokerConfig, MarketSnapshot, PaperBroker
from quantagent.paper.portfolio import Portfolio
from quantagent.paper.risk import RiskEngine, RiskLimits


def _broker(tmp_path, *, risk_engine=None, cash: float = 1_000_000.0) -> PaperBroker:
    return PaperBroker(
        Portfolio(portfolio_id="p", cash=cash, initial_cash=cash),
        lg.EventLedger(tmp_path / "operational.jsonl"),
        run_id="test",
        config=BrokerConfig(participation_cap=0.10),
        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),
        risk_engine=risk_engine,
    )


def _market(**overrides) -> MarketSnapshot:
    base = dict(
        symbol="600000.SH", trade_date="2026-08-18", last_price=10.0,
        previous_close=10.0, session_volume=10_000_000.0, board="SH_Main",
    )
    base.update(overrides)
    return MarketSnapshot(**base)


def _order(quantity: float) -> po.Order:
    return po.Order(
        symbol="600000.SH", side=po.BUY, quantity=quantity,
        order_type=po.LIMIT, limit_price=10.0, board="SH_Main",
    )


def test_venue_without_a_risk_engine_reports_that_it_has_none(tmp_path) -> None:
    """"No portfolio risk applied" must be readable, never inferred from silence."""
    broker = _broker(tmp_path)
    assert broker.risk_engine_attached is False


def test_venue_with_a_risk_engine_reports_it(tmp_path) -> None:
    broker = _broker(tmp_path, risk_engine=RiskEngine(run_id="test"))
    assert broker.risk_engine_attached is True


def test_concentrated_order_is_rejected_by_the_single_name_limit(tmp_path) -> None:
    broker = _broker(tmp_path, risk_engine=RiskEngine(run_id="test"))
    # 50,000 shares at 10.00 = 500,000 on a 1,000,000 book = 50% of one name,
    # far past the 10% default.
    order = broker.submit(_order(50_000), _market())

    assert order.state == po.REJECTED
    assert "single_name_weight" in (order.reject_reason or "")
    assert broker.portfolio.position("600000.SH").total == 0


def test_the_same_order_fills_when_no_risk_engine_is_attached(tmp_path) -> None:
    """Pins what production was doing before: the limit simply did not exist."""
    broker = _broker(tmp_path)
    order = broker.submit(_order(50_000), _market())

    assert order.state != po.REJECTED
    assert broker.portfolio.position("600000.SH").total > 0


def test_a_compliant_order_still_passes(tmp_path) -> None:
    broker = _broker(tmp_path, risk_engine=RiskEngine(run_id="test"))
    # 5,000 shares at 10.00 = 50,000 = 5% of the book.
    order = broker.submit(_order(5_000), _market())

    assert order.state != po.REJECTED
    assert broker.portfolio.position("600000.SH").total > 0


def test_declared_limits_are_honoured_over_the_defaults(tmp_path) -> None:
    """A run that wants a concentrated book must declare it, and then gets it."""
    engine = RiskEngine(
        limits=RiskLimits(max_single_name_weight=1.0, max_order_notional=1e9),
        run_id="test",
    )
    broker = _broker(tmp_path, risk_engine=engine)
    order = broker.submit(_order(50_000), _market())

    assert order.state != po.REJECTED


def test_instrument_rules_still_run_before_portfolio_rules(tmp_path) -> None:
    """A suspended name is refused on tradability, not on a portfolio limit."""
    broker = _broker(tmp_path, risk_engine=RiskEngine(run_id="test"))
    order = broker.submit(_order(50_000), _market(is_suspended=True))

    assert order.state == po.REJECTED
    assert "single_name_weight" not in (order.reject_reason or "")
