"""What the paper venue hands the risk engine, and what happens when it cannot.

`PaperBroker._validate` used to pass only `reference_price` and
`session_volume`. So the industry limit could never fire (no industry, no
weights), the stale-quote check always measured 0 s, model/dataset approval
were recorded as passed by default, and the book was valued with the order's
own price only — which raised `UnpriceablePosition` on the second name held.
"""

from __future__ import annotations

import quantagent.paper.orders as po
from quantagent.paper import ledger as lg
from quantagent.paper.broker import BrokerConfig, MarketSnapshot, PaperBroker
from quantagent.paper.portfolio import Portfolio
from quantagent.paper.risk import (
    SCOPE_GLOBAL,
    SCOPE_PORTFOLIO,
    RiskEngine,
    RiskLimits,
)

DAY = "2026-08-18"
A, B = "600000.SH", "600036.SH"


class SpyRisk(RiskEngine):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[dict] = []

    def check_order(self, order, portfolio, **kwargs):
        self.calls.append(dict(kwargs))
        return super().check_order(order, portfolio, **kwargs)


def broker(tmp_path, *, limits=None, industry_map=None, engine=None) -> PaperBroker:
    return PaperBroker(
        Portfolio(portfolio_id="p", cash=1_000_000.0, initial_cash=1_000_000.0),
        lg.EventLedger(tmp_path / "operational.jsonl"),
        run_id="t",
        config=BrokerConfig(participation_cap=0.10),
        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),
        risk_engine=engine or RiskEngine(limits or RiskLimits(max_participation=1.0), run_id="t"),
        industry_map=industry_map if industry_map is not None else {A: "bank", B: "bank"},
    )


def snapshot(symbol: str = A, *, day: str = DAY, clock: str = "10:00:00",
             price: float = 10.0) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol, trade_date=day, last_price=price, previous_close=price,
        session_volume=1e8, board="SH_Main", clock=clock,
    )


def buy(symbol: str = A, qty: float = 5_000, price: float = 10.0) -> po.Order:
    return po.Order(symbol=symbol, side=po.BUY, quantity=qty, limit_price=price,
                    board="SH_Main")


def test_every_measured_input_reaches_check_order(tmp_path) -> None:
    spy = SpyRisk(RiskLimits(max_participation=1.0), run_id="t")
    venue = broker(tmp_path, engine=spy)
    venue.submit(buy(), snapshot())

    passed = spy.calls[-1]
    assert passed["industry"] == "bank"
    assert passed["industry_weights"] == {}
    assert passed["prices"] == {A: 10.0}
    assert passed["quote_age_seconds"] == 0.0
    # Not measured by the venue, so not claimed as approved either.
    assert "model_approved" not in passed and "dataset_approved" not in passed


def test_the_second_name_is_valued_against_the_whole_book(tmp_path) -> None:
    venue = broker(tmp_path)
    first = venue.submit(buy(A), snapshot(A))
    second = venue.submit(buy(B), snapshot(B))

    assert first.state == po.FILLED
    assert second.state == po.FILLED
    decision = venue.last_order_decision
    assert decision.approved
    assert {c.name for c in decision.checks} >= {"book_priceable", "industry_weight"}


def test_an_unpriceable_held_position_is_a_rejection_not_an_exception(tmp_path) -> None:
    venue = broker(tmp_path)
    held = venue.portfolio.position("601398.SH")  # held, never quoted to this venue
    held.total = held.sellable = 1_000.0
    held.average_cost = 5.0

    order = venue.submit(buy(B), snapshot(B))

    assert order.state == po.REJECTED
    assert "book_priceable" in order.reject_reason
    failed = {c.name: c for c in venue.last_order_decision.checks if not c.passed}
    assert failed["book_priceable"].measured == ["601398.SH"]
    book = venue.canonical.replay_book()
    assert [o.status.value for o in book.orders()] == ["REJECTED"]


def test_a_buy_without_an_industry_is_refused_as_unmeasured(tmp_path) -> None:
    venue = broker(tmp_path, industry_map={})
    order = venue.submit(buy(), snapshot())

    assert order.state == po.REJECTED
    assert "industry_unmeasured" in order.reject_reason


def test_an_industry_limit_of_one_is_the_explicit_opt_out(tmp_path) -> None:
    venue = broker(tmp_path, industry_map={},
                   limits=RiskLimits(max_participation=1.0, max_industry_weight=1.0))
    order = venue.submit(buy(), snapshot())

    assert order.state == po.FILLED
    assert not any(c.name.startswith("industry") for c in venue.last_order_decision.checks)


def test_a_sell_is_not_blocked_by_a_missing_industry(tmp_path) -> None:
    venue = broker(tmp_path, industry_map={})
    held = venue.portfolio.position(A)
    held.total = held.sellable = 1_000.0
    held.average_cost = 10.0
    order = venue.submit(
        po.Order(symbol=A, side=po.SELL, quantity=1_000, limit_price=10.0), snapshot()
    )

    assert order.state == po.FILLED


def test_a_quote_older_than_the_venue_clock_is_stale(tmp_path) -> None:
    venue = broker(tmp_path)
    venue.submit(buy(A, qty=100), snapshot(A, clock="14:00:00"))
    late = venue.submit(buy(B, qty=100), snapshot(B, clock="13:00:00"))

    assert late.state == po.REJECTED
    assert "stale_data" in late.reject_reason
    stale = next(c for c in venue.last_order_decision.checks if c.name == "stale_data")
    assert stale.measured == 3600.0


def test_portfolio_kill_switch_is_reduce_only(tmp_path) -> None:
    venue = broker(tmp_path)
    held = venue.portfolio.position(A)
    held.total = held.sellable = 1_000.0
    held.average_cost = 10.0
    venue.risk_engine.kill_switch.trigger(SCOPE_PORTFOLIO, "drawdown breach")

    refused = venue.submit(buy(B, qty=100), snapshot(B))
    allowed = venue.submit(
        po.Order(symbol=A, side=po.SELL, quantity=1_000, limit_price=10.0), snapshot(A)
    )

    assert refused.state == po.REJECTED and "kill_switch" in refused.reject_reason
    assert allowed.state == po.FILLED


def test_global_kill_switch_blocks_sells_too(tmp_path) -> None:
    venue = broker(tmp_path)
    held = venue.portfolio.position(A)
    held.total = held.sellable = 1_000.0
    held.average_cost = 10.0
    venue.risk_engine.kill_switch.trigger(SCOPE_GLOBAL, "operator halt")

    order = venue.submit(
        po.Order(symbol=A, side=po.SELL, quantity=1_000, limit_price=10.0), snapshot(A)
    )

    assert order.state == po.REJECTED and "kill_switch" in order.reject_reason
