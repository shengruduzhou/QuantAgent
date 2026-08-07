"""Paper's economic state must be reconstructable from the canonical ledger.

Round 4 unified paper's *rules* with the canonical state model. That is not the
same as unifying its *record of account*: paper still computed cash and
positions in its own Portfolio and logged to its own EventLedger, so a paper run
could not be rebuilt from the canonical record.

These prove the economic figures agree, and that the canonical ledger alone is
sufficient to recover them.
"""

from __future__ import annotations

import pytest

from quantagent.domain.ledger import CanonicalLedger, LedgerCorruption
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import OrderStatus
from quantagent.paper import ledger as lg
from quantagent.paper.broker import MarketSnapshot, PaperBroker
from quantagent.paper.orders import BUY, Order
from quantagent.paper.portfolio import Portfolio

SYMBOL = "600000.SH"
INITIAL = 1_000_000.0


@pytest.fixture
def paper(tmp_path):
    portfolio = Portfolio(portfolio_id="p1", cash=INITIAL, initial_cash=INITIAL)
    broker = PaperBroker(
        portfolio,
        lg.EventLedger(tmp_path / "paper.jsonl"),
        run_id="run_1",
        canonical_ledger_path=str(tmp_path / "canonical.jsonl"),
        lineage=Lineage(research_id="res_1", strategy_version_id="sv_1", run_id="run_1"),
    )
    return broker, portfolio, tmp_path / "canonical.jsonl"


def _market(price: float = 10.0, date: str = "2026-08-04") -> MarketSnapshot:
    return MarketSnapshot(
        symbol=SYMBOL, trade_date=date, last_price=price,
        previous_close=price, session_volume=1e8,
    )


def _buy(quantity: float = 1_000, limit: float = 10.05) -> Order:
    return Order(symbol=SYMBOL, side=BUY, quantity=quantity, limit_price=limit)


def test_paper_writes_every_order_to_the_canonical_ledger(paper):
    broker, _, _ = paper

    broker.submit(_buy(), _market())

    assert len(broker.canonical) > 0
    assert broker.canonical.verify()["valid"]
    assert len(broker.book.orders()) == 1


def test_paper_cash_reconstructs_exactly_from_the_canonical_ledger(paper):
    broker, portfolio, path = paper
    broker.submit(_buy(), _market())

    _, account = CanonicalLedger(path).replay(initial_cash=INITIAL)

    assert account.cash == pytest.approx(portfolio.cash, abs=1e-6), (
        "paper's own cash and the canonical replay must not disagree"
    )


def test_paper_position_reconstructs_from_the_canonical_ledger(paper):
    broker, portfolio, path = paper
    broker.submit(_buy(quantity=1_000), _market())

    _, account = CanonicalLedger(path).replay(initial_cash=INITIAL)

    assert account.position(SYMBOL) == 1_000
    assert SYMBOL in portfolio.positions


def test_paper_order_state_and_quantities_survive_replay(paper):
    broker, _, path = paper
    broker.submit(_buy(quantity=1_000), _market())

    book, _ = CanonicalLedger(path).replay(initial_cash=INITIAL)

    order = book.orders()[0]
    assert order.status is OrderStatus.FILLED
    assert order.cumulative_quantity == 1_000
    assert order.leaves_quantity == 0
    assert order.last_quantity == 1_000


def test_paper_replay_is_hash_stable(paper):
    broker, _, path = paper
    broker.submit(_buy(), _market())

    hashes = {
        CanonicalLedger(path).replay(initial_cash=INITIAL)[1].content_hash()
        for _ in range(3)
    }

    assert len(hashes) == 1


def test_paper_orders_carry_the_runs_lineage(paper):
    broker, _, path = paper
    broker.submit(_buy(), _market())

    book, _ = CanonicalLedger(path).replay(initial_cash=INITIAL)

    order = book.orders()[0]
    assert order.lineage.run_id == "run_1"
    assert order.lineage.strategy_version_id == "sv_1"
    assert order.lineage.signal_id is not None
    assert order.lineage.order_intent_id is not None


def test_a_paper_rejection_reaches_the_canonical_ledger(paper):
    """A refusal must be part of the record of account, not only a paper log."""
    broker, _, path = paper
    # Far more than the account can afford: rejected at fill time.
    broker.submit(_buy(quantity=10_000_000, limit=10.05), _market())

    book, _ = CanonicalLedger(path).replay(initial_cash=INITIAL)

    rejected = [o for o in book.orders() if o.status is OrderStatus.REJECTED]
    assert rejected, "the rejection must be reconstructable"
    assert rejected[0].reason
    assert rejected[0].cumulative_quantity == 0


def test_a_corrupt_canonical_ledger_blocks_paper_projection(paper):
    broker, _, path = paper
    broker.submit(_buy(), _market())
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"reason": null', '"reason": "tampered"', 1)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(LedgerCorruption):
        CanonicalLedger(path).replay(initial_cash=INITIAL)
