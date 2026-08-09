from __future__ import annotations

import pytest

from quantagent.domain.ledger import CanonicalLedger
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import Fill, OrderBook, OrderEventType, OrderIntent, Side
from quantagent.paper.recovery import RecoveryRefused, recover_from_canonical


def _filled_buy(path, *, trade_date: str = "2026-08-07") -> None:
    """Write one fully-filled canonical buy using only public ledger/order APIs."""

    ledger = CanonicalLedger(path)
    book = OrderBook()
    intent = OrderIntent.create(
        symbol="600000.SH",
        side=Side.BUY,
        quantity=100,
        trade_date=trade_date,
        lineage=Lineage(run_id="paper-test", strategy_version_id="v1"),
        reference_price=10.0,
    )
    order = book.open(intent)
    ledger.append(book.history_of(order.order_id)[-1], trade_date=trade_date, intent=intent)
    for event_type in (
        OrderEventType.RISK_APPROVED,
        OrderEventType.SUBMITTED,
        OrderEventType.ACCEPTED,
    ):
        book.apply(order.order_id, event_type)
        ledger.append(book.history_of(order.order_id)[-1], trade_date=trade_date)

    fill = Fill(
        execution_id="fill-1",
        order_id=order.order_id,
        symbol="600000.SH",
        side=Side.BUY,
        quantity=100,
        price=10.0,
        reference_price=10.0,
        commission=5.0,
        stamp_duty=0.0,
        transfer_fee=0.0,
        filled_at=f"{trade_date}T14:59:00+08:00",
        lineage=intent.lineage.derive(execution_id="fill-1"),
    )
    book.apply(order.order_id, OrderEventType.FILLED, fill=fill)
    ledger.append(book.history_of(order.order_id)[-1], trade_date=trade_date)


def test_same_session_recovery_keeps_new_buy_unsellable(tmp_path) -> None:
    path = tmp_path / "canonical.jsonl"
    _filled_buy(path)
    recovered = recover_from_canonical(
        str(path),
        portfolio_id="paper",
        initial_cash=100_000.0,
        as_of_session="2026-08-07",
    )
    position = recovered.portfolio.positions["600000.SH"]
    assert position.total == 100
    assert position.sellable == 0
    assert position.pending_settlement == 100


def test_next_session_recovery_makes_prior_buy_sellable(tmp_path) -> None:
    path = tmp_path / "canonical.jsonl"
    _filled_buy(path)
    recovered = recover_from_canonical(
        str(path),
        portfolio_id="paper",
        initial_cash=100_000.0,
        as_of_session="2026-08-10",
    )
    position = recovered.portfolio.positions["600000.SH"]
    assert position.total == 100
    assert position.sellable == 100
    assert position.pending_settlement == 0


def test_recovery_cannot_rewind_full_ledger_to_before_latest_economic_event(tmp_path) -> None:
    path = tmp_path / "canonical.jsonl"
    _filled_buy(path, trade_date="2026-08-07")
    with pytest.raises(RecoveryRefused, match="precedes latest canonical economic session"):
        recover_from_canonical(
            str(path),
            portfolio_id="paper",
            initial_cash=100_000.0,
            as_of_session="2026-08-06",
        )
