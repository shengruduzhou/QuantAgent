from __future__ import annotations

import pytest

from quantagent.domain.ledger import CanonicalLedger, mirror_fill, mirror_open
from quantagent.domain.lineage import Lineage
from quantagent.domain.orders import OrderBook, OrderIntent, Side
from quantagent.paper.recovery import RecoveryRefused, recover_from_canonical


def _filled_buy(path, *, trade_date: str = "2026-08-07") -> None:
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
    order = mirror_open(book, ledger, intent, trade_date=trade_date)
    # PaperBroker normally emits risk/submitted before fill. For the replay
    # accounting contract, mirror_fill applies the canonical fill lifecycle.
    mirror_fill(
        book,
        ledger,
        order.order_id,
        execution_id="fill-1",
        quantity=100,
        price=10.0,
        commission=5.0,
        stamp_duty=0.0,
        transfer_fee=0.0,
        trade_date=trade_date,
    )


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
