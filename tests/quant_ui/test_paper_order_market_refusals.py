"""`/api/paper/orders` refusals for missing or unusable market data.

The production container builds `PaperOrderService` without a market source, so
every drained order is refused `market_data_unavailable`. That refusal used to
happen inside the venue, after the order manager had already written CREATED +
RISK_APPROVED + SUBMITTED, so the record of account showed a working order that
no venue held. Market data is now resolved before the order manager opens a
canonical order; a refusal that still happens inside the venue terminates the
canonical order as REJECTED.
"""

from __future__ import annotations

from quantagent.domain.ledger import CanonicalLedger
from quantagent.paper.broker import MarketSnapshot
from services.quant_api.services.paper_orders import (
    MARKET_DATA_INVALID,
    MARKET_DATA_UNAVAILABLE,
    PaperOrderService,
)

SYMBOL = "600000.SH"
SESSION = "2025-06-03"
WORKING = {"SUBMITTED", "ACCEPTED", "PARTIALLY_FILLED", "APPROVED", "PENDING_RISK"}


def _request(key: str = "k1") -> dict:
    return {
        "idempotencyKey": key, "runId": "r", "symbol": SYMBOL, "side": "BUY",
        "quantity": 1000, "limitPrice": 10.0, "tradeDate": SESSION, "signalId": key,
    }


def test_refused_for_missing_market_data_leaves_no_working_order(tmp_path) -> None:
    service = PaperOrderService(tmp_path / "paper_orders")  # as container.py builds it
    try:
        service.submit(_request())
        result = service.drain()[0]
        orders = service.orders()
    finally:
        service.close()

    assert result["reason"] == MARKET_DATA_UNAVAILABLE
    assert result["venueStatus"] == "REFUSED"
    assert [o for o in orders if o["status"] in WORKING] == []


def test_nan_session_volume_is_refused_not_filled_uncapped(tmp_path) -> None:
    """An unmeasured volume used to fill 15,000 shares at zero impact."""
    def nan_volume(symbol: str, trade_date: str) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=symbol, trade_date=trade_date, last_price=10.0, previous_close=10.0,
            session_volume=float("nan"), board="SH_Main",
        )

    service = PaperOrderService(tmp_path / "p", market_source=nan_volume)
    try:
        service.submit(_request() | {"quantity": 15_000})
        result = service.drain()[0]
        cash = service.broker.portfolio.cash
        orders = service.orders()
    finally:
        service.close()

    assert result["reason"] == MARKET_DATA_INVALID
    assert result["filledQuantity"] in (None, 0)
    assert cash == 1_000_000.0
    assert [o for o in orders if o["status"] in WORKING] == []


def test_market_data_vanishing_inside_the_venue_terminates_the_canonical_order(tmp_path) -> None:
    """The pre-resolution saw a snapshot; the venue's own lookup did not."""
    calls = {"n": 0}

    def flaky(symbol: str, trade_date: str) -> MarketSnapshot | None:
        calls["n"] += 1
        if calls["n"] > 1:
            return None
        return MarketSnapshot(
            symbol=symbol, trade_date=trade_date, last_price=10.0, previous_close=10.0,
            session_volume=1e8, board="SH_Main",
        )

    service = PaperOrderService(tmp_path / "p", market_source=flaky)
    try:
        service.submit(_request())
        result = service.drain()[0]
    finally:
        service.close()

    book = CanonicalLedger(tmp_path / "p" / "canonical.jsonl").replay_book()
    assert result["venueStatus"] == "REJECTED"
    assert result["reason"] == MARKET_DATA_UNAVAILABLE
    assert [(o.status.value, o.leaves_quantity) for o in book.orders()] == [("REJECTED", 0)]
