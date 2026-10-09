"""A long-lived HTTP paper service must settle T+1 when the session rolls.

The service never called `close_session`, so shares bought through
`/api/paper/orders` stayed pending-settlement forever in a running process and
could only be sold after a restart rebuilt sellability from the ledger
(reported by ENG_RISK, round 29).
"""

from __future__ import annotations

from quantagent.paper.broker import MarketSnapshot
from quantagent.paper.risk import RiskLimits
from services.quant_api.services.paper_orders import PaperOrderService

SYMBOL = "600000.SH"


def _market(symbol: str, trade_date: str) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol, trade_date=trade_date, last_price=10.0, previous_close=10.0,
        session_volume=50_000_000.0, board="SH_Main",
    )


def _order(key: str, side: str, trade_date: str) -> dict:
    return {
        "idempotencyKey": key, "runId": "r", "symbol": SYMBOL, "side": side,
        "quantity": 1000, "limitPrice": 10.0, "tradeDate": trade_date, "signalId": key,
    }


def test_shares_bought_yesterday_are_sellable_today_without_a_restart(tmp_path) -> None:
    service = PaperOrderService(
        tmp_path / "p",
        market_source=_market,
        risk_limits=RiskLimits(max_participation=1.0, max_industry_weight=1.0),
    )
    try:
        service.submit(_order("buy", "BUY", "2025-06-03"))
        bought = service.drain()[0]
        service.submit(_order("same-day-sell", "SELL", "2025-06-03"))
        same_day = service.drain()[0]
        service.submit(_order("next-day-sell", "SELL", "2025-06-04"))
        next_day = service.drain()[0]
    finally:
        service.close()

    assert bought["venueStatus"] == "FILLED"
    assert same_day["venueStatus"] == "REJECTED"  # T+1: not sellable on the buy day
    assert next_day["venueStatus"] == "FILLED", next_day
