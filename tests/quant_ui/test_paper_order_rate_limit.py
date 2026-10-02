"""`max_orders_per_symbol_per_day` on the production submission path.

`OrderManagerConfig.max_orders_per_symbol_per_day` (5) was read only inside
`OrderManager.reconcile()`. Both production paper paths call `submit_orders()`,
so eight orders for one symbol in one session all reached the venue. The limit
is now applied to every non-forensic submission against the canonical
same-session count, and a refusal is recorded on the ledger.
"""

from __future__ import annotations

from quantagent.domain.ledger import CanonicalLedger
from quantagent.paper.broker import MarketSnapshot
from services.quant_api.services.paper_orders import PaperOrderService

SYM = "600000.SH"
D1, D2 = "2025-06-03", "2025-06-04"


def market(symbol: str, trade_date: str) -> MarketSnapshot | None:
    if symbol != SYM:
        return None
    return MarketSnapshot(symbol=symbol, trade_date=trade_date, last_price=10.0,
                          previous_close=10.0, session_volume=1e8, board="SH_Main")


def service(root) -> PaperOrderService:
    return PaperOrderService(root, market_source=market, industry_map={SYM: "bank"})


def send(svc, key: str, date: str = D1) -> dict:
    svc.submit({
        "idempotencyKey": key, "runId": "rate", "symbol": SYM, "side": "BUY",
        "quantity": 100, "limitPrice": 10.0, "tradeDate": date, "signalId": key,
    })
    return svc.drain()[-1]


def test_orders_beyond_the_per_symbol_daily_limit_never_reach_the_venue(tmp_path) -> None:
    svc = service(tmp_path / "p")
    try:
        limit = svc.manager.config.max_orders_per_symbol_per_day
        results = [send(svc, f"k{i}") for i in range(limit + 3)]
    finally:
        svc.close()

    reached = [r for r in results if r["venueStatus"] != "REJECTED"]
    refused = [r for r in results if r["venueStatus"] == "REJECTED"]
    assert len(reached) == limit
    assert len(refused) == 3
    assert all(f"already has {limit} submitted orders" in r["reason"] for r in refused)
    book = CanonicalLedger(tmp_path / "p" / "canonical.jsonl").replay_book()
    rejected = [o for o in book.orders() if o.status.value == "REJECTED"]
    assert len(rejected) == 3
    decisions = [
        event.risk_decision for order in rejected
        for event in book.history_of(order.order_id) if event.risk_decision
    ]
    assert {d.rule for d in decisions} == {"max_orders_per_symbol_per_day"}


def test_the_count_survives_a_restart_and_resets_next_session(tmp_path) -> None:
    root = tmp_path / "p"
    svc = service(root)
    limit = svc.manager.config.max_orders_per_symbol_per_day
    for i in range(limit):
        send(svc, f"a{i}")
    svc.close()

    restarted = service(root)
    try:
        same_day = send(restarted, "after-restart")
        next_day = send(restarted, "next-session", date=D2)
    finally:
        restarted.close()

    assert same_day["venueStatus"] == "REJECTED", "a restart refunded today's order count"
    assert next_day["venueStatus"] == "FILLED"
