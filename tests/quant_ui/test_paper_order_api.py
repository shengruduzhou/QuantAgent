"""M1-13: idempotency of the real HTTP -> queue -> worker -> OMS -> paper path.

Every scenario Module One's gate lists for the production entry point, driven
through the actual FastAPI route rather than through `OrderManager` directly:
double click, repeated POST, timeout after success, gateway retry, duplicate queue
delivery, concurrent workers, concurrent processes, simultaneous recovery, a crash
at each commit boundary, duplicate and reordered callbacks, duplicate execution
id, a changed fingerprint, a later-session order, and cancel.

The measurement in all of them is the same and is taken from the ledger, not from
the API's reply: **how many economic orders and fills exist**. An endpoint that
answers "accepted" twice is fine; one that produces two orders is not.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient
import pytest

from quantagent.domain.ledger import CanonicalLedger
from quantagent.domain.orders import OrderStatus
from quantagent.paper.broker import MarketSnapshot
from services.quant_api.app import create_app
from services.quant_api.config import ApiSettings
from services.quant_api.services.paper_orders import (
    EXECUTED,
    INTERRUPTED,
    MARKET_DATA_UNAVAILABLE,
    PaperOrderRequest,
    PaperOrderService,
    QUEUED,
    SubmissionRejected,
    WriterLockUnavailable,
)

SYMBOL = "600000.SH"
SESSION_1 = "2026-08-04"
SESSION_2 = "2026-08-05"
INITIAL = 1_000_000.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def market_source(symbol: str, trade_date: str) -> MarketSnapshot | None:
    """Deterministic market data. Returns None for an unknown symbol on purpose."""
    if symbol != SYMBOL:
        return None
    return MarketSnapshot(
        symbol=symbol, trade_date=trade_date, last_price=10.00,
        previous_close=10.00, session_volume=1e8, board="SH_Main",
    )


def order_payload(key: str, **overrides) -> dict:
    """A submission. `signalId` defaults to the delivery key.

    Delivery identity and economic identity are separate on this path, and most
    tests here want two genuinely distinct requests — so by default each delivery
    key carries its own signal. `test_a_fresh_key_for_economics_that_already_traded`
    overrides `signalId` to share one, which is the case where the economic guard
    rather than the request guard has to do the work.
    """
    return {
        "idempotencyKey": key,
        "runId": "run_api",
        "symbol": SYMBOL,
        "side": "BUY",
        "quantity": 1_000,
        "limitPrice": 10.05,
        "tradeDate": SESSION_1,
        "signalId": key,
        **overrides,
    }


@pytest.fixture
def service(tmp_path) -> PaperOrderService:
    svc = PaperOrderService(
        tmp_path / "paper_orders", market_source=market_source, initial_cash=INITIAL
    )
    yield svc
    svc.close()


@pytest.fixture
def client(tmp_path):
    """A real app whose runtime root is isolated, so it holds its own writer lock."""
    settings = ApiSettings(
        project_root=PROJECT_ROOT,
        runtime_root=tmp_path / "runtime",
        cache_root=tmp_path / "runtime" / "cache",
        jobs_root=tmp_path / "runtime" / "jobs",
    ).ensure()
    app = create_app(settings)
    app.state.services.paper_orders.market_source = market_source
    app.state.services.paper_orders.initial_cash = INITIAL
    app.state.services.paper_orders.broker.portfolio.cash = INITIAL
    app.state.services.paper_orders.broker.portfolio.initial_cash = INITIAL
    with TestClient(app) as test_client:
        yield test_client
    app.state.services.paper_orders.close()


def economic_facts(ledger_path: Path) -> dict:
    """The only measurement that matters: what the record of account holds."""
    book = CanonicalLedger(ledger_path).replay_book()
    return {
        "orders": len(book.orders()),
        "fills": len(book.fills()),
        "execution_ids": len({fill.execution_id for fill in book.fills()}),
        "filled_quantity": sum(fill.quantity for fill in book.fills()),
    }


# -- the endpoint exists and is economically real -----------------------------
def test_submission_reaches_the_ledger_through_the_route(client):
    accepted = client.post("/api/paper/orders", json=order_payload("k1"))
    assert accepted.status_code == 200
    assert accepted.json()["data"]["state"] == QUEUED

    drained = client.post("/api/paper/orders/drain")
    assert drained.status_code == 200
    assert drained.json()["data"][0]["state"] == EXECUTED

    orders = client.get("/api/paper/orders").json()["data"]
    assert len(orders) == 1
    assert orders[0]["status"] == OrderStatus.FILLED.value
    assert orders[0]["cumulativeQuantity"] == 1_000
    assert orders[0]["lineage"]["run_id"]

    account = client.get("/api/paper/account").json()["data"]
    assert account["cash"] < INITIAL, "a filled buy must have consumed cash"
    # No market source is wired, so the held position has no mark. NAV is then
    # unknown — not zero, and not a stale close carried forward (DEF-021).
    assert account["nav"] is None
    assert account["unpriceableSymbols"] == [SYMBOL]
    assert "unknown rather than zero" in account["reason"]


def test_the_endpoint_queues_rather_than_executing_inline(client):
    """Response and execution are separate commit points, by design."""
    client.post("/api/paper/orders", json=order_payload("k1"))
    assert economic_facts_from_client(client)["orders"] == 0
    client.post("/api/paper/orders/drain")
    assert economic_facts_from_client(client)["orders"] == 1


def economic_facts_from_client(client) -> dict:
    orders = client.get("/api/paper/orders").json()["data"]
    return {"orders": len(orders)}


# -- duplicate delivery, every shape ------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        "double_click",
        "repeated_post",
        "timeout_after_success",
        "gateway_retry",
    ],
)
def test_repeated_delivery_produces_one_economic_order(client, shape):
    """Four client-side shapes, one economic outcome.

    They differ only in *when* the retry arrives relative to the work: a double
    click and a repeated POST retry before draining, a timeout-after-success and a
    gateway retry after the work has already happened.
    """
    payload = order_payload("k1")
    first = client.post("/api/paper/orders", json=payload)
    assert first.status_code == 200

    if shape in {"double_click", "repeated_post"}:
        second = client.post("/api/paper/orders", json=payload)
        client.post("/api/paper/orders/drain")
    else:
        client.post("/api/paper/orders/drain")
        second = client.post("/api/paper/orders", json=payload)

    assert second.status_code == 200
    assert second.json()["data"]["duplicate"] is True
    client.post("/api/paper/orders/drain")

    orders = client.get("/api/paper/orders").json()["data"]
    assert len(orders) == 1, f"{shape} produced {len(orders)} economic orders"
    assert sum(o["cumulativeQuantity"] for o in orders) == 1_000


def test_duplicate_queue_delivery_executes_once(service):
    """The same queue entry handed to the worker twice."""
    service.submit(order_payload("k1"))
    first = service.drain()
    second = service.drain()

    assert first and first[0]["state"] == EXECUTED
    assert second == [], "the queue entry was still pending after execution"
    assert economic_facts(service.ledger_path)["orders"] == 1


def test_a_changed_fingerprint_conflicts_instead_of_returning_the_original(client):
    client.post("/api/paper/orders", json=order_payload("k1"))
    conflict = client.post("/api/paper/orders", json=order_payload("k1", quantity=500))

    assert conflict.status_code == 409
    assert "fingerprint" in conflict.text
    client.post("/api/paper/orders/drain")
    orders = client.get("/api/paper/orders").json()["data"]
    assert len(orders) == 1
    assert orders[0]["quantity"] == 1_000, "the conflicting request must not have traded"


def test_a_later_session_order_is_not_suppressed(client):
    """The same economics on a later date is a real second order, not a duplicate."""
    client.post("/api/paper/orders", json=order_payload("day1", tradeDate=SESSION_1))
    client.post("/api/paper/orders/drain")
    client.post("/api/paper/orders", json=order_payload("day2", tradeDate=SESSION_2))
    client.post("/api/paper/orders/drain")

    orders = client.get("/api/paper/orders").json()["data"]
    assert len(orders) == 2, "the second session's buy was wrongly de-duplicated"
    assert {o["tradeDate"] for o in orders} == {SESSION_1, SESSION_2}


# -- failing closed -----------------------------------------------------------
def test_a_submission_without_an_idempotency_key_is_refused(client):
    payload = order_payload("k1")
    del payload["idempotencyKey"]
    assert client.post("/api/paper/orders", json=payload).status_code == 422
    assert client.get("/api/paper/orders").json()["data"] == []


@pytest.mark.parametrize("missing", ["runId", "signalId"])
def test_a_submission_without_lineage_is_refused(service, missing):
    """Both halves of lineage are required, and both fail closed.

    `runId` makes the resulting order traceable; `signalId` is the economic
    identity the order-intent guard de-duplicates on. Neither has a default,
    because a defaulted lineage field is a guard that quietly stops working.
    """
    payload = order_payload("k1")
    payload[missing] = ""
    with pytest.raises(SubmissionRejected) as caught:
        service.submit(payload)
    assert caught.value.reason == "missing_lineage"
    assert economic_facts(service.ledger_path)["orders"] == 0


def test_the_schema_also_refuses_a_missing_signal_id(client):
    payload = order_payload("k1")
    del payload["signalId"]
    assert client.post("/api/paper/orders", json=payload).status_code == 422
    assert client.get("/api/paper/orders").json()["data"] == []


def test_two_signals_wanting_the_same_trade_both_execute(service):
    """The other direction: over-suppression is a defect too.

    Two sleeves can legitimately want the same symbol, side, quantity and date on
    the same day. A guard keyed on economics alone would silently drop the second —
    the INC-E1 defect class this project has already paid for once.
    """
    service.submit(order_payload("d1", signalId="sig_momentum"))
    service.submit(order_payload("d2", signalId="sig_value"))
    service.drain()

    facts = economic_facts(service.ledger_path)
    assert facts["orders"] == 2, "a legitimate second sleeve order was suppressed"
    assert facts["filled_quantity"] == 2_000


def test_an_unbounded_order_is_refused(client):
    payload = order_payload("k1")
    payload["limitPrice"] = 0
    assert client.post("/api/paper/orders", json=payload).status_code == 422


def test_live_intent_is_refused_before_anything_is_recorded(client):
    payload = order_payload("k1", signalId="enable_live trading now")
    refused = client.post("/api/paper/orders", json=payload)

    assert refused.status_code == 451, refused.text
    assert "LIVE_DISABLED" in refused.text
    assert client.get("/api/paper/orders").json()["data"] == []


def test_missing_market_data_is_an_explicit_rejection_not_a_fabricated_fill(service):
    """No price, no fill. The alternative is a fill invented by the server."""
    service.submit(order_payload("k1", symbol="000001.SZ"))
    result = service.drain()

    assert result[0]["reason"] == MARKET_DATA_UNAVAILABLE
    assert economic_facts(service.ledger_path)["fills"] == 0


def test_the_policy_route_states_the_boundary(client):
    policy = client.get("/api/paper/policy").json()["data"]
    assert policy["mode"] == "PAPER"
    assert policy["liveTradingAvailable"] is False
    assert policy["simulatesOrders"] is True
    assert policy["writable"] is True


# -- concurrency --------------------------------------------------------------
def test_both_guards_are_durable_and_share_one_file(service):
    """The request guard and the economic-intent guard must both survive a restart.

    DEF-015: the `OrderManager` was constructed without an `idempotency_path`, so
    its intent guard was `IdempotencyStore(None)` — in memory, forgotten on
    restart and invisible to a second process. Only the request-level claim was
    protecting anything, which meant a caller inventing a fresh idempotency key
    for economics that had already traded would have got a second order.
    """
    service.submit(order_payload("k1"))
    service.drain()

    keys = [
        json.loads(line)["key"]
        for line in (service.root / "claims.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(k.startswith("req_") for k in keys), "no durable request claim"
    assert any(k.startswith("idem_") for k in keys), (
        "the OMS's economic-intent claim never reached disk"
    )
    assert service.manager.claims.path == service.claims.path


def test_a_fresh_key_for_economics_that_already_traded_is_still_stopped(service):
    """The second guard's whole purpose, exercised end to end.

    A client that loses its idempotency key and retries with a new one has, from
    the request layer's point of view, made a brand new request. The economic-intent
    guard is what stops it from becoming a second order.
    """
    service.submit(order_payload("first-delivery", signalId="sig_alpha"))
    service.drain()
    service.submit(order_payload("second-delivery", signalId="sig_alpha"))
    service.drain()

    facts = economic_facts(service.ledger_path)
    assert facts["orders"] == 1, f"{facts['orders']} orders for one economic intent"
    assert facts["fills"] == 1
    assert service.status("second-delivery", "run_api")["state"] == EXECUTED


def test_concurrent_workers_execute_one_economic_order(service):
    """Eight threads drain the same queue entry."""
    service.submit(order_payload("k1"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: service.drain(), range(8)))

    facts = economic_facts(service.ledger_path)
    assert facts["orders"] == 1, f"{facts['orders']} economic orders from 8 workers"
    assert facts["fills"] == 1
    assert facts["filled_quantity"] == 1_000


def test_concurrent_posts_of_the_same_key_claim_once(service):
    """Sixteen simultaneous deliveries of one request."""
    payload = order_payload("k1")

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: service.submit(payload), range(16)))

    granted = [r for r in results if not r["duplicate"]]
    assert len(granted) == 1, f"{len(granted)} callers were each told they had won"
    service.drain()
    assert economic_facts(service.ledger_path)["orders"] == 1


_WORKER = """
import json, sys
from pathlib import Path
sys.path.insert(0, {root!r})
sys.path.insert(0, str(Path({root!r}) / "src"))
from quantagent.paper.broker import MarketSnapshot
from services.quant_api.services.paper_orders import PaperOrderService

def market(symbol, trade_date):
    if symbol != {symbol!r}:
        return None
    return MarketSnapshot(symbol=symbol, trade_date=trade_date, last_price=10.00,
                          previous_close=10.00, session_volume=1e8, board="SH_Main")

svc = PaperOrderService({root_dir!r}, market_source=market, initial_cash={cash!r})
try:
    print(json.dumps({{"writable": svc.writable, "drained": svc.drain() if svc.writable else []}}))
finally:
    svc.close()
"""


def _run_worker(root_dir: Path) -> dict:
    script = _WORKER.format(
        root=str(PROJECT_ROOT), root_dir=str(root_dir), symbol=SYMBOL, cash=INITIAL
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_two_worker_processes_produce_one_economic_order(service):
    """Cross-process: the guard has to be the OS and the file, not a thread lock."""
    service.submit(order_payload("k1"))
    root = service.root
    service.close()  # release the single-writer lock so the children can take it

    first = _run_worker(root)
    second = _run_worker(root)

    assert first["writable"] and second["writable"]
    facts = economic_facts(root / "canonical.jsonl")
    assert facts["orders"] == 1, f"{facts['orders']} orders from two processes"
    assert facts["fills"] == 1


def test_a_second_writer_in_another_process_is_refused(service):
    """The enforced deployment contract: single host, single writer."""
    result = _run_worker(service.root)

    assert result["writable"] is False, (
        "a second process took the writer lock while the first still held it"
    )
    assert result["drained"] == []


def test_a_writer_on_another_host_is_refused(tmp_path, monkeypatch):
    """`flock` is per-host, so the occupancy record is what blocks a second host.

    Simulated by writing a fresh record naming a different machine: on a shared
    filesystem the advisory lock would be granted, and granting it is precisely
    the failure — two hosts sizing orders from two in-memory portfolios.
    """
    from services.quant_api.services import paper_orders as po

    root = tmp_path / "paper"
    root.mkdir(parents=True)
    (root / "writer.lock").write_text(
        json.dumps(
            {
                "host": "some-other-machine",
                "pid": 4242,
                "acquiredAt": "2026-08-05T09:00:00+00:00",
                "heartbeatAt": _now_iso(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    service = PaperOrderService(root, market_source=market_source)
    try:
        assert service.writable is False
        assert "some-other-machine" in service.writer_lock_error
        assert "distributed writers" in service.writer_lock_error
        with pytest.raises(WriterLockUnavailable):
            service.submit(order_payload("k1"))
    finally:
        service.close()


def test_a_stale_heartbeat_from_a_dead_host_does_not_hold_the_account_hostage(tmp_path):
    """A dead writer must not lock the account forever."""
    from services.quant_api.services import paper_orders as po

    root = tmp_path / "paper"
    root.mkdir(parents=True)
    stale = datetime.now(timezone.utc) - timedelta(
        seconds=po.WRITER_HEARTBEAT_STALE_SECONDS + 60
    )
    (root / "writer.lock").write_text(
        json.dumps(
            {
                "host": "some-other-machine",
                "pid": 4242,
                "acquiredAt": stale.isoformat(timespec="seconds"),
                "heartbeatAt": stale.isoformat(timespec="seconds"),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    service = PaperOrderService(root, market_source=market_source)
    try:
        assert service.writable is True, "a stale record blocked a legitimate takeover"
    finally:
        service.close()


def test_an_unreadable_lock_record_reads_as_stale_not_as_a_live_holder(tmp_path):
    """One corrupt byte must not lock the account permanently."""
    root = tmp_path / "paper"
    root.mkdir(parents=True)
    (root / "writer.lock").write_text("{not json at all", encoding="utf-8")

    service = PaperOrderService(root, market_source=market_source)
    try:
        assert service.writable is True
    finally:
        service.close()


def test_the_heartbeat_advances_as_the_writer_works(service):
    """A busy writer must never look dead to another host."""
    lock = service.root / "writer.lock"
    first = json.loads(lock.read_text(encoding="utf-8"))
    service.submit(order_payload("k1"))
    second = json.loads(lock.read_text(encoding="utf-8"))

    assert second["host"] == first["host"]
    assert second["acquiredAt"] == first["acquiredAt"], "acquisition time must be stable"
    assert second["heartbeatAt"] >= first["heartbeatAt"]


def test_a_clean_shutdown_hands_the_lock_over_at_once(tmp_path):
    """No waiting out a heartbeat window after an orderly stop."""
    root = tmp_path / "paper"
    first = PaperOrderService(root, market_source=market_source)
    assert first.writable
    first.close()

    assert (root / "writer.lock").read_text(encoding="utf-8").strip() == ""
    second = PaperOrderService(root, market_source=market_source)
    try:
        assert second.writable is True
    finally:
        second.close()


def test_a_non_writer_instance_refuses_to_submit(tmp_path):
    holder = PaperOrderService(tmp_path / "p", market_source=market_source)
    shadow = PaperOrderService(tmp_path / "p", market_source=market_source)
    try:
        assert holder.writable and not shadow.writable
        with pytest.raises(WriterLockUnavailable, match="single host, single writer"):
            shadow.submit(order_payload("k1"))
        with pytest.raises(WriterLockUnavailable):
            shadow.drain()
        # Reads still work: refusing to answer questions would be a worse outcome.
        assert shadow.orders() == []
        assert shadow.account()["writable"] is False
    finally:
        shadow.close()
        holder.close()


# -- crash boundaries ---------------------------------------------------------
def test_crash_before_the_claim_leaves_nothing_behind(service):
    """Nothing was claimed, so nothing is owed and nothing was executed."""
    assert service.pending() == []
    assert economic_facts(service.ledger_path)["orders"] == 0
    assert service.status("never-sent", "run_api") is None


def test_crash_after_the_claim_is_recovered_from_the_ledger(service):
    """Claimed, not executed. Recovery must decide from the record of account."""
    service.submit(order_payload("k1"))
    assert service.pending()

    settled = service.recover()

    assert settled[0]["state"] == INTERRUPTED, (
        "a claim with no order on the chain means nothing executed; reporting it as "
        "executed would invent a fill"
    )
    assert economic_facts(service.ledger_path)["orders"] == 0
    assert service.pending() == [], "recovery must settle the queue, not leave it"


def test_an_interrupted_submission_is_never_retried_automatically(service):
    """Retrying past a claim is how duplicates are created, so it does not happen."""
    service.submit(order_payload("k1"))
    service.recover()

    assert service.drain() == []
    assert economic_facts(service.ledger_path)["orders"] == 0
    status = service.status("k1", "run_api")
    assert status["state"] == INTERRUPTED
    assert "new idempotency key" in status["reason"]


def test_a_resubmission_under_a_new_key_after_an_interruption_trades_once(service):
    service.submit(order_payload("k1"))
    service.recover()

    service.submit(order_payload("k2"))
    service.drain()

    facts = economic_facts(service.ledger_path)
    assert facts["orders"] == 1, "the recovery path must not have left a phantom order"
    assert facts["fills"] == 1


def test_crash_after_execution_is_recovered_as_executed(service, tmp_path):
    """The work happened; a restart must find it, not repeat it."""
    service.submit(order_payload("k1"))
    service.drain()
    before = economic_facts(service.ledger_path)
    root = service.root
    service.close()

    restarted = PaperOrderService(root, market_source=market_source, initial_cash=INITIAL)
    try:
        assert restarted.pending() == []
        assert restarted.status("k1", "run_api")["state"] == EXECUTED
        assert restarted.drain() == []
        assert economic_facts(restarted.ledger_path) == before
        # Economic state continues rather than restarting at the opening balance.
        assert restarted.broker.portfolio.cash == pytest.approx(
            restarted.account()["cash"], abs=1e-6
        )
        assert restarted.broker.portfolio.cash < INITIAL
        # And with a mark supplied, NAV is a number again.
        priced = restarted.account({SYMBOL: 10.0})
        assert priced["nav"] is not None
        assert priced["unpriceableSymbols"] == []
    finally:
        restarted.close()


def test_crash_after_the_canonical_append_loses_no_order_and_invents_no_fill(
    service, monkeypatch
):
    """The OMS recorded the order; the venue never got to fill it.

    The dangerous outcomes here are a *second* order on retry, and a fill reported
    for an order that never traded. Recovery must produce neither: it settles the
    request against the order the chain actually holds, cumulative quantity zero.
    """
    from quantagent.paper.broker import PaperBroker

    def die(self, order, market):
        raise RuntimeError("killed after the canonical append, before acceptance")

    monkeypatch.setattr(PaperBroker, "_validate", die)
    service.submit(order_payload("k1"))
    with pytest.raises(RuntimeError, match="killed after the canonical append"):
        service.drain()
    monkeypatch.undo()

    facts = economic_facts(service.ledger_path)
    assert facts["orders"] == 1, "the order the OMS recorded must still be there"
    assert facts["fills"] == 0, "a fill was invented for an order that never traded"

    settled = service.recover()

    assert settled[0]["state"] == EXECUTED
    assert settled[0]["filledQuantity"] == 0
    assert settled[0]["venueStatus"] == OrderStatus.SUBMITTED.value
    assert economic_facts(service.ledger_path)["orders"] == 1, "recovery duplicated it"
    assert service.drain() == [], "the request must not be queued for another attempt"


def test_crash_after_paper_acceptance_recovers_the_real_fill(service, monkeypatch):
    """The venue filled; the process died before the request was marked done.

    This is the window that produces duplicates in systems that retry on restart:
    the work is complete but nothing says so. Recovery must read the fill off the
    chain rather than resubmitting.
    """
    def die_after_venue(key, *, outcome, payload=None):
        # The OMS resolves its own intent claim on its own store instance over the
        # same file; this is the *request-level* resolve, which runs only after the
        # venue has already filled. Failing it is exactly the post-acceptance,
        # pre-acknowledgement window.
        raise RuntimeError("killed after paper acceptance, before resolve")

    monkeypatch.setattr(service.claims, "resolve", die_after_venue)
    service.submit(order_payload("k1"))
    with pytest.raises(RuntimeError, match="killed after paper acceptance"):
        service.drain()
    monkeypatch.undo()

    before = economic_facts(service.ledger_path)
    assert before["fills"] == 1, "the venue's fill must be on the chain"

    settled = service.recover()

    assert settled[0]["state"] == EXECUTED
    assert settled[0]["filledQuantity"] == 1_000, "the confirmed fill was lost"
    assert economic_facts(service.ledger_path) == before, "recovery moved money"
    assert service.drain() == []


def test_simultaneous_recovery_settles_once(service):
    """Two recovery passes at the same time must not double-resolve."""
    service.submit(order_payload("k1"))
    service.drain()
    service.submit(order_payload("k2"))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: service.recover(), range(4)))

    facts = economic_facts(service.ledger_path)
    assert facts["orders"] == 1
    assert facts["fills"] == 1
    assert service.status("k2", "run_api")["state"] == INTERRUPTED


# -- venue-side redelivery through the API's own broker -----------------------
def test_a_duplicate_execution_report_moves_no_money(service):
    service.submit(order_payload("k1"))
    service.drain()
    before = economic_facts(service.ledger_path)
    cash_before = service.broker.portfolio.cash

    order = next(iter(service.broker.orders.values()))
    fill = next(f for f in service.broker.fills if f.order_id == order.order_id)
    booked = service.broker.apply_execution_report(order, fill, trade_date=SESSION_1)

    assert booked is False, "a re-delivered execution report was booked a second time"
    assert economic_facts(service.ledger_path) == before
    assert service.broker.portfolio.cash == pytest.approx(cash_before)


def test_a_reordered_lifecycle_event_is_refused(service):
    from quantagent.domain.orders import IllegalTransition, OrderEventType

    service.submit(order_payload("k1"))
    service.drain()
    before = economic_facts(service.ledger_path)
    order = next(iter(service.broker.orders.values()))

    with pytest.raises(IllegalTransition):
        service.broker._canonical_event(
            order, OrderEventType.ACCEPTED, trade_date=SESSION_1
        )
    assert economic_facts(service.ledger_path) == before


def test_one_execution_id_appears_once_in_the_chain(service):
    service.submit(order_payload("k1"))
    service.drain()

    facts = economic_facts(service.ledger_path)
    assert facts["fills"] == facts["execution_ids"], (
        "an execution id appears more than once in the chain"
    )


# -- cancellation -------------------------------------------------------------
def test_cancelling_a_working_order_keeps_its_executed_quantity(tmp_path):
    """A thin book leaves a remainder, which cancel must not erase."""
    thin = "000002.SZ"

    def thin_market(symbol: str, trade_date: str) -> MarketSnapshot | None:
        if symbol != thin:
            return None
        return MarketSnapshot(
            symbol=symbol, trade_date=trade_date, last_price=20.00,
            previous_close=20.00, session_volume=10_000, board="SZ_Main",
        )

    service = PaperOrderService(
        tmp_path / "paper", market_source=thin_market, initial_cash=INITIAL
    )
    try:
        service.submit(
            order_payload("k1", symbol=thin, quantity=3_000, limitPrice=20.20)
        )
        service.drain()
        service.cancel("k1", "run_api")

        book = CanonicalLedger(service.ledger_path).replay_book()
        order = book.orders()[0]
        assert order.status is OrderStatus.CANCELLED
        assert order.cumulative_quantity == 1_000, "the cancel erased executed quantity"
        assert order.leaves_quantity == 0
    finally:
        service.close()


def test_cancelling_an_unknown_submission_is_refused(service):
    with pytest.raises(SubmissionRejected) as caught:
        service.cancel("never-sent", "run_api")
    assert caught.value.reason == "unknown_submission"


# -- the arrival log ---------------------------------------------------------
def test_every_delivery_the_service_saw_is_audited(client, tmp_path):
    """Accepted, duplicate and conflicting deliveries all leave a record.

    Scope note: a request the *schema* rejects (a missing key, a zero price) never
    reaches the service and so is not in this log — FastAPI refuses it at the
    boundary. `test_a_service_level_refusal_is_audited` covers the refusals the
    service itself makes.
    """
    client.post("/api/paper/orders", json=order_payload("k1"))
    client.post("/api/paper/orders", json=order_payload("k1"))
    client.post("/api/paper/orders", json=order_payload("k1", quantity=500))

    arrivals = [
        json.loads(line)
        for line in (tmp_path / "runtime" / "paper_orders" / "arrivals.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    outcomes = [record["outcome"] for record in arrivals]
    assert outcomes.count("accepted") == 1
    assert outcomes.count("duplicate") == 1
    assert outcomes.count("conflict") == 1
    conflict = next(r for r in arrivals if r["outcome"] == "conflict")
    assert conflict["storedFingerprint"] != conflict["attemptedFingerprint"]


def test_a_service_level_refusal_is_audited(service):
    """A refusal is evidence too: "did the client ever ask?" must be answerable."""
    payload = order_payload("k1")
    payload["runId"] = ""
    with pytest.raises(SubmissionRejected):
        service.submit(payload)

    arrivals = [
        json.loads(line)
        for line in (service.root / "arrivals.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [r["outcome"] for r in arrivals] == ["refused"]
    assert arrivals[0]["reason"] == "missing_lineage"


def test_recovery_does_not_attribute_another_requests_order(service):
    """Two keys, identical economics: one executed, one interrupted.

    Matching a queued request to a ledger order by economics alone made these
    indistinguishable, so an interrupted submission was reported as executed on the
    strength of a different request's fill. The match is by lineage.
    """
    service.submit(order_payload("first"))
    service.drain()
    service.submit(order_payload("second"))

    settled = service.recover()

    assert service.status("first", "run_api")["state"] == EXECUTED
    assert service.status("second", "run_api")["state"] == INTERRUPTED
    assert len(settled) == 1
    assert economic_facts(service.ledger_path)["orders"] == 1
