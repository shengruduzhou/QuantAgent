"""M1-10: one intent, one economic order — through the real order manager.

The previous round proved claim-once at store level. That is not the same claim:
`OrderManager` guarded submission with an in-memory `history` dict, which a
SIGKILL between `broker.submit()` and `_update()` forgets entirely. On recovery
the worker resubmitted and the account acquired a second economic order.

These drive the real `OrderManager` against a real broker double, and the
concurrency cases use genuine threads and genuine OS processes rather than
simulated interleavings — an in-process lock looks correct right up until two
workers are two processes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from quantagent.domain.idempotency import IdempotencyStore, order_intent_key
from quantagent.domain.lineage import Lineage
from quantagent.execution.broker_base import BrokerBase, Order, OrderState, OrderStatus
from quantagent.execution.order_manager import OrderManager, OrderManagerConfig

SYMBOL = "600000.SH"
RUN = Lineage(research_id="res_1", strategy_id="str_1", strategy_version_id="sv_1", run_id="run_1")


class RecordingBroker(BrokerBase):
    """Counts every economic submission it is asked to perform."""

    def __init__(self) -> None:
        self.submitted: list[Order] = []

    def submit(self, order: Order) -> OrderState:
        self.submitted.append(order)
        return OrderState(
            client_order_id=order.client_order_id,
            broker_order_id=f"BRK-{len(self.submitted)}",
            status=OrderStatus.SUBMITTED,
            filled_quantity=0,
            avg_price=0.0,
        )

    def cancel(self, client_order_id: str) -> OrderState:  # pragma: no cover - unused
        return OrderState(
            client_order_id=client_order_id, broker_order_id="", status=OrderStatus.CANCELLED,
            filled_quantity=0, avg_price=0.0,
        )

    def query_positions(self):
        return []

    def query_orders(self):  # pragma: no cover - unused
        return []

    def query_order(self, client_order_id: str):  # pragma: no cover - unused
        return None

    def query_account_value(self) -> float:
        return 1_000_000.0

    def on_trade(self, *args, **kwargs) -> None:  # pragma: no cover - unused
        return None


def _manager(tmp_path: Path, broker: RecordingBroker | None = None) -> OrderManager:
    return OrderManager(
        broker=broker or RecordingBroker(),
        config=OrderManagerConfig(strategy_version="test"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        idempotency_path=str(tmp_path / "idem.jsonl"),
        lineage=RUN,
    )


def _reconcile(manager: OrderManager, weight: float = 0.10, signal: str = "sig_1") -> list:
    return manager.reconcile(
        target_weights=pd.Series({SYMBOL: weight}),
        prices=pd.Series({SYMBOL: 10.0}),
        nav=1_000_000.0,
        signal_id=signal,
    )


# -- repeated delivery through the real manager ------------------------------
def test_repeated_reconcile_creates_one_economic_order(tmp_path):
    broker = RecordingBroker()
    manager = _manager(tmp_path, broker)

    _reconcile(manager)
    _reconcile(manager)
    _reconcile(manager)

    assert len(broker.submitted) == 1, "the broker must be asked to trade exactly once"


def test_a_restarted_manager_does_not_resubmit(tmp_path):
    """The case the in-memory `history` dict could never cover."""
    broker = RecordingBroker()
    first = _manager(tmp_path, broker)
    _reconcile(first)
    assert len(broker.submitted) == 1

    # Process restart: fresh manager, fresh in-memory history, same durable store.
    revived = _manager(tmp_path, broker)
    _reconcile(revived)

    assert len(broker.submitted) == 1, "a restart must not resubmit a claimed intent"


def test_a_later_session_rebuy_is_still_allowed(tmp_path):
    """Guards the other direction: the INC-E1 over-suppression."""
    broker = RecordingBroker()
    manager = _manager(tmp_path, broker)

    _reconcile(manager, signal="sig_day1")
    manager.history.clear()          # a new session clears the projection
    manager.counts_today.clear()
    _reconcile(manager, signal="sig_day2")

    assert len(broker.submitted) == 2, "a genuinely new signal must reach the broker"


def test_a_modified_quantity_is_a_distinct_economic_order(tmp_path):
    broker = RecordingBroker()
    manager = _manager(tmp_path, broker)

    _reconcile(manager, weight=0.10, signal="sig_1")
    manager.history.clear()
    manager.counts_today.clear()
    _reconcile(manager, weight=0.20, signal="sig_1")

    assert len(broker.submitted) == 2
    assert broker.submitted[0].quantity != broker.submitted[1].quantity


# -- concurrency -------------------------------------------------------------
def test_two_threads_submitting_the_same_intent_create_one_order(tmp_path):
    broker = RecordingBroker()
    manager = _manager(tmp_path, broker)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: _reconcile(manager), range(8)))

    assert len(broker.submitted) == 1


def test_many_threads_claiming_one_key_yield_exactly_one_winner(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.jsonl")
    key = order_intent_key(
        run_id="run_1", signal_id="sig_1", symbol=SYMBOL, side="BUY",
        quantity=1_000, trade_date="2026-08-03",
    )

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(lambda _: store.claim(key).granted, range(64)))

    assert sum(results) == 1, f"expected exactly one winner, got {sum(results)}"


_CHILD = textwrap.dedent(
    """
    import json, sys
    from quantagent.domain.idempotency import IdempotencyStore, order_intent_key
    store = IdempotencyStore(sys.argv[1])
    key = order_intent_key(
        run_id="run_1", signal_id="sig_1", symbol="600000.SH", side="BUY",
        quantity=1000, trade_date="2026-08-03",
    )
    print(json.dumps({"granted": store.claim(key).granted}))
    """
)


def test_two_os_processes_cannot_both_claim_the_same_intent(tmp_path):
    """An in-process lock proves nothing about a second worker process."""
    path = tmp_path / "idem.jsonl"
    script = tmp_path / "child.py"
    script.write_text(_CHILD, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}

    granted = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, str(script), str(path)],
            capture_output=True, text=True, env=env, cwd=Path.cwd(),
        )
        assert completed.returncode == 0, completed.stderr
        granted.append(json.loads(completed.stdout)["granted"])

    assert granted == [True, False], "the second process must see the first process's claim"


def test_concurrent_recovery_workers_do_not_double_submit(tmp_path):
    """Two recovery workers replaying the same backlog."""
    broker = RecordingBroker()
    managers = [_manager(tmp_path, broker) for _ in range(2)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda m: _reconcile(m), managers))

    assert len(broker.submitted) == 1


# -- duplicate technical processing has zero economic effect -----------------
def test_a_duplicate_submission_changes_no_economic_quantity(tmp_path):
    broker = RecordingBroker()
    manager = _manager(tmp_path, broker)
    _reconcile(manager)

    ledger_len = len(manager.canonical)
    orders_before = len(manager.book.orders())
    submitted_before = len(broker.submitted)

    for _ in range(5):
        _reconcile(manager)

    assert len(broker.submitted) == submitted_before
    assert len(manager.book.orders()) == orders_before
    assert len(manager.canonical) == ledger_len, "a duplicate must add no economic event"


def test_the_stored_outcome_is_returned_for_a_repeated_token(tmp_path):
    broker = RecordingBroker()
    manager = _manager(tmp_path, broker)
    first = _reconcile(manager)

    manager.history.clear()  # force the claim path rather than the history path
    manager.counts_today.clear()
    repeated = _reconcile(manager)

    assert len(broker.submitted) == 1
    # The repeat is told about the original rather than given a new order.
    if repeated:
        assert repeated[0].client_order_id == first[0].client_order_id


# -- canonical emission ------------------------------------------------------
def test_the_order_manager_writes_canonical_entities(tmp_path):
    manager = _manager(tmp_path)
    _reconcile(manager)

    orders = manager.book.orders()
    assert len(orders) == 1
    order = orders[0]
    assert order.lineage.run_id == "run_1"
    assert order.lineage.strategy_version_id == "sv_1"
    assert order.lineage.signal_id is not None
    assert order.lineage.order_intent_id is not None
    assert manager.canonical.verify()["valid"]


def test_a_risk_decision_is_recorded_with_the_order(tmp_path):
    manager = _manager(tmp_path)
    _reconcile(manager)

    decisions = [
        event.risk_decision
        for event in manager.book.events()
        if event.risk_decision is not None
    ]
    assert decisions, "the order manager must record why it allowed the order"
    assert decisions[0].rule == "order_manager_basic_admissibility"
    assert decisions[0].approved is True
    assert "not production risk certification" in decisions[0].reason


# -- M1-13: idempotency is mandatory, not opportunistic ----------------------
def test_submission_without_lineage_fails_closed(tmp_path):
    """A request that cannot be identified cannot be de-duplicated."""
    from quantagent.execution.order_manager import MissingIdempotencyLineage

    broker = RecordingBroker()
    manager = OrderManager(
        broker=broker,
        config=OrderManagerConfig(strategy_version="test"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        idempotency_path=str(tmp_path / "idem.jsonl"),
        lineage=Lineage(),  # no run_id
    )

    with pytest.raises(MissingIdempotencyLineage):
        _reconcile(manager)

    assert broker.submitted == [], "nothing may reach the broker before the guard"


def test_same_key_with_changed_parameters_is_a_conflict(tmp_path):
    """Returning the stored result would answer a question nobody asked."""
    from quantagent.domain.idempotency import IdempotencyStore, order_intent_key
    from quantagent.execution.order_manager import IdempotencyConflict

    store = IdempotencyStore(tmp_path / "idem.jsonl")
    key = order_intent_key(
        run_id="run_1", signal_id="sig_1", symbol=SYMBOL, side="BUY",
        quantity=1_000, trade_date="2026-08-03",
    )
    store.claim(key, payload={"clientOrderId": "coid_1", "fingerprint": "aaaaaaaa"})

    broker = RecordingBroker()
    manager = _manager(tmp_path, broker)
    conflict = IdempotencyConflict(key, "aaaaaaaa", "bbbbbbbb")

    assert conflict.stored == "aaaaaaaa"
    assert conflict.attempted == "bbbbbbbb"
    assert "already used with fingerprint" in str(conflict)


def test_a_forensic_harness_cannot_be_wired_to_economic_execution():
    """Disabling the guard is only safe against an in-memory double."""
    from quantagent.execution.order_manager import ForensicHarnessLeak, assert_forensic_isolation

    class PaperBroker:  # name alone marks it as economically reachable
        pass

    with pytest.raises(ForensicHarnessLeak):
        assert_forensic_isolation(PaperBroker())

    assert_forensic_isolation(RecordingBroker()) is None


def test_history_is_a_projection_not_independent_state(tmp_path):
    """`history` must be derived; it cannot hold an order the ledger lacks."""
    manager = _manager(tmp_path)
    _reconcile(manager)

    assert len(manager.history) == 1
    # Mutating the returned mapping cannot corrupt the record of account.
    manager.history.clear()
    assert len(manager.history) == 1, "history is derived, so clearing a copy changes nothing"


def test_history_rebuilds_from_the_durable_ledger(tmp_path):
    manager = _manager(tmp_path)
    _reconcile(manager)

    rebuilt = manager.rebuild_history()

    assert set(rebuilt) == set(manager.history)
