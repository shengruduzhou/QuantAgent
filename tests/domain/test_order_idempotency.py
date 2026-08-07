"""One order intent may never become two economic orders.

Module One's acceptance criterion. Each test below is one of the delivery paths
that produces a duplicate in practice; the store has to collapse all of them to
a single economic action while still allowing genuinely new actions through.

The failure mode being guarded is asymmetric and both directions are costly:
too permissive duplicates real money, too aggressive silently drops a
legitimate re-trade (the INC-E1 defect this project already shipped once, where
a per-(symbol, side) key with no date suppressed a later rebuy).
"""

from __future__ import annotations

import json

import pytest

from quantagent.domain.idempotency import (
    DuplicateAction,
    IdempotencyStore,
    broker_callback_key,
    order_intent_key,
)

_INTENT = dict(
    run_id="run_001",
    signal_id="sig_20260803",
    symbol="600000.SH",
    side="BUY",
    quantity=10_000,
    trade_date="2026-08-03",
)


def _submit(store: IdempotencyStore, key: str, order_id: str) -> str | None:
    """Simulate a guarded submit: only the winner creates an order."""
    result = store.claim(key)
    if not result.granted:
        return result.record.outcome
    store.resolve(key, outcome=order_id)
    return order_id


# -- the seven delivery paths ------------------------------------------------
def test_repeated_ui_clicks_create_one_order(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.jsonl")
    key = order_intent_key(**_INTENT)

    first = _submit(store, key, "ord_A")
    second = _submit(store, key, "ord_B")
    third = _submit(store, key, "ord_C")

    assert first == "ord_A"
    # Later clicks are told about the original order, never given a new one.
    assert second == "ord_A"
    assert third == "ord_A"
    assert len(store) == 1


def test_api_retry_after_a_timeout_that_actually_succeeded(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.jsonl")
    key = order_intent_key(**_INTENT)

    # The first call succeeded server-side; the client saw a timeout and retried.
    _submit(store, key, "ord_A")
    retried = store.claim(key)

    assert retried.duplicate
    assert retried.record.outcome == "ord_A"


def test_worker_killed_mid_submit_does_not_resubmit_on_restart(tmp_path):
    """The claim is fsynced before the action, so a SIGKILL cannot lose it."""
    path = tmp_path / "idem.jsonl"
    key = order_intent_key(**_INTENT)

    dying = IdempotencyStore(path)
    assert dying.claim(key).granted
    # Process dies here: claimed, but never resolved.
    del dying

    restarted = IdempotencyStore(path)
    assert restarted.claim(key).duplicate, "a restart must not re-submit a claimed intent"


def test_socket_reconnect_replaying_its_buffer(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.jsonl")
    key = order_intent_key(**_INTENT)
    _submit(store, key, "ord_A")

    # The reconnect replays the last N messages, including one already applied.
    replayed = [store.claim(key) for _ in range(5)]

    assert all(result.duplicate for result in replayed)
    assert len(store) == 1


def test_duplicate_broker_callback_applies_once(tmp_path):
    """A redelivered fill must not credit the position twice."""
    store = IdempotencyStore(tmp_path / "idem.jsonl")
    key = broker_callback_key(
        broker_order_id="BRK-77", execution_id="EXEC-1", event_type="FILL"
    )

    applied = [store.claim(key).granted for _ in range(3)]

    assert applied == [True, False, False]


def test_a_second_distinct_execution_on_the_same_order_still_applies(tmp_path):
    """Partial fills are separate economic events and must not be collapsed."""
    store = IdempotencyStore(tmp_path / "idem.jsonl")
    first = broker_callback_key(broker_order_id="BRK-77", execution_id="EXEC-1", event_type="FILL")
    second = broker_callback_key(broker_order_id="BRK-77", execution_id="EXEC-2", event_type="FILL")

    assert store.claim(first).granted
    assert store.claim(second).granted, "a distinct execution is not a duplicate"


def test_replaying_a_historical_event_log_creates_no_new_orders(tmp_path):
    path = tmp_path / "idem.jsonl"
    store = IdempotencyStore(path)
    keys = [
        order_intent_key(**{**_INTENT, "signal_id": f"sig_{i}", "trade_date": f"2026-08-0{i}"})
        for i in range(1, 5)
    ]
    for index, key in enumerate(keys):
        _submit(store, key, f"ord_{index}")
    assert len(store) == 4

    # Rebuild state from the log: every action is recognised as already done.
    rebuilt = IdempotencyStore(path)
    granted = [rebuilt.claim(key).granted for key in keys]

    assert granted == [False, False, False, False]
    assert len(rebuilt) == 4


def test_process_recovery_preserves_the_recorded_outcome(tmp_path):
    path = tmp_path / "idem.jsonl"
    key = order_intent_key(**_INTENT)
    _submit(IdempotencyStore(path), key, "ord_A")

    recovered = IdempotencyStore(path)

    assert recovered.get(key).outcome == "ord_A"


# -- the other direction: real actions must not be suppressed ----------------
def test_the_same_symbol_and_side_on_a_later_day_is_a_new_order(tmp_path):
    """Guards the INC-E1 regression: a coarse key silently dropped rebuys."""
    store = IdempotencyStore(tmp_path / "idem.jsonl")

    monday = order_intent_key(**{**_INTENT, "trade_date": "2026-08-03", "signal_id": "sig_mon"})
    wednesday = order_intent_key(**{**_INTENT, "trade_date": "2026-08-05", "signal_id": "sig_wed"})

    assert _submit(store, monday, "ord_mon") == "ord_mon"
    assert _submit(store, wednesday, "ord_wed") == "ord_wed", "a later-day rebuy is not a duplicate"


def test_a_revised_quantity_is_a_distinct_action(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.jsonl")

    original = order_intent_key(**_INTENT)
    revised = order_intent_key(**{**_INTENT, "quantity": 5_000})

    assert original != revised
    assert store.claim(original).granted
    assert store.claim(revised).granted


def test_different_runs_do_not_share_claims(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.jsonl")

    first = order_intent_key(**_INTENT)
    second = order_intent_key(**{**_INTENT, "run_id": "run_002"})

    assert store.claim(first).granted
    assert store.claim(second).granted


# -- durability mechanics ----------------------------------------------------
def test_a_torn_trailing_write_does_not_destroy_earlier_claims(tmp_path):
    path = tmp_path / "idem.jsonl"
    store = IdempotencyStore(path)
    good = order_intent_key(**_INTENT)
    _submit(store, good, "ord_A")

    # Simulate a process killed mid-append.
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "idem_partial", "claimed')

    recovered = IdempotencyStore(path)

    assert recovered.seen(good), "an intact earlier claim must survive a torn tail"
    assert recovered.get(good).outcome == "ord_A"


def test_strict_mode_raises_so_unexpected_duplicates_are_not_swallowed(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.jsonl")
    key = order_intent_key(**_INTENT)
    store.claim(key)

    with pytest.raises(DuplicateAction) as raised:
        store.claim(key, strict=True)

    assert raised.value.key == key


def test_claims_are_readable_append_only_records(tmp_path):
    path = tmp_path / "idem.jsonl"
    store = IdempotencyStore(path)
    key = order_intent_key(**_INTENT)
    store.claim(key)
    store.resolve(key, outcome="ord_A", payload={"quantity": 10_000})

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    # The claim and its resolution are both retained; nothing is overwritten.
    assert len(lines) == 2
    assert lines[0]["outcome"] is None
    assert lines[1]["outcome"] == "ord_A"
    assert lines[1]["payload"]["quantity"] == 10_000


def test_an_in_memory_store_does_not_survive_restart(tmp_path):
    """Documented limitation: only a durable store protects across a restart."""
    volatile = IdempotencyStore(None)
    key = order_intent_key(**_INTENT)
    assert volatile.claim(key).granted

    assert IdempotencyStore(None).claim(key).granted
