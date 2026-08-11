from __future__ import annotations

from pathlib import Path

import pytest

from quantagent.execution.parent_child import (
    ChildStatus,
    ExecutionAlgorithm,
    ParentChildExecutionEngine,
    ParentExecutionConflict,
    ParentExecutionCorruption,
    ParentExecutionError,
    ParentExecutionStore,
    ParentOrderSpec,
    child_lineage_id,
)


SCHEDULE = (
    "2026-08-12T09:35:00+08:00",
    "2026-08-12T10:30:00+08:00",
    "2026-08-12T13:30:00+08:00",
    "2026-08-12T14:55:00+08:00",
)


def _store(tmp_path: Path) -> ParentExecutionStore:
    return ParentExecutionStore(tmp_path / "parent.json")


def test_twap_allocates_exact_parent_and_redelivers_same_released_child(tmp_path: Path) -> None:
    parent = ParentOrderSpec(
        parent_id="p-twap",
        symbol="600000.SH",
        side="buy",
        total_quantity=1_000,
        algorithm=ExecutionAlgorithm.TWAP,
        schedule_times=SCHEDULE,
        lot_size=100,
    )
    engine = ParentChildExecutionEngine(parent, _store(tmp_path))
    state = engine.state()
    assert sum(child.quantity for child in state.children) == 1_000
    assert [child.quantity for child in state.children] == [300, 300, 200, 200]

    released = engine.release_due(now="2026-08-12T10:00:00+08:00")
    assert len(released) == 1
    assert released[0].quantity == 300
    assert released[0].status is ChildStatus.RELEASED

    restarted = ParentChildExecutionEngine(parent, _store(tmp_path))
    redelivered = restarted.release_due(now="2026-08-12T10:00:00+08:00")
    assert [child.child_id for child in redelivered] == [released[0].child_id]
    assert child_lineage_id(parent.parent_id, released[0].child_id).endswith(
        released[0].child_id
    )


def test_vwap_requires_finite_profile_and_allocates_by_forecast_volume(tmp_path: Path) -> None:
    for profile in [(), (0.1, 0.2, float("inf"), 0.7), (0.1, 0.2, float("nan"), 0.7)]:
        with pytest.raises(ParentExecutionError, match="volume profile"):
            ParentOrderSpec(
                parent_id="bad-vwap",
                symbol="600000.SH",
                side="buy",
                total_quantity=1_000,
                algorithm=ExecutionAlgorithm.VWAP,
                schedule_times=SCHEDULE,
                volume_profile=profile,
            ).canonical()

    parent = ParentOrderSpec(
        parent_id="p-vwap",
        symbol="600000.SH",
        side="buy",
        total_quantity=1_000,
        algorithm=ExecutionAlgorithm.VWAP,
        schedule_times=SCHEDULE,
        volume_profile=(0.1, 0.2, 0.3, 0.4),
    )
    state = ParentChildExecutionEngine(parent, _store(tmp_path)).state()
    assert [child.quantity for child in state.children] == [100, 200, 300, 400]


def test_pov_is_monotonic_restart_safe_and_has_one_active_child(tmp_path: Path) -> None:
    parent = ParentOrderSpec(
        parent_id="p-pov",
        symbol="600000.SH",
        side="buy",
        total_quantity=2_000,
        algorithm=ExecutionAlgorithm.POV,
        participation_rate=0.10,
        max_child_quantity=500,
    )
    engine = ParentChildExecutionEngine(parent, _store(tmp_path))

    first = engine.release_due(
        now="2026-08-12T09:40:00+08:00",
        cumulative_market_volume=4_000,
    )
    assert len(first) == 1
    assert first[0].quantity == 400

    restarted = ParentChildExecutionEngine(parent, _store(tmp_path))
    same = restarted.release_due(
        now="2026-08-12T09:41:00+08:00",
        cumulative_market_volume=10_000,
    )
    # The market-volume target grew, but the existing live child prevents a
    # second simultaneous order. The same deterministic child is redelivered.
    assert len(same) == 1
    assert same[0].child_id == first[0].child_id
    assert restarted.state().last_observed_cumulative_volume == 10_000

    with pytest.raises(ParentExecutionError, match="moved backwards"):
        restarted.release_due(
            now="2026-08-12T09:42:00+08:00",
            cumulative_market_volume=9_900,
        )

    restarted.acknowledge(
        first[0].child_id,
        status=ChildStatus.FILLED,
        filled_quantity=400,
    )
    second = restarted.release_due(
        now="2026-08-12T10:00:00+08:00",
        cumulative_market_volume=10_000,
    )
    assert len(second) == 1
    assert second[0].quantity == 500
    assert restarted.state().committed_quantity == 900

    restarted.acknowledge(
        second[0].child_id,
        status=ChildStatus.FILLED,
        filled_quantity=500,
    )
    tail = restarted.release_due(
        now="2026-08-12T10:01:00+08:00",
        cumulative_market_volume=10_000,
    )
    assert len(tail) == 1
    assert tail[0].quantity == 100
    assert restarted.state().committed_quantity == 1_000


def test_partial_child_is_not_released_again_at_full_quantity(tmp_path: Path) -> None:
    parent = ParentOrderSpec(
        parent_id="p-partial",
        symbol="600000.SH",
        side="buy",
        total_quantity=1_000,
        algorithm=ExecutionAlgorithm.POV,
        participation_rate=0.10,
    )
    engine = ParentChildExecutionEngine(parent, _store(tmp_path))
    child = engine.release_due(
        now="2026-08-12T09:40:00+08:00",
        cumulative_market_volume=5_000,
    )[0]
    engine.acknowledge(
        child.child_id,
        status=ChildStatus.PARTIAL,
        filled_quantity=200,
    )

    assert engine.released_children() == ()
    assert engine.release_due(
        now="2026-08-12T09:45:00+08:00",
        cumulative_market_volume=8_000,
    ) == ()
    assert engine.state().outstanding_quantity == 300


def test_cancelled_partial_child_only_commits_actual_fill(tmp_path: Path) -> None:
    parent = ParentOrderSpec(
        parent_id="p-cancel",
        symbol="600000.SH",
        side="buy",
        total_quantity=1_000,
        algorithm=ExecutionAlgorithm.POV,
        participation_rate=0.10,
    )
    engine = ParentChildExecutionEngine(parent, _store(tmp_path))
    child = engine.release_due(
        now="2026-08-12T09:40:00+08:00",
        cumulative_market_volume=5_000,
    )[0]
    engine.acknowledge(
        child.child_id,
        status=ChildStatus.CANCELLED,
        filled_quantity=200,
    )
    state = engine.state()
    assert state.filled_quantity == 200
    assert state.outstanding_quantity == 0

    released = engine.release_due(
        now="2026-08-12T09:45:00+08:00",
        cumulative_market_volume=5_000,
    )
    assert len(released) == 1
    assert released[0].quantity == 300


def test_iceberg_has_one_active_display_child_and_replenishes_after_fill(tmp_path: Path) -> None:
    parent = ParentOrderSpec(
        parent_id="p-ice",
        symbol="600000.SH",
        side="sell",
        total_quantity=1_000,
        algorithm=ExecutionAlgorithm.ICEBERG,
        display_quantity=300,
    )
    engine = ParentChildExecutionEngine(parent, _store(tmp_path))
    first = engine.release_due(now="2026-08-12T09:35:00+08:00")
    assert len(first) == 1
    assert first[0].quantity == 300

    still_first = engine.release_due(now="2026-08-12T09:36:00+08:00")
    assert len(still_first) == 1
    assert still_first[0].child_id == first[0].child_id

    engine.acknowledge(
        first[0].child_id,
        status=ChildStatus.FILLED,
        filled_quantity=300,
    )
    second = engine.release_due(now="2026-08-12T09:37:00+08:00")
    assert len(second) == 1
    assert second[0].quantity == 300
    assert second[0].child_id != first[0].child_id


def test_terminal_child_state_is_immutable(tmp_path: Path) -> None:
    parent = ParentOrderSpec(
        parent_id="p-terminal",
        symbol="600000.SH",
        side="buy",
        total_quantity=1_000,
        algorithm=ExecutionAlgorithm.ICEBERG,
        display_quantity=500,
    )
    engine = ParentChildExecutionEngine(parent, _store(tmp_path))
    child = engine.release_due(now="2026-08-12T09:35:00+08:00")[0]
    engine.acknowledge(
        child.child_id,
        status=ChildStatus.CANCELLED,
        filled_quantity=200,
    )
    # Exact redelivery is idempotent.
    state = engine.acknowledge(
        child.child_id,
        status=ChildStatus.CANCELLED,
        filled_quantity=200,
    )
    assert state.filled_quantity == 200
    with pytest.raises(ParentExecutionError, match="terminal child state"):
        engine.acknowledge(
            child.child_id,
            status=ChildStatus.REJECTED,
            filled_quantity=200,
        )


def test_parent_economics_are_immutable_across_restart(tmp_path: Path) -> None:
    store = _store(tmp_path)
    parent = ParentOrderSpec(
        parent_id="p-one",
        symbol="600000.SH",
        side="buy",
        total_quantity=1_000,
        algorithm=ExecutionAlgorithm.TWAP,
        schedule_times=SCHEDULE,
    )
    ParentChildExecutionEngine(parent, store)
    changed = ParentOrderSpec(
        parent_id="p-one",
        symbol="600000.SH",
        side="buy",
        total_quantity=2_000,
        algorithm=ExecutionAlgorithm.TWAP,
        schedule_times=SCHEDULE,
    )
    with pytest.raises(ParentExecutionConflict):
        ParentChildExecutionEngine(changed, store)


def test_tampered_snapshot_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    parent = ParentOrderSpec(
        parent_id="p-tamper",
        symbol="600000.SH",
        side="buy",
        total_quantity=1_000,
        algorithm=ExecutionAlgorithm.TWAP,
        schedule_times=SCHEDULE,
    )
    ParentChildExecutionEngine(parent, store)
    text = store.path.read_text(encoding="utf-8").replace(
        '"total_quantity": 1000',
        '"total_quantity": 2000',
    )
    store.path.write_text(text, encoding="utf-8")
    with pytest.raises(ParentExecutionCorruption, match="digest mismatch"):
        store.read()


def test_acknowledgement_cannot_overfill_or_move_fill_backwards(tmp_path: Path) -> None:
    parent = ParentOrderSpec(
        parent_id="p-ack",
        symbol="600000.SH",
        side="buy",
        total_quantity=1_000,
        algorithm=ExecutionAlgorithm.ICEBERG,
        display_quantity=500,
    )
    engine = ParentChildExecutionEngine(parent, _store(tmp_path))
    child = engine.release_due(now="2026-08-12T09:35:00+08:00")[0]
    engine.acknowledge(
        child.child_id,
        status=ChildStatus.PARTIAL,
        filled_quantity=300,
    )
    with pytest.raises(ParentExecutionError, match="moved backwards"):
        engine.acknowledge(
            child.child_id,
            status=ChildStatus.PARTIAL,
            filled_quantity=200,
        )
    with pytest.raises(ParentExecutionError, match="exceeds child quantity"):
        engine.acknowledge(
            child.child_id,
            status=ChildStatus.FILLED,
            filled_quantity=600,
        )
