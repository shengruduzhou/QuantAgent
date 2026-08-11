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
)


SCHEDULE = (
    "2026-08-12T09:35:00+08:00",
    "2026-08-12T10:30:00+08:00",
    "2026-08-12T13:30:00+08:00",
    "2026-08-12T14:55:00+08:00",
)


def _store(tmp_path: Path) -> ParentExecutionStore:
    return ParentExecutionStore(tmp_path / "parent.json")


def test_twap_allocates_exact_parent_and_releases_only_due_slices(tmp_path: Path) -> None:
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

    # Restart/redelivery produces the same deterministic child identity, not a
    # new economic order.
    restarted = ParentChildExecutionEngine(parent, _store(tmp_path))
    redelivered = restarted.release_due(now="2026-08-12T10:00:00+08:00")
    assert [child.child_id for child in redelivered] == [released[0].child_id]


def test_vwap_requires_profile_and_allocates_by_forecast_volume(tmp_path: Path) -> None:
    with pytest.raises(ParentExecutionError, match="requires.*volume profile|volume profile"):
        ParentOrderSpec(
            parent_id="bad-vwap",
            symbol="600000.SH",
            side="buy",
            total_quantity=1_000,
            algorithm=ExecutionAlgorithm.VWAP,
            schedule_times=SCHEDULE,
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


def test_pov_is_volume_driven_monotonic_and_restart_safe(tmp_path: Path) -> None:
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
        cumulative_market_volume=4_000,
    )
    assert len(same) == 1
    assert same[0].child_id == first[0].child_id

    with pytest.raises(ParentExecutionError, match="moved backwards"):
        restarted.release_due(
            now="2026-08-12T09:42:00+08:00",
            cumulative_market_volume=3_900,
        )

    restarted.acknowledge(
        first[0].child_id,
        status=ChildStatus.FILLED,
        filled_quantity=400,
    )
    next_children = restarted.release_due(
        now="2026-08-12T10:00:00+08:00",
        cumulative_market_volume=10_000,
    )
    active = [child for child in next_children if child.status is ChildStatus.RELEASED]
    assert sum(child.quantity - child.filled_quantity for child in active) == 600
    assert restarted.state().committed_quantity == 1_000


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

    engine.acknowledge(first[0].child_id, status=ChildStatus.FILLED, filled_quantity=300)
    second = engine.release_due(now="2026-08-12T09:37:00+08:00")
    active = [child for child in second if child.status in {ChildStatus.RELEASED, ChildStatus.PARTIAL}]
    assert len(active) == 1
    assert active[0].quantity == 300
    assert active[0].child_id != first[0].child_id


def test_cancelled_partial_child_only_commits_actual_fill(tmp_path: Path) -> None:
    parent = ParentOrderSpec(
        parent_id="p-pov-partial",
        symbol="600000.SH",
        side="buy",
        total_quantity=1_000,
        algorithm=ExecutionAlgorithm.POV,
        participation_rate=0.10,
    )
    engine = ParentChildExecutionEngine(parent, _store(tmp_path))
    child = engine.release_due(
        now="2026-08-12T09:40:00+08:00", cumulative_market_volume=5_000
    )[0]
    engine.acknowledge(child.child_id, status=ChildStatus.CANCELLED, filled_quantity=200)
    state = engine.state()
    assert state.filled_quantity == 200
    assert state.outstanding_quantity == 0

    released = engine.release_due(
        now="2026-08-12T09:45:00+08:00", cumulative_market_volume=5_000
    )
    outstanding = [row for row in released if row.status is ChildStatus.RELEASED]
    assert outstanding[-1].quantity == 300


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
    text = store.path.read_text(encoding="utf-8").replace('"total_quantity": 1000', '"total_quantity": 2000')
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
    engine.acknowledge(child.child_id, status=ChildStatus.PARTIAL, filled_quantity=300)
    with pytest.raises(ParentExecutionError, match="moved backwards"):
        engine.acknowledge(child.child_id, status=ChildStatus.PARTIAL, filled_quantity=200)
    with pytest.raises(ParentExecutionError, match="exceeds child quantity"):
        engine.acknowledge(child.child_id, status=ChildStatus.FILLED, filled_quantity=600)
