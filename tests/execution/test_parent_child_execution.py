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

SESSION = "2026-08-12"
SCHEDULE = (
    "2026-08-12T09:35:00+08:00",
    "2026-08-12T10:30:00+08:00",
    "2026-08-12T13:30:00+08:00",
    "2026-08-12T14:55:00+08:00",
)


def _store(tmp_path: Path) -> ParentExecutionStore:
    return ParentExecutionStore(tmp_path / "parent.json")


def _parent(**kwargs) -> ParentOrderSpec:
    base = dict(
        parent_id="p",
        session_date=SESSION,
        symbol="600000.SH",
        side="buy",
        total_quantity=1_000,
        algorithm=ExecutionAlgorithm.TWAP,
        schedule_times=SCHEDULE,
    )
    base.update(kwargs)
    return ParentOrderSpec(**base)


def test_twap_allocates_exact_parent_and_restart_redelivers_same_child(tmp_path: Path) -> None:
    parent = _parent(parent_id="p-twap")
    engine = ParentChildExecutionEngine(parent, _store(tmp_path))
    assert [row.quantity for row in engine.state().children] == [300, 300, 200, 200]

    first = engine.release_due(now="2026-08-12T10:00:00+08:00")
    assert len(first) == 1
    restarted = ParentChildExecutionEngine(parent, _store(tmp_path))
    redelivery = restarted.release_due(now="2026-08-12T10:00:00+08:00")
    assert [row.child_id for row in redelivery] == [first[0].child_id]
    assert child_lineage_id(parent.parent_id, first[0].child_id).endswith(first[0].child_id)


def test_parent_and_schedule_are_bound_to_one_trading_session(tmp_path: Path) -> None:
    with pytest.raises(ParentExecutionError, match="session_date"):
        _parent(session_date="not-a-date").canonical()
    with pytest.raises(ParentExecutionError, match="parent session_date"):
        _parent(
            schedule_times=(
                "2026-08-12T09:35:00+08:00",
                "2026-08-13T10:30:00+08:00",
            )
        ).canonical()

    engine = ParentChildExecutionEngine(_parent(), _store(tmp_path))
    with pytest.raises(ParentExecutionError, match="outside immutable parent session_date"):
        engine.release_due(now="2026-08-13T09:35:00+08:00")


def test_vwap_requires_finite_forecast_profile(tmp_path: Path) -> None:
    for profile in [(), (0.1, 0.2, float("inf"), 0.7), (0.1, 0.2, float("nan"), 0.7)]:
        with pytest.raises(ParentExecutionError, match="volume profile"):
            _parent(
                algorithm=ExecutionAlgorithm.VWAP,
                volume_profile=profile,
            ).canonical()

    parent = _parent(
        parent_id="p-vwap",
        algorithm=ExecutionAlgorithm.VWAP,
        volume_profile=(0.1, 0.2, 0.3, 0.4),
    )
    state = ParentChildExecutionEngine(parent, _store(tmp_path)).state()
    assert [row.quantity for row in state.children] == [100, 200, 300, 400]


def test_pov_is_session_monotonic_restart_safe_and_single_active(tmp_path: Path) -> None:
    parent = _parent(
        parent_id="p-pov",
        total_quantity=2_000,
        algorithm=ExecutionAlgorithm.POV,
        schedule_times=(),
        participation_rate=0.10,
        max_child_quantity=500,
    )
    engine = ParentChildExecutionEngine(parent, _store(tmp_path))
    first = engine.release_due(
        now="2026-08-12T09:40:00+08:00",
        cumulative_market_volume=4_000,
    )
    assert len(first) == 1 and first[0].quantity == 400

    restarted = ParentChildExecutionEngine(parent, _store(tmp_path))
    same = restarted.release_due(
        now="2026-08-12T09:41:00+08:00",
        cumulative_market_volume=10_000,
    )
    assert len(same) == 1 and same[0].child_id == first[0].child_id
    assert restarted.state().last_observed_cumulative_volume == 10_000

    with pytest.raises(ParentExecutionError, match="moved backwards"):
        restarted.release_due(
            now="2026-08-12T09:42:00+08:00",
            cumulative_market_volume=9_900,
        )

    restarted.acknowledge(first[0].child_id, status=ChildStatus.FILLED, filled_quantity=400)
    second = restarted.release_due(
        now="2026-08-12T10:00:00+08:00",
        cumulative_market_volume=10_000,
    )
    assert len(second) == 1 and second[0].quantity == 500


def test_partial_child_is_not_redelivered_as_full_size(tmp_path: Path) -> None:
    parent = _parent(
        parent_id="p-partial",
        algorithm=ExecutionAlgorithm.POV,
        schedule_times=(),
        participation_rate=0.10,
    )
    engine = ParentChildExecutionEngine(parent, _store(tmp_path))
    child = engine.release_due(
        now="2026-08-12T09:40:00+08:00", cumulative_market_volume=5_000
    )[0]
    engine.acknowledge(child.child_id, status=ChildStatus.PARTIAL, filled_quantity=200)
    assert engine.released_children() == ()
    assert engine.release_due(
        now="2026-08-12T09:45:00+08:00", cumulative_market_volume=8_000
    ) == ()
    assert engine.state().outstanding_quantity == 300


def test_cancelled_partial_child_allows_residual_replan_after_terminal(tmp_path: Path) -> None:
    parent = _parent(
        parent_id="p-cancel",
        algorithm=ExecutionAlgorithm.POV,
        schedule_times=(),
        participation_rate=0.10,
    )
    engine = ParentChildExecutionEngine(parent, _store(tmp_path))
    child = engine.release_due(
        now="2026-08-12T09:40:00+08:00", cumulative_market_volume=5_000
    )[0]
    engine.acknowledge(child.child_id, status=ChildStatus.CANCELLED, filled_quantity=200)
    next_child = engine.release_due(
        now="2026-08-12T09:45:00+08:00", cumulative_market_volume=5_000
    )
    assert len(next_child) == 1 and next_child[0].quantity == 300


def test_iceberg_has_one_active_display_child(tmp_path: Path) -> None:
    parent = _parent(
        parent_id="p-ice",
        side="sell",
        algorithm=ExecutionAlgorithm.ICEBERG,
        schedule_times=(),
        display_quantity=300,
    )
    engine = ParentChildExecutionEngine(parent, _store(tmp_path))
    first = engine.release_due(now="2026-08-12T09:35:00+08:00")
    assert len(first) == 1 and first[0].quantity == 300
    assert engine.release_due(now="2026-08-12T09:36:00+08:00")[0].child_id == first[0].child_id
    engine.acknowledge(first[0].child_id, status=ChildStatus.FILLED, filled_quantity=300)
    second = engine.release_due(now="2026-08-12T09:37:00+08:00")
    assert len(second) == 1 and second[0].child_id != first[0].child_id


def test_acknowledgement_state_machine_is_strict_and_terminal_immutable(tmp_path: Path) -> None:
    parent = _parent(
        parent_id="p-ack",
        algorithm=ExecutionAlgorithm.ICEBERG,
        schedule_times=(),
        display_quantity=500,
    )
    engine = ParentChildExecutionEngine(parent, _store(tmp_path))
    child = engine.release_due(now="2026-08-12T09:35:00+08:00")[0]

    with pytest.raises(ParentExecutionError, match="REJECTED child must have zero"):
        engine.acknowledge(child.child_id, status=ChildStatus.REJECTED, filled_quantity=1)
    with pytest.raises(ParentExecutionError, match="RELEASED acknowledgement requires zero"):
        engine.acknowledge(child.child_id, status=ChildStatus.RELEASED, filled_quantity=1)
    with pytest.raises(ParentExecutionError, match="FILLED child"):
        engine.acknowledge(child.child_id, status=ChildStatus.FILLED, filled_quantity=400)

    engine.acknowledge(child.child_id, status=ChildStatus.CANCELLED, filled_quantity=200)
    engine.acknowledge(child.child_id, status=ChildStatus.CANCELLED, filled_quantity=200)
    with pytest.raises(ParentExecutionError, match="terminal child state"):
        engine.acknowledge(child.child_id, status=ChildStatus.REJECTED, filled_quantity=0)


def test_parent_economics_are_immutable_and_tampering_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    parent = _parent(parent_id="p-immutable")
    ParentChildExecutionEngine(parent, store)
    with pytest.raises(ParentExecutionConflict):
        ParentChildExecutionEngine(
            _parent(parent_id="p-immutable", total_quantity=2_000),
            store,
        )

    text = store.path.read_text(encoding="utf-8").replace(
        '"total_quantity": 1000', '"total_quantity": 2000'
    )
    store.path.write_text(text, encoding="utf-8")
    with pytest.raises(ParentExecutionCorruption, match="digest mismatch"):
        store.read()


def test_duplicate_json_key_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
    with pytest.raises(ParentExecutionCorruption, match="duplicate JSON key"):
        store.read()
