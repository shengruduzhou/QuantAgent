from __future__ import annotations

import pandas as pd

from quantagent.portfolio.position_age_tracker import (
    UNKNOWN_INITIAL_HORIZON_DAYS,
    PositionAgeTracker,
)


def test_begin_session_drops_persisted_symbol_absent_from_authoritative_holdings(tmp_path) -> None:
    state_path = tmp_path / "position_age.parquet"
    tracker = PositionAgeTracker(state_path=state_path)
    tracker.record_session(
        pd.Timestamp("2026-08-03"),
        {"SOLD": 0.20, "HELD": 0.30},
        {"SOLD": 60, "HELD": 60},
    )
    tracker.record_session(
        pd.Timestamp("2026-08-04"),
        {"SOLD": 0.20, "HELD": 0.30},
        {},
    )
    tracker.persist()

    restarted = PositionAgeTracker.from_state(state_path)
    assert restarted.is_locked("SOLD", pd.Timestamp("2026-08-05"))

    # Canonical account recovery is authoritative: SOLD was actually liquidated
    # before restart, so its persisted lifecycle history must not survive into
    # the first-date lock calculation. HELD keeps its historical age/horizon.
    restarted.begin_session({"HELD": 0.30})
    snapshot = restarted.snapshot().set_index("symbol")

    assert "SOLD" not in snapshot.index
    assert "HELD" in snapshot.index
    assert int(snapshot.loc["HELD", "expected_horizon_days"]) == 60
    assert not restarted.is_locked("SOLD", pd.Timestamp("2026-08-05"))
    assert restarted.is_locked("HELD", pd.Timestamp("2026-08-05"))

    # If SOLD is selected again after restart it is a genuinely new position,
    # not a continuation of stale persisted age.
    restarted.record_session(
        pd.Timestamp("2026-08-05"),
        {"HELD": 0.30, "SOLD": 0.10},
        {"SOLD": 20},
    )
    assert restarted.age_for("SOLD", pd.Timestamp("2026-08-05")) == 0
    assert restarted.is_locked("SOLD", pd.Timestamp("2026-08-05"))


def test_explicit_empty_authoritative_holdings_clear_all_persisted_records(tmp_path) -> None:
    state_path = tmp_path / "position_age.parquet"
    tracker = PositionAgeTracker(state_path=state_path)
    tracker.record_session(
        pd.Timestamp("2026-08-03"),
        {"A": 0.20, "B": 0.30},
        {"A": 20, "B": 60},
    )
    tracker.persist()

    restarted = PositionAgeTracker.from_state(state_path)
    restarted.begin_session({})

    assert restarted.snapshot().empty


def test_real_horizon_replaces_unknown_restart_sentinel_before_lock(tmp_path) -> None:
    state_path = tmp_path / "position_age.parquet"
    tracker = PositionAgeTracker(state_path=state_path)
    tracker.begin_session({"HELD": 0.25})
    first = tracker.snapshot().set_index("symbol")
    assert int(first.loc["HELD", "expected_horizon_days"]) == UNKNOWN_INITIAL_HORIZON_DAYS
    tracker.persist()

    restarted = PositionAgeTracker.from_state(state_path)
    restarted.begin_session({"HELD": 0.25}, {"HELD": 5})
    updated = restarted.snapshot().set_index("symbol")
    assert int(updated.loc["HELD", "expected_horizon_days"]) == 5
