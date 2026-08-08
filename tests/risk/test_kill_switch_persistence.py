from __future__ import annotations

import json

import pytest

from quantagent.risk.kill_switch import KillSwitch


def test_persistent_kill_switch_survives_restart(tmp_path) -> None:
    path = tmp_path / "risk" / "kill_switch.json"
    first = KillSwitch(state_path=path)
    first.trigger("severe_reconciliation_mismatch")
    assert first.triggered is True

    restarted = KillSwitch(state_path=path)
    assert restarted.triggered is True
    assert "severe_reconciliation_mismatch" in restarted.reasons


def test_persistent_kill_switch_cannot_be_cleared_without_reason(tmp_path) -> None:
    switch = KillSwitch(state_path=tmp_path / "kill.json")
    switch.trigger("data_provider_failure")
    with pytest.raises(ValueError, match="explicit release reason"):
        switch.release()
    assert switch.triggered is True


def test_explicit_single_reason_release_is_persisted(tmp_path) -> None:
    path = tmp_path / "kill.json"
    switch = KillSwitch(state_path=path)
    switch.trigger("data_provider_failure")
    switch.trigger("audit_write_failure")
    switch.release("data_provider_failure")

    restarted = KillSwitch(state_path=path)
    assert "data_provider_failure" not in restarted.reasons
    assert "audit_write_failure" in restarted.reasons
    assert restarted.triggered is True


def test_corrupt_state_fails_closed(tmp_path) -> None:
    path = tmp_path / "kill.json"
    path.write_text("{not-json", encoding="utf-8")
    switch = KillSwitch(state_path=path)
    assert switch.triggered is True
    assert switch.manual_triggered is True
    assert switch.reasons == ["kill_switch_state_unreadable"]
    # The corrupt file is replaced by an auditable fail-closed state.
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["manual_triggered"] is True


def test_research_in_memory_release_keeps_backward_compatibility() -> None:
    switch = KillSwitch()
    switch.trigger("research_test")
    switch.release()
    assert switch.triggered is False
