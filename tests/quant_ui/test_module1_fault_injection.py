"""Run the fault-injection harness as part of the suite (M1-11).

A chaos harness nobody runs is documentation. This drives
`scripts/module1_fault_injection.py` in-process so every real `SIGKILL`, every
storage fault and the lock-contention race are checked on every full-suite run,
and a regression in any of them fails here rather than in a script someone
remembers to invoke.

It spawns real child processes and kills them, so it is slower than the rest of
the suite. That cost is the point: an injected exception unwinds the stack and
flushes buffers, and a SIGKILL does not.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_harness():
    """Import the script by path; `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "module1_fault_injection", PROJECT_ROOT / "scripts" / "module1_fault_injection.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def experiments():
    harness = _load_harness()
    return harness, harness.finalise(harness.run_all())


def test_every_injected_fault_leaves_the_invariants_intact(experiments):
    _, results = experiments
    failures = {
        experiment.experiment_id: experiment.violations
        for experiment in results
        if not experiment.passed
    }
    assert not failures, failures


def test_the_control_run_actually_traded(experiments):
    """Without this the whole suite could pass by never executing anything."""
    _, results = experiments
    control = next(e for e in results if e.experiment_id == "sigkill.none")
    assert control.measurement.fills == 1
    assert control.measurement.orders == 1
    assert control.measurement.chain_valid


def test_the_process_faults_were_delivered_by_a_real_signal(experiments):
    """Guards against the harness quietly degrading to an injected exception."""
    harness, results = experiments
    import signal

    killed = [
        e for e in results
        if e.experiment_id.startswith("sigkill.") and e.experiment_id != "sigkill.none"
    ]
    assert len(killed) == len(harness.KILL_POINTS) - 1
    for experiment in killed:
        assert experiment.child_signal == signal.SIGKILL, (
            f"{experiment.experiment_id} was not killed by SIGKILL "
            f"(signal={experiment.child_signal})"
        )


def test_a_signal_mid_append_produces_a_recoverable_torn_tail(experiments):
    """A real partial write, not a test truncating a file itself."""
    _, results = experiments
    torn = next(e for e in results if e.experiment_id == "sigkill.mid_ledger_append")
    assert torn.measurement.torn_tail is True
    assert torn.measurement.chain_valid is True, (
        "everything before the torn record must remain verifiable"
    )
    assert torn.measurement.fills == 0, "a half-written record must not book money"


def test_a_tampered_chain_is_refused_rather_than_replayed(experiments):
    _, results = experiments
    for experiment_id in ("integrity.missing_record", "integrity.corrupted_record"):
        experiment = next(e for e in results if e.experiment_id == experiment_id)
        assert experiment.measurement.chain_valid is False
        assert experiment.measurement.orders == 0, "a projection was exposed anyway"


def test_the_storage_faults_ran_against_a_chain_that_already_held_records(experiments):
    """Otherwise they prove only that an error surfaces, not that data survives."""
    _, results = experiments
    for experiment_id in (
        "storage.fsync_failure", "storage.disk_full", "storage.read_only_path"
    ):
        experiment = next(e for e in results if e.experiment_id == experiment_id)
        assert experiment.measurement.fills == 1, (
            f"{experiment_id} had nothing committed to survive the fault"
        )
        assert experiment.measurement.chain_valid is True
        assert any("seeded" in note for note in experiment.measurement.notes)


def test_no_fault_ever_produced_a_second_order_for_one_intent(experiments):
    _, results = experiments
    for experiment in results:
        assert experiment.measurement.max_orders_per_intent <= 1, (
            f"{experiment.experiment_id} duplicated an economic order"
        )


def test_concurrent_processes_terminate_and_produce_one_order(experiments):
    _, results = experiments
    contention = next(
        e for e in results if e.experiment_id == "lock.contention.four_processes"
    )
    assert contention.passed
    assert contention.measurement.orders == 1
    assert contention.measurement.fills == 1
