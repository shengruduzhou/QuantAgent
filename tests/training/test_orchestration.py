"""Training run lifecycle, liveness proof, preflight and checkpoint integrity.

The tests are organised around the ways a training system misreports itself: a
stale PID read as a running job, an unreviewed configuration reaching the GPU
queue, a rebuilt dataset silently producing incomparable numbers, or a truncated
checkpoint that still loads.
"""

from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timedelta, timezone

import pytest

from quantagent.training import orchestration as orch


def _manifest(**kw) -> orch.RunManifest:
    base = dict(
        run_id="run-1", experiment_id="exp-1", model_family="lightgbm",
        horizon="short_5d", seed=42, dataset_path="runtime/data/gold/full_universe/dataset.parquet",
        dataset_hash="aaaa", schema_hash="bbbb", feature_hash="cccc",
        label_hash="dddd", fold_hash="eeee",
        configuration={"epochs": 1, "batch_size": 512},
    )
    base.update(kw)
    return orch.RunManifest(**base)


@pytest.fixture
def run(tmp_path):
    return orch.TrainingRun(_manifest(), root=tmp_path)


def _advance(run: orch.TrainingRun) -> None:
    """Take a run from DRAFT to ARMED through the legal path."""
    run.validate({"dataset_exists": True})
    run.freeze()
    run.arm(confirmed_hash=run.manifest.configuration_hash)


class TestLifecycleTransitions:
    def test_happy_path(self, run):
        _advance(run)
        run.launch(pid=os.getpid(), host="localhost", gpu="RTX3090")
        assert run.manifest.status == orch.RUNNING
        run.checkpoint("/tmp/ckpt.pkl", epoch=1, metric=0.5)
        assert run.manifest.status == orch.RUNNING
        run.complete()
        assert run.manifest.status == orch.COMPLETED

    def test_illegal_transition_refused(self, run):
        with pytest.raises(orch.LifecycleError, match="illegal transition"):
            run.transition(orch.RUNNING)

    def test_cannot_arm_without_freezing(self, run):
        run.validate({"ok": True})
        with pytest.raises(orch.LifecycleError, match="freeze the configuration first"):
            run.arm(confirmed_hash="whatever")

    def test_cannot_launch_without_arming(self, run):
        run.validate({"ok": True})
        run.freeze()
        with pytest.raises(orch.LifecycleError, match="must be ARMED"):
            run.launch(pid=1234, host="localhost")

    def test_arming_requires_the_frozen_hash(self, run):
        """An operator cannot arm a configuration they have not actually seen."""
        run.validate({"ok": True})
        run.freeze()
        with pytest.raises(orch.LifecycleError, match="does not match the frozen"):
            run.arm(confirmed_hash="0000000000000000")

    def test_editing_after_freeze_returns_to_draft(self, run):
        run.validate({"ok": True})
        run.freeze()
        run.transition(orch.DRAFT, reason="configuration edited")
        assert run.manifest.status == orch.DRAFT

    def test_terminal_states_are_terminal(self, run):
        _advance(run)
        run.launch(pid=os.getpid(), host="h")
        run.complete()
        with pytest.raises(orch.LifecycleError):
            run.transition(orch.RUNNING)

    def test_pause_and_resume(self, run):
        _advance(run)
        run.launch(pid=os.getpid(), host="h")
        run.checkpoint("/tmp/c.pkl", epoch=2)
        run.pause()
        assert run.manifest.status == orch.PAUSED
        run.resume()
        assert run.manifest.status == orch.RUNNING

    def test_resume_without_checkpoint_refused(self, run):
        _advance(run)
        run.launch(pid=os.getpid(), host="h")
        run.pause()
        with pytest.raises(orch.LifecycleError, match="no checkpoint"):
            run.resume()

    def test_cancel_from_running(self, run):
        _advance(run)
        run.launch(pid=os.getpid(), host="h")
        run.cancel()
        assert run.manifest.status == orch.CANCELLED

    def test_quarantine_records_reason(self, run):
        _advance(run)
        run.launch(pid=os.getpid(), host="h")
        run.quarantine("suspected leakage")
        assert run.manifest.status == orch.QUARANTINED
        assert run.manifest.failure_reason == "suspected leakage"

    def test_history_records_every_transition(self, run):
        _advance(run)
        transitions = [(h["from"], h["to"]) for h in run.manifest.history]
        assert (orch.DRAFT, orch.VALIDATING) in transitions
        assert (orch.VALIDATED, orch.FROZEN) in transitions
        assert (orch.FROZEN, orch.ARMED) in transitions


class TestLivenessNotJustPid:
    def test_running_with_fresh_heartbeat_is_alive(self, run):
        _advance(run)
        run.launch(pid=os.getpid(), host="h")
        assert run.manifest.liveness()["verdict"] == "ALIVE"
        assert run.manifest.liveness()["observably_running"] is True

    def test_stale_pid_is_not_running(self, run):
        """The core guarantee: a PID file alone proves nothing."""
        _advance(run)
        run.launch(pid=os.getpid(), host="h")
        # A PID that certainly does not exist.
        run.manifest.pid = 999_999_999
        liveness = run.manifest.liveness()
        assert liveness["verdict"] == "STALE_PID"
        assert liveness["observably_running"] is False

    def test_stale_heartbeat_is_not_running_even_with_live_process(self, run):
        _advance(run)
        run.launch(pid=os.getpid(), host="h")
        old = datetime.now(timezone.utc) - timedelta(hours=2)
        run.manifest.heartbeat = old.isoformat(timespec="seconds")
        liveness = run.manifest.liveness()
        assert liveness["process_alive"] is True
        assert liveness["verdict"] == "STALE_HEARTBEAT"
        assert liveness["observably_running"] is False

    def test_non_active_status_is_reported_as_such(self, run):
        assert run.manifest.liveness()["verdict"] == "NOT_ACTIVE"

    def test_liveness_states_the_pid_caveat(self, run):
        assert "a PID alone is not proof" in run.manifest.liveness()["note"]


class TestPreflight:
    def _args(self, tmp_path, **kw):
        dataset = tmp_path / "dataset.parquet"
        dataset.write_bytes(b"x")
        base = dict(
            dataset_path=dataset, expected_dataset_hash="aaaa",
            actual_dataset_hash="aaaa", expected_schema_hash="bbbb",
            actual_schema_hash="bbbb", folds=[{"fold": 0}],
            train_rows=1000, validation_rows=100, output_dir=tmp_path,
            min_free_bytes=1,
        )
        base.update(kw)
        return base

    def test_clean_preflight_passes(self, tmp_path):
        checks, _ = orch.preflight_checks(**self._args(tmp_path))
        assert all(checks.values())

    def test_dataset_hash_mismatch_detected(self, tmp_path):
        checks, details = orch.preflight_checks(
            **self._args(tmp_path, actual_dataset_hash="zzzz"))
        assert checks["dataset_hash_matches"] is False
        assert "rebuilt since this configuration was frozen" in details["dataset_hash_matches"]

    def test_schema_drift_detected(self, tmp_path):
        checks, _ = orch.preflight_checks(
            **self._args(tmp_path, actual_schema_hash="zzzz"))
        assert checks["schema_hash_matches"] is False

    def test_empty_train_fold_detected(self, tmp_path):
        checks, _ = orch.preflight_checks(**self._args(tmp_path, train_rows=0))
        assert checks["no_empty_train_fold"] is False

    def test_empty_validation_fold_detected(self, tmp_path):
        checks, _ = orch.preflight_checks(**self._args(tmp_path, validation_rows=0))
        assert checks["no_empty_validation_fold"] is False

    def test_missing_folds_detected(self, tmp_path):
        checks, _ = orch.preflight_checks(**self._args(tmp_path, folds=[]))
        assert checks["folds_defined"] is False

    def test_insufficient_disk_detected(self, tmp_path):
        checks, _ = orch.preflight_checks(
            **self._args(tmp_path, min_free_bytes=1 << 60))
        assert checks["sufficient_disk"] is False

    def test_gpu_required_but_absent(self, tmp_path):
        checks, _ = orch.preflight_checks(
            **self._args(tmp_path, require_gpu=True, gpu_available=False))
        assert checks["gpu_available_if_required"] is False

    def test_failed_preflight_fails_the_run(self, run):
        with pytest.raises(orch.PreflightError, match="preflight failed"):
            run.validate({"dataset_exists": False},
                         details={"dataset_exists": "not found"})
        assert run.manifest.status == orch.FAILED
        assert "preflight failed" in run.manifest.failure_reason


class TestCheckpointIntegrity:
    def test_atomic_write_and_verified_reload(self, tmp_path):
        path = tmp_path / "ckpt.pkl"
        digest = orch.write_checkpoint_atomically({"epoch": 3, "weights": [1, 2]}, path)
        assert path.exists()
        assert (tmp_path / "ckpt.pkl.sha256").read_text().strip() == digest
        payload = orch.load_checkpoint_verified(path)
        assert payload["epoch"] == 3

    def test_no_temporary_file_survives(self, tmp_path):
        path = tmp_path / "ckpt.pkl"
        orch.write_checkpoint_atomically({"a": 1}, path)
        assert not list(tmp_path.glob("*.tmp"))

    def test_corrupt_checkpoint_refused(self, tmp_path):
        path = tmp_path / "ckpt.pkl"
        orch.write_checkpoint_atomically({"epoch": 1}, path)
        path.write_bytes(pickle.dumps({"epoch": 99}))  # digest no longer matches
        with pytest.raises(ValueError, match="corrupt"):
            orch.load_checkpoint_verified(path)

    def test_missing_checkpoint_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            orch.load_checkpoint_verified(tmp_path / "nope.pkl")

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_non_finite_loss_stops_training(self, value):
        with pytest.raises(orch.PreflightError) as exc:
            orch.guard_loss(value, epoch=2)
        # The machine-readable code lives in .failures; the message is for humans.
        assert exc.value.failures == ["non_finite_loss"]
        assert "epoch 2" in str(exc.value)

    def test_finite_loss_passes(self):
        orch.guard_loss(0.42, epoch=1)


class TestPersistenceAndClone:
    def test_manifest_round_trip(self, run, tmp_path):
        _advance(run)
        reloaded = orch.TrainingRun.load(run.manifest_path)
        assert reloaded.manifest.run_id == "run-1"
        assert reloaded.manifest.status == orch.ARMED
        assert reloaded.manifest.configuration_hash == run.manifest.configuration_hash

    def test_configuration_hash_covers_data_hashes(self, tmp_path):
        first = orch.TrainingRun(_manifest(), root=tmp_path)
        second = orch.TrainingRun(_manifest(run_id="run-2", dataset_hash="different"),
                                  root=tmp_path)
        assert (first.manifest.compute_configuration_hash()
                != second.manifest.compute_configuration_hash())

    def test_clone_does_not_inherit_results(self, run, tmp_path):
        _advance(run)
        run.launch(pid=os.getpid(), host="h")
        run.checkpoint("/tmp/c.pkl", epoch=5, metric=0.1)
        clone = run.clone(run_id="run-2")
        assert clone.manifest.status == orch.DRAFT
        assert clone.manifest.checkpoint_path is None
        assert clone.manifest.best_metric is None
        assert clone.manifest.latest_epoch == 0
        assert clone.manifest.pid is None
        assert clone.manifest.history == []

    def test_save_is_atomic(self, run, tmp_path):
        run.save()
        assert not list(tmp_path.glob("*.tmp"))
        json.loads(run.manifest_path.read_text(encoding="utf-8"))
