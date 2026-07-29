"""Production training run lifecycle: states, manifest, and liveness proof.

The property this module exists to guarantee: **a PID is not proof that a job is
running.** A stale PID file, a process that was OOM-killed, or a host that
rebooted all leave a record that looks alive. So liveness requires a *fresh
heartbeat* as well as a live process, and a run whose heartbeat has expired is
reported as such rather than as RUNNING.

The second guarantee is that a run is reproducible from its manifest alone. Every
hash a result depends on -- dataset, schema, features, labels, folds,
configuration -- is recorded at freeze time, so a later run that silently reads a
rebuilt dataset fails the hash check instead of producing incomparable numbers.

State machine::

    DRAFT -> VALIDATING -> VALIDATED -> FROZEN -> ARMED -> QUEUED -> STARTING
          -> RUNNING <-> CHECKPOINTING
          -> PAUSING -> PAUSED -> RESUMING -> RUNNING
          -> COMPLETED | FAILED | CANCELLED | QUARANTINED

Transitions are enforced. Arming an unfrozen configuration, or launching an
unarmed run, raises rather than being permitted "just this once" -- those are
precisely the shortcuts that put an unreviewed configuration into a GPU queue.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# --- lifecycle states -------------------------------------------------------
DRAFT = "DRAFT"
VALIDATING = "VALIDATING"
VALIDATED = "VALIDATED"
FROZEN = "FROZEN"
ARMED = "ARMED"
QUEUED = "QUEUED"
STARTING = "STARTING"
RUNNING = "RUNNING"
CHECKPOINTING = "CHECKPOINTING"
PAUSING = "PAUSING"
PAUSED = "PAUSED"
RESUMING = "RESUMING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"
QUARANTINED = "QUARANTINED"

STATES: tuple[str, ...] = (
    DRAFT, VALIDATING, VALIDATED, FROZEN, ARMED, QUEUED, STARTING, RUNNING,
    CHECKPOINTING, PAUSING, PAUSED, RESUMING, COMPLETED, FAILED, CANCELLED,
    QUARANTINED,
)
TERMINAL_STATES: frozenset[str] = frozenset(
    {COMPLETED, FAILED, CANCELLED, QUARANTINED}
)
#: States in which a process should exist.
ACTIVE_STATES: frozenset[str] = frozenset(
    {STARTING, RUNNING, CHECKPOINTING, PAUSING, RESUMING}
)

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    DRAFT: frozenset({VALIDATING, CANCELLED}),
    VALIDATING: frozenset({VALIDATED, FAILED, DRAFT}),
    VALIDATED: frozenset({FROZEN, DRAFT, CANCELLED}),
    # Freezing pins the configuration hash; editing sends it back to DRAFT so a
    # frozen hash can never describe a configuration that has since changed.
    FROZEN: frozenset({ARMED, DRAFT, CANCELLED}),
    ARMED: frozenset({QUEUED, FROZEN, CANCELLED}),
    QUEUED: frozenset({STARTING, CANCELLED}),
    STARTING: frozenset({RUNNING, FAILED, CANCELLED}),
    RUNNING: frozenset({CHECKPOINTING, PAUSING, COMPLETED, FAILED, CANCELLED, QUARANTINED}),
    CHECKPOINTING: frozenset({RUNNING, FAILED, CANCELLED, QUARANTINED}),
    PAUSING: frozenset({PAUSED, FAILED, CANCELLED}),
    PAUSED: frozenset({RESUMING, CANCELLED, QUARANTINED}),
    RESUMING: frozenset({RUNNING, FAILED, CANCELLED}),
    COMPLETED: frozenset(),
    FAILED: frozenset({QUARANTINED}),
    CANCELLED: frozenset(),
    QUARANTINED: frozenset(),
}

#: A heartbeat older than this means the run is not observably alive.
HEARTBEAT_TIMEOUT_SECONDS = 180


class LifecycleError(RuntimeError):
    """Raised on an illegal lifecycle transition."""


class PreflightError(RuntimeError):
    """Raised when a run may not proceed. Carries the failed checks."""

    def __init__(self, failures: Sequence[str], message: str) -> None:
        super().__init__(message)
        self.failures = list(failures)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


@dataclass
class RunManifest:
    """Everything needed to reproduce, audit or recover a training run."""

    run_id: str
    experiment_id: str
    model_family: str
    horizon: str
    seed: int
    source_commit: str = "unknown"
    dataset_path: str = ""
    dataset_hash: str = ""
    schema_hash: str = ""
    feature_hash: str = ""
    label_hash: str = ""
    fold_hash: str = ""
    configuration_hash: str = ""
    configuration: dict[str, Any] = field(default_factory=dict)
    host: str = ""
    gpu: str | None = None
    pid: int | None = None
    heartbeat: str | None = None
    started_at: str | None = None
    updated_at: str = field(default_factory=_now)
    checkpoint_path: str | None = None
    latest_epoch: int = 0
    best_epoch: int | None = None
    best_metric: float | None = None
    status: str = DRAFT
    failure_reason: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -- configuration freezing -------------------------------------------
    def compute_configuration_hash(self) -> str:
        return _hash_payload({
            "model_family": self.model_family, "horizon": self.horizon,
            "seed": self.seed, "configuration": self.configuration,
            "dataset_hash": self.dataset_hash, "schema_hash": self.schema_hash,
            "feature_hash": self.feature_hash, "label_hash": self.label_hash,
            "fold_hash": self.fold_hash,
        })

    # -- liveness ----------------------------------------------------------
    @property
    def heartbeat_age_seconds(self) -> float | None:
        if not self.heartbeat:
            return None
        try:
            beat = datetime.fromisoformat(self.heartbeat)
        except ValueError:
            return None
        if beat.tzinfo is None:
            beat = beat.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - beat).total_seconds()

    def process_alive(self) -> bool:
        """Whether the recorded PID currently exists.

        Necessary but NOT sufficient for liveness: a recycled PID belongs to a
        different process entirely, which is why :meth:`liveness` also requires
        a fresh heartbeat.
        """
        if self.pid is None:
            return False
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists but owned by another user
        return True

    def liveness(self, *, timeout: int = HEARTBEAT_TIMEOUT_SECONDS) -> dict[str, Any]:
        """Report whether this run is observably alive, and why or why not."""
        age = self.heartbeat_age_seconds
        alive_process = self.process_alive()
        fresh_heartbeat = age is not None and age <= timeout
        should_be_active = self.status in ACTIVE_STATES

        if not should_be_active:
            verdict = "NOT_ACTIVE"
        elif alive_process and fresh_heartbeat:
            verdict = "ALIVE"
        elif alive_process and not fresh_heartbeat:
            verdict = "STALE_HEARTBEAT"
        elif not alive_process and self.pid is not None:
            verdict = "STALE_PID"
        else:
            verdict = "NO_PROCESS"

        return {
            "verdict": verdict,
            "status": self.status,
            "pid": self.pid,
            "process_alive": alive_process,
            "heartbeat": self.heartbeat,
            "heartbeat_age_seconds": age,
            "heartbeat_timeout_seconds": timeout,
            "observably_running": verdict == "ALIVE",
            "note": (
                "a PID alone is not proof of a running job; a recycled PID and a "
                "stale heartbeat both present as an existing process"
            ),
        }


class TrainingRun:
    """A single governed training run and its persisted lifecycle."""

    def __init__(self, manifest: RunManifest, *, root: str | Path) -> None:
        self.manifest = manifest
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- persistence -------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.root / f"{self.manifest.run_id}.manifest.json"

    def save(self) -> None:
        """Atomic write: a crash mid-save must not truncate the manifest."""
        self.manifest.updated_at = _now()
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.manifest.to_dict(), ensure_ascii=False, indent=2,
                       default=str),
            encoding="utf-8",
        )
        os.replace(temporary, self.manifest_path)

    @classmethod
    def load(cls, path: str | Path) -> "TrainingRun":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(RunManifest(**payload), root=Path(path).parent)

    # -- transitions -------------------------------------------------------
    def transition(self, new_state: str, *, reason: str | None = None) -> None:
        if new_state not in STATES:
            raise LifecycleError(f"unknown state {new_state!r}")
        current = self.manifest.status
        if new_state not in ALLOWED_TRANSITIONS[current]:
            raise LifecycleError(
                f"illegal transition {current} -> {new_state} for run "
                f"{self.manifest.run_id}"
            )
        self.manifest.history.append({
            "at": _now(), "from": current, "to": new_state, "reason": reason,
        })
        self.manifest.status = new_state
        if reason and new_state in (FAILED, QUARANTINED):
            self.manifest.failure_reason = reason
        self.save()

    # -- controls ----------------------------------------------------------
    def validate(self, checks: Mapping[str, bool], *, details: Mapping[str, str] | None = None) -> None:
        """Run preflight. Any failure leaves the run unvalidated."""
        self.transition(VALIDATING)
        failures = [name for name, ok in checks.items() if not ok]
        if failures:
            detail = "; ".join(
                f"{name}: {(details or {}).get(name, 'failed')}" for name in failures
            )
            self.transition(FAILED, reason=f"preflight failed -- {detail}")
            raise PreflightError(failures, f"preflight failed: {detail}")
        self.transition(VALIDATED)

    def freeze(self) -> str:
        """Pin the configuration hash. Nothing may launch unfrozen."""
        self.manifest.configuration_hash = self.manifest.compute_configuration_hash()
        self.transition(FROZEN, reason=f"configuration_hash={self.manifest.configuration_hash}")
        return self.manifest.configuration_hash

    def arm(self, *, confirmed_hash: str) -> None:
        """Arm the run, requiring the operator to echo the frozen hash back.

        Requiring the hash means an operator cannot arm a configuration they
        have not actually seen -- the UI's confirmation step becomes evidence
        rather than a click-through.
        """
        if self.manifest.status != FROZEN:
            raise LifecycleError(
                f"cannot arm from {self.manifest.status}; freeze the configuration first"
            )
        if confirmed_hash != self.manifest.configuration_hash:
            raise LifecycleError(
                f"armed hash {confirmed_hash!r} does not match the frozen "
                f"{self.manifest.configuration_hash!r}; the configuration changed "
                "since it was reviewed"
            )
        self.transition(ARMED)

    def launch(self, *, pid: int, host: str, gpu: str | None = None) -> None:
        if self.manifest.status != ARMED:
            raise LifecycleError(
                f"cannot launch from {self.manifest.status}; a run must be ARMED "
                "so an unreviewed configuration cannot reach the GPU queue"
            )
        self.transition(QUEUED)
        self.transition(STARTING)
        self.manifest.pid = pid
        self.manifest.host = host
        self.manifest.gpu = gpu
        self.manifest.started_at = _now()
        self.beat()
        self.transition(RUNNING)

    def beat(self) -> None:
        self.manifest.heartbeat = _now()
        self.save()

    def checkpoint(self, path: str | Path, *, epoch: int,
                   metric: float | None = None) -> None:
        self.transition(CHECKPOINTING)
        self.manifest.checkpoint_path = str(path)
        self.manifest.latest_epoch = epoch
        if metric is not None and (
            self.manifest.best_metric is None or metric < self.manifest.best_metric
        ):
            self.manifest.best_metric = metric
            self.manifest.best_epoch = epoch
        self.beat()
        self.transition(RUNNING)

    def pause(self) -> None:
        self.transition(PAUSING, reason="pause requested at checkpoint boundary")
        self.transition(PAUSED)

    def resume(self) -> None:
        if self.manifest.checkpoint_path is None:
            raise LifecycleError("cannot resume a run with no checkpoint")
        self.transition(RESUMING)
        self.transition(RUNNING)

    def cancel(self, reason: str = "operator cancelled") -> None:
        self.transition(CANCELLED, reason=reason)

    def complete(self) -> None:
        self.transition(COMPLETED)

    def fail(self, reason: str) -> None:
        self.transition(FAILED, reason=reason)

    def quarantine(self, reason: str) -> None:
        self.transition(QUARANTINED, reason=reason)

    def clone(self, *, run_id: str) -> "TrainingRun":
        """Copy the configuration into a fresh DRAFT run.

        Deliberately does not copy results, PID, heartbeat or checkpoint: a
        clone that inherited its parent's outputs would report another run's
        numbers as its own.
        """
        payload = self.manifest.to_dict()
        for key in ("pid", "heartbeat", "started_at", "checkpoint_path",
                    "best_metric", "best_epoch", "failure_reason"):
            payload[key] = None
        payload.update({
            "run_id": run_id, "status": DRAFT, "latest_epoch": 0,
            "history": [], "configuration_hash": "", "updated_at": _now(),
        })
        clone = TrainingRun(RunManifest(**payload), root=self.root)
        clone.save()
        return clone


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def preflight_checks(
    *,
    dataset_path: str | Path,
    expected_dataset_hash: str | None,
    actual_dataset_hash: str | None,
    expected_schema_hash: str | None,
    actual_schema_hash: str | None,
    folds: Sequence[Mapping[str, Any]],
    train_rows: int,
    validation_rows: int,
    output_dir: str | Path,
    min_free_bytes: int = 5 << 30,
    require_gpu: bool = False,
    gpu_available: bool = False,
) -> tuple[dict[str, bool], dict[str, str]]:
    """Everything that must hold before a run may be armed."""
    dataset = Path(dataset_path)
    free = shutil.disk_usage(Path(output_dir).parent
                             if not Path(output_dir).exists() else output_dir).free

    checks = {
        "dataset_exists": dataset.exists(),
        "dataset_hash_matches": (
            expected_dataset_hash is None
            or expected_dataset_hash == actual_dataset_hash
        ),
        "schema_hash_matches": (
            expected_schema_hash is None
            or expected_schema_hash == actual_schema_hash
        ),
        "folds_defined": len(folds) > 0,
        "no_empty_train_fold": train_rows > 0,
        "no_empty_validation_fold": validation_rows > 0,
        "sufficient_disk": free >= min_free_bytes,
        "gpu_available_if_required": (not require_gpu) or gpu_available,
    }
    details = {
        "dataset_exists": f"{dataset} not found",
        "dataset_hash_matches":
            f"expected {expected_dataset_hash}, dataset is {actual_dataset_hash} -- "
            "the dataset was rebuilt since this configuration was frozen",
        "schema_hash_matches":
            f"expected {expected_schema_hash}, dataset is {actual_schema_hash} -- schema drift",
        "folds_defined": "no walk-forward folds defined",
        "no_empty_train_fold": f"train fold has {train_rows} rows",
        "no_empty_validation_fold": f"validation fold has {validation_rows} rows",
        "sufficient_disk": f"{free / 2**30:.1f} GiB free, need {min_free_bytes / 2**30:.1f} GiB",
        "gpu_available_if_required": "GPU required but not available",
    }
    return checks, details


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def write_checkpoint_atomically(payload: Mapping[str, Any], path: str | Path) -> str:
    """Write a checkpoint plus a sidecar digest, atomically.

    Writing in place risks a truncated file if the process dies mid-write, and a
    truncated checkpoint that still loads is worse than one that fails loudly --
    hence the digest.
    """
    import pickle

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    data = pickle.dumps(dict(payload))
    temporary.write_bytes(data)
    with open(temporary, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    digest = hashlib.sha256(data).hexdigest()
    target.with_suffix(target.suffix + ".sha256").write_text(digest, encoding="utf-8")
    return digest


def load_checkpoint_verified(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint, refusing one whose digest does not match."""
    import pickle

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"checkpoint {target} does not exist")
    data = target.read_bytes()
    sidecar = target.with_suffix(target.suffix + ".sha256")
    if sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").strip()
        actual = hashlib.sha256(data).hexdigest()
        if expected != actual:
            raise ValueError(
                f"checkpoint {target} is corrupt: digest {actual[:16]} does not "
                f"match the recorded {expected[:16]}"
            )
    return pickle.loads(data)


def guard_loss(value: float, *, epoch: int) -> None:
    """Stop on a non-finite loss instead of training through it."""
    import math

    if value is None or not math.isfinite(value):
        raise PreflightError(
            ["non_finite_loss"],
            f"loss is {value!r} at epoch {epoch}; continuing would produce a "
            "checkpoint that cannot be evaluated",
        )
