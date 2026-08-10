"""Deterministic experiment identity, cumulative trial accounting and one-shot holdout seals.

This module is intentionally small and policy-focused.  Statistical selection remains in
``selection_governance`` and nonlinear model comparison remains in ``model_comparison``;
this layer makes the *research process* auditable so those statistics are not fed an
optimistically small trial count or a repeatedly inspected final holdout.

A trial has two identities:

* ``fingerprint`` identifies the economic/statistical experiment and is stable across
  reruns.  Wall-clock time and mutable status are deliberately excluded.
* ``event_hash`` identifies one ledger event and therefore includes timestamp/status.

The distinction matters for multiple-testing governance: rerunning an old recipe is still
another attempt and must increase the conservative DSR/PBO trial count, while provenance
must also be able to tell that the recipe itself did not change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, TypeVar


_JSON_SCALARS = (str, int, float, bool, type(None))
_T = TypeVar("_T")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> Any:
    """Return a JSON-stable representation without relying on ``default=str``.

    Silent ``str(object)`` fallbacks are unsafe for experiment identity because many
    Python objects include memory addresses or unstable reprs.  Unknown objects are
    rejected so a fingerprint cannot look deterministic when it is not.
    """

    if isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        canonical_items = [_canonical(item) for item in value]
        return sorted(canonical_items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if hasattr(value, "item") and callable(value.item):
        return _canonical(value.item())
    raise TypeError(f"unsupported non-deterministic fingerprint value: {type(value).__name__}")


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExperimentSpec:
    """Immutable identity of one research recipe on one declared data/split contract."""

    family: str
    candidate_id: str
    parameters: Mapping[str, Any]
    dataset_hash: str
    train_window: tuple[str, str]
    search_window: tuple[str, str]
    metric: str
    git_hash: str
    recipe_hash: str = ""
    split_id: str = ""
    parent_fingerprint: str = ""

    def __post_init__(self) -> None:
        required = {
            "family": self.family,
            "candidate_id": self.candidate_id,
            "dataset_hash": self.dataset_hash,
            "metric": self.metric,
            "git_hash": self.git_hash,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"experiment identity fields cannot be empty: {missing}")
        if len(self.train_window) != 2 or len(self.search_window) != 2:
            raise ValueError("train_window and search_window must be (start, end)")
        # Force canonicalisation at construction time so bad parameters fail before a run.
        _canonical(self.parameters)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": "quantagent.experiment.v1",
            "family": self.family,
            "candidate_id": self.candidate_id,
            "parameters": _canonical(self.parameters),
            "dataset_hash": self.dataset_hash,
            "train_window": list(self.train_window),
            "search_window": list(self.search_window),
            "metric": self.metric,
            "git_hash": self.git_hash,
            "recipe_hash": self.recipe_hash,
            "split_id": self.split_id,
            "parent_fingerprint": self.parent_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return _digest(self.identity_payload())


@dataclass(frozen=True)
class ExperimentEvent:
    spec: ExperimentSpec
    status: str = "registered"
    created_at: str = field(default_factory=_utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        if not str(self.status).strip():
            raise ValueError("experiment event status cannot be empty")
        payload = {
            **self.spec.identity_payload(),
            "fingerprint": self.spec.fingerprint,
            "status": self.status,
            "created_at": self.created_at,
            "metadata": _canonical(self.metadata),
        }
        payload["event_hash"] = _digest(payload)
        return payload


class ExperimentLedger:
    """Append-only attempt ledger used for conservative multiple-testing counts."""

    def __init__(self, path: str | Path = "runtime/state/experiment_trials_v2.jsonl") -> None:
        self.path = Path(path)

    def append(self, event: ExperimentEvent) -> dict[str, Any]:
        payload = event.to_dict()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        return payload

    def read(self, family: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt experiment ledger line {line_number}") from exc
            if family is None or row.get("family") == family:
                rows.append(row)
        return rows

    def attempt_count(self, family: str | None = None) -> int:
        """Count attempts, not unique recipes; conservative for DSR/multiple testing."""
        return len(self.read(family=family))

    def unique_fingerprint_count(self, family: str | None = None) -> int:
        return len({str(row.get("fingerprint", "")) for row in self.read(family=family) if row.get("fingerprint")})

    def verify(self) -> None:
        """Fail if any stored event has been edited without recomputing its digest."""
        for index, row in enumerate(self.read(), start=1):
            expected = row.get("event_hash")
            payload = dict(row)
            payload.pop("event_hash", None)
            if not expected or _digest(payload) != expected:
                raise ValueError(f"experiment ledger integrity failure at line {index}")


@dataclass(frozen=True)
class FinalHoldoutSpec:
    """Identity of a final holdout independent of whichever candidate consumes it."""

    family: str
    dataset_hash: str
    holdout_window: tuple[str, str]
    label_contract_hash: str = ""

    def __post_init__(self) -> None:
        if not self.family.strip() or not self.dataset_hash.strip():
            raise ValueError("holdout family and dataset_hash are required")
        if len(self.holdout_window) != 2:
            raise ValueError("holdout_window must be (start, end)")

    @property
    def holdout_key(self) -> str:
        return _digest(
            {
                "schema": "quantagent.final_holdout.v1",
                "family": self.family,
                "dataset_hash": self.dataset_hash,
                "holdout_window": list(self.holdout_window),
                "label_contract_hash": self.label_contract_hash,
            }
        )


class FinalHoldoutLedger:
    """Atomic one-shot final-holdout consumer.

    Each holdout identity maps to one immutable seal file. ``O_EXCL`` means two jobs
    racing for the same final holdout cannot both succeed.  A failed/crashed run still
    burns the holdout, which is intentionally conservative: once final data may have
    been observed it must not silently become reusable tuning data.
    """

    def __init__(self, directory: str | Path = "runtime/state/final_holdout_seals") -> None:
        self.directory = Path(directory)

    def consume(
        self,
        spec: FinalHoldoutSpec,
        *,
        candidate_fingerprint: str,
        git_hash: str,
        run_id: str = "",
    ) -> dict[str, Any]:
        if len(candidate_fingerprint) != 64:
            raise ValueError("candidate_fingerprint must be a SHA-256 hex digest")
        if not git_hash.strip():
            raise ValueError("git_hash is required for final holdout consumption")
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{spec.holdout_key}.json"
        payload = {
            "schema": "quantagent.final_holdout.seal.v1",
            "holdout_key": spec.holdout_key,
            "family": spec.family,
            "dataset_hash": spec.dataset_hash,
            "holdout_window": list(spec.holdout_window),
            "label_contract_hash": spec.label_contract_hash,
            "candidate_fingerprint": candidate_fingerprint,
            "git_hash": git_hash,
            "run_id": run_id,
            "consumed_at": _utc_now(),
        }
        payload["seal_hash"] = _digest(payload)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(target, flags, 0o444)
        except FileExistsError as exc:
            existing = json.loads(target.read_text(encoding="utf-8"))
            raise RuntimeError(
                "final holdout has already been consumed; one-shot policy blocks reuse "
                f"(candidate={existing.get('candidate_fingerprint', 'unknown')})"
            ) from exc
        try:
            os.write(
                fd,
                (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
            )
        finally:
            os.close(fd)
        return payload


def with_cumulative_trial_count(report: _T, cumulative_trials: int) -> _T:
    """Raise a dataclass report's ``n_trials`` without ever reducing it."""
    if cumulative_trials < 0:
        raise ValueError("cumulative_trials must be non-negative")
    current = int(getattr(report, "n_trials"))
    return replace(report, n_trials=max(current, int(cumulative_trials)))


__all__ = [
    "ExperimentEvent",
    "ExperimentLedger",
    "ExperimentSpec",
    "FinalHoldoutLedger",
    "FinalHoldoutSpec",
    "with_cumulative_trial_count",
]
