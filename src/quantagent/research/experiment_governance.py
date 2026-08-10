"""Deterministic experiment identity, cumulative trial accounting and one-shot holdout seals.

Statistical tests are only as honest as the research process that feeds them. This module
gives each research recipe a stable identity, records every attempt together with its
pre-declared search multiplicity, and makes final-holdout access one-shot.

``fingerprint`` identifies the economic/statistical experiment and excludes wall-clock
time/status. ``event_hash`` identifies one immutable ledger event and includes those
event fields. Re-running the same recipe keeps the same fingerprint but consumes
additional multiple-testing budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, TypeVar


_JSON_SCALARS = (str, int, float, bool, type(None))
_T = TypeVar("_T")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation or fail closed."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not allowed in experiment identity")
        return value
    if isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        canonical_items = [_canonical(item) for item in value]
        return sorted(
            canonical_items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
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


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class ExperimentSpec:
    """Immutable identity of one research recipe on one declared data/split contract.

    ``declared_trial_count`` is the conservative number of configurations consumed to
    arrive at this final-holdout attempt: factor recipes, interaction candidates, model
    classes, parameter variants and any other outcome-conditioned search. It is part of
    the fingerprint, so quietly changing the search breadth changes experiment identity.
    """

    family: str
    candidate_id: str
    parameters: Mapping[str, Any]
    dataset_hash: str
    train_window: tuple[str, str]
    search_window: tuple[str, str]
    metric: str
    git_hash: str
    declared_trial_count: int
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
        if int(self.declared_trial_count) < 1:
            raise ValueError("declared_trial_count must be >= 1")
        _canonical(self.parameters)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": "quantagent.experiment.v2",
            "family": self.family,
            "candidate_id": self.candidate_id,
            "parameters": _canonical(self.parameters),
            "dataset_hash": self.dataset_hash,
            "train_window": list(self.train_window),
            "search_window": list(self.search_window),
            "metric": self.metric,
            "git_hash": self.git_hash,
            "declared_trial_count": int(self.declared_trial_count),
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
    """Append-only attempt ledger for lineage and conservative selection correction."""

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
        return len(self.read(family=family))

    def multiple_testing_trial_count(self, family: str | None = None) -> int:
        """Sum pre-declared search multiplicity over attempts; never use unique recipes."""
        total = 0
        for row in self.read(family=family):
            value = int(row.get("declared_trial_count", 0))
            if value < 1:
                raise ValueError("experiment ledger contains invalid declared_trial_count")
            total += value
        return total

    def unique_fingerprint_count(self, family: str | None = None) -> int:
        return len(
            {
                str(row.get("fingerprint", ""))
                for row in self.read(family=family)
                if row.get("fingerprint")
            }
        )

    def verify(self) -> None:
        """Detect accidental/manual mutation of event or identity fields."""
        for index, row in enumerate(self.read(), start=1):
            expected_event = row.get("event_hash")
            event_payload = dict(row)
            event_payload.pop("event_hash", None)
            if not expected_event or _digest(event_payload) != expected_event:
                raise ValueError(f"experiment ledger integrity failure at line {index}")

            identity_keys = (
                "schema",
                "family",
                "candidate_id",
                "parameters",
                "dataset_hash",
                "train_window",
                "search_window",
                "metric",
                "git_hash",
                "declared_trial_count",
                "recipe_hash",
                "split_id",
                "parent_fingerprint",
            )
            identity = {key: row.get(key) for key in identity_keys}
            expected_fingerprint = row.get("fingerprint")
            if not expected_fingerprint or _digest(identity) != expected_fingerprint:
                raise ValueError(f"experiment fingerprint mismatch at line {index}")


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

    A failed/crashed run still burns the holdout: once final data may have been observed,
    it must never silently become reusable tuning data.
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
        if not _is_sha256(candidate_fingerprint):
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
            existing_candidate = "unknown"
            try:
                existing_candidate = str(
                    json.loads(target.read_text(encoding="utf-8")).get(
                        "candidate_fingerprint", "unknown"
                    )
                )
            except (OSError, json.JSONDecodeError):
                pass
            raise RuntimeError(
                "final holdout has already been consumed; one-shot policy blocks reuse "
                f"(candidate={existing_candidate})"
            ) from exc
        try:
            os.write(
                fd,
                (
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8"),
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
