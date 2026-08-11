"""Durable parent -> child execution planning for paper/research workflows.

This module separates a portfolio-level parent instruction from deterministic
child slices.  It deliberately stops at the planning/state boundary: it does
*not* submit venue orders and it does not certify production execution.

Safety properties
-----------------
* parent economics are immutable once persisted;
* snapshots are cross-process locked, atomically replaced, fsync'd and hashed;
* TWAP/VWAP schedules are timezone-explicit and deterministic;
* VWAP requires a finite positive forecast-volume profile (no TWAP fallback);
* POV requires monotonic cumulative market-volume evidence and exposes at most
  one active child at a time;
* iceberg exposes at most one active displayed child at a time;
* deterministic child ids survive restart and are suitable for downstream
  idempotency lineage, but this module never bypasses the existing OMS/RMS;
* PARTIAL children are not re-released as their original full quantity;
* terminal child states are immutable;
* fills + outstanding quantity can never exceed the parent quantity.

Static TWAP/VWAP plans do not silently redistribute a cancelled residual into
later slices.  Such catch-up is a separate, explicit policy decision because it
changes market-impact and benchmark economics.  TCA should report the resulting
underfill instead of hiding it.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from threading import RLock
from typing import Callable, Iterator, Mapping, Sequence

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None

try:
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - Unix
    _msvcrt = None


PARENT_EXECUTION_SCHEMA = "quantagent.execution.parent_child.v2"


class ParentExecutionError(RuntimeError):
    """Base parent/child execution contract failure."""


class ParentExecutionCorruption(ParentExecutionError):
    """Persisted parent execution state cannot be trusted."""


class ParentExecutionConflict(ParentExecutionError):
    """A caller attempted to reinterpret immutable parent economics."""


class ExecutionAlgorithm(str, Enum):
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    ICEBERG = "iceberg"


class ChildStatus(str, Enum):
    PLANNED = "planned"
    RELEASED = "released"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


_ACTIVE_CHILD_STATUSES = frozenset({ChildStatus.RELEASED, ChildStatus.PARTIAL})
_TERMINAL_CHILD_STATUSES = frozenset(
    {ChildStatus.FILLED, ChildStatus.CANCELLED, ChildStatus.REJECTED}
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_time(value: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ParentExecutionError("schedule timestamp must be non-empty")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ParentExecutionError(f"invalid schedule timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise ParentExecutionError(
            f"schedule timestamp {value!r} must include an explicit timezone"
        )
    return parsed


def _canonical_times(values: Sequence[str]) -> tuple[str, ...]:
    parsed = [_parse_time(value) for value in values]
    if not parsed:
        raise ParentExecutionError("TWAP/VWAP requires at least one schedule timestamp")
    if any(right <= left for left, right in zip(parsed, parsed[1:])):
        raise ParentExecutionError("schedule timestamps must be strictly increasing")
    return tuple(value.isoformat() for value in parsed)


def _normalise_profile(values: Sequence[float], n: int) -> tuple[float, ...]:
    if len(values) != n:
        raise ParentExecutionError(
            f"VWAP volume profile length {len(values)} != schedule length {n}"
        )
    floats = tuple(float(value) for value in values)
    if any((not math.isfinite(value)) or value < 0.0 for value in floats):
        raise ParentExecutionError(
            "VWAP volume profile must contain finite non-negative values"
        )
    total = sum(floats)
    if not math.isfinite(total) or total <= 0.0:
        raise ParentExecutionError("VWAP volume profile must have finite positive mass")
    return tuple(value / total for value in floats)


def _content_child_id(
    parent_id: str,
    sequence: int,
    quantity: int,
    scheduled_at: str,
) -> str:
    material = f"{parent_id}|{sequence}|{quantity}|{scheduled_at}".encode("utf-8")
    return f"child-{sha256(material).hexdigest()[:24]}"


def child_lineage_id(parent_id: str, child_id: str) -> str:
    """Return deterministic lineage text for a future guarded OMS adapter.

    This is intentionally metadata only.  A venue adapter must still pass the
    child through the repository's OMS/RMS/idempotency controls.
    """

    parent = str(parent_id).strip()
    child = str(child_id).strip()
    if not parent or not child:
        raise ParentExecutionError("parent_id and child_id must be non-empty")
    return f"parent:{parent}:child:{child}"


@dataclass(frozen=True, slots=True)
class ParentOrderSpec:
    parent_id: str
    symbol: str
    side: str
    total_quantity: int
    algorithm: ExecutionAlgorithm
    schedule_times: tuple[str, ...] = ()
    volume_profile: tuple[float, ...] = ()
    participation_rate: float | None = None
    display_quantity: int | None = None
    max_child_quantity: int | None = None
    lot_size: int = 100
    signal_id: str = ""
    strategy_version: str = ""

    def canonical(self) -> "ParentOrderSpec":
        parent_id = str(self.parent_id).strip()
        symbol = str(self.symbol).strip()
        side = str(self.side).strip().lower()
        if not parent_id or not symbol:
            raise ParentExecutionError("parent_id and symbol must be non-empty")
        if side not in {"buy", "sell"}:
            raise ParentExecutionError("parent side must be buy or sell")

        quantity = int(self.total_quantity)
        lot_size = int(self.lot_size)
        if quantity <= 0 or lot_size <= 0:
            raise ParentExecutionError("total_quantity and lot_size must be positive")
        if quantity % lot_size != 0:
            raise ParentExecutionError(
                "parent quantity must be lot-aligned; venue-specific odd-lot "
                "liquidation is outside this planner"
            )

        algo = ExecutionAlgorithm(self.algorithm)
        schedule: tuple[str, ...] = ()
        profile: tuple[float, ...] = ()
        participation: float | None = None
        display: int | None = None

        if algo in {ExecutionAlgorithm.TWAP, ExecutionAlgorithm.VWAP}:
            schedule = _canonical_times(self.schedule_times)
            if algo is ExecutionAlgorithm.VWAP:
                profile = _normalise_profile(self.volume_profile, len(schedule))
            elif self.volume_profile:
                raise ParentExecutionError("TWAP must not carry a VWAP volume profile")
        elif self.schedule_times or self.volume_profile:
            raise ParentExecutionError(
                f"{algo.value} is event-driven and must not carry a static schedule/profile"
            )

        if algo is ExecutionAlgorithm.POV:
            if self.participation_rate is None:
                raise ParentExecutionError("POV requires participation_rate")
            participation = float(self.participation_rate)
            if not math.isfinite(participation) or not 0.0 < participation <= 1.0:
                raise ParentExecutionError(
                    "POV participation_rate must be finite and in (0, 1]"
                )
        elif self.participation_rate is not None:
            raise ParentExecutionError(
                f"participation_rate is only valid for POV, not {algo.value}"
            )

        if algo is ExecutionAlgorithm.ICEBERG:
            if self.display_quantity is None:
                raise ParentExecutionError("iceberg requires display_quantity")
            display = int(self.display_quantity)
            if display <= 0 or display % lot_size != 0:
                raise ParentExecutionError(
                    "iceberg display_quantity must be positive and lot-aligned"
                )
        elif self.display_quantity is not None:
            raise ParentExecutionError(
                f"display_quantity is only valid for iceberg, not {algo.value}"
            )

        max_child = None
        if self.max_child_quantity is not None:
            max_child = int(self.max_child_quantity)
            if max_child <= 0 or max_child % lot_size != 0:
                raise ParentExecutionError(
                    "max_child_quantity must be positive and lot-aligned"
                )

        return ParentOrderSpec(
            parent_id=parent_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            algorithm=algo,
            schedule_times=schedule,
            volume_profile=profile,
            participation_rate=participation,
            display_quantity=display,
            max_child_quantity=max_child,
            lot_size=lot_size,
            signal_id=str(self.signal_id),
            strategy_version=str(self.strategy_version),
        )


@dataclass(frozen=True, slots=True)
class ChildExecution:
    child_id: str
    sequence: int
    quantity: int
    scheduled_at: str
    status: ChildStatus = ChildStatus.PLANNED
    filled_quantity: int = 0
    last_updated_at: str = ""

    @property
    def outstanding_quantity(self) -> int:
        if self.status in _TERMINAL_CHILD_STATUSES:
            return 0
        if self.status is ChildStatus.PLANNED:
            return 0
        return max(0, int(self.quantity) - int(self.filled_quantity))


@dataclass(frozen=True, slots=True)
class ParentExecutionState:
    schema_version: str
    parent: ParentOrderSpec
    children: tuple[ChildExecution, ...]
    last_observed_cumulative_volume: int
    revision: int
    created_at: str
    updated_at: str
    payload_sha256: str = ""

    @property
    def filled_quantity(self) -> int:
        return sum(int(child.filled_quantity) for child in self.children)

    @property
    def outstanding_quantity(self) -> int:
        return sum(child.outstanding_quantity for child in self.children)

    @property
    def remaining_quantity(self) -> int:
        return max(0, self.parent.total_quantity - self.filled_quantity)

    @property
    def committed_quantity(self) -> int:
        return self.filled_quantity + self.outstanding_quantity

    @property
    def complete(self) -> bool:
        return self.filled_quantity >= self.parent.total_quantity

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parent": {
                **asdict(self.parent),
                "algorithm": self.parent.algorithm.value,
                "schedule_times": list(self.parent.schedule_times),
                "volume_profile": list(self.parent.volume_profile),
            },
            "children": [
                {**asdict(child), "status": child.status.value}
                for child in self.children
            ],
            "last_observed_cumulative_volume": self.last_observed_cumulative_volume,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "payload_sha256": self.payload_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ParentExecutionState":
        parent_payload = dict(payload["parent"])  # type: ignore[arg-type]
        parent_payload["algorithm"] = ExecutionAlgorithm(str(parent_payload["algorithm"]))
        parent_payload["schedule_times"] = tuple(parent_payload.get("schedule_times") or ())
        parent_payload["volume_profile"] = tuple(parent_payload.get("volume_profile") or ())
        children = tuple(
            ChildExecution(
                child_id=str(row["child_id"]),
                sequence=int(row["sequence"]),
                quantity=int(row["quantity"]),
                scheduled_at=str(row["scheduled_at"]),
                status=ChildStatus(str(row["status"])),
                filled_quantity=int(row.get("filled_quantity", 0)),
                last_updated_at=str(row.get("last_updated_at") or ""),
            )
            for row in payload.get("children", ())  # type: ignore[union-attr]
        )
        return cls(
            schema_version=str(payload["schema_version"]),
            parent=ParentOrderSpec(**parent_payload).canonical(),
            children=children,
            last_observed_cumulative_volume=int(
                payload.get("last_observed_cumulative_volume", 0)
            ),
            revision=int(payload.get("revision", 0)),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            payload_sha256=str(payload.get("payload_sha256") or ""),
        )


def _payload_digest(payload: Mapping[str, object]) -> str:
    material = dict(payload)
    material.pop("payload_sha256", None)
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _with_digest(state: ParentExecutionState) -> ParentExecutionState:
    provisional = replace(state, payload_sha256="")
    return replace(provisional, payload_sha256=_payload_digest(provisional.to_dict()))


def _verify_state(state: ParentExecutionState) -> None:
    if state.schema_version != PARENT_EXECUTION_SCHEMA:
        raise ParentExecutionCorruption(
            f"unsupported parent execution schema {state.schema_version!r}"
        )
    if _payload_digest(state.to_dict()) != state.payload_sha256:
        raise ParentExecutionCorruption("parent execution state digest mismatch")
    if state.filled_quantity > state.parent.total_quantity:
        raise ParentExecutionCorruption("child fills exceed parent quantity")
    if state.committed_quantity > state.parent.total_quantity:
        raise ParentExecutionCorruption("filled + outstanding child quantity exceeds parent")
    if state.last_observed_cumulative_volume < 0:
        raise ParentExecutionCorruption("negative observed cumulative market volume")

    seen: set[str] = set()
    for expected_sequence, child in enumerate(state.children):
        if child.child_id in seen:
            raise ParentExecutionCorruption(f"duplicate child id {child.child_id}")
        seen.add(child.child_id)
        if child.sequence != expected_sequence:
            raise ParentExecutionCorruption("child sequence is non-contiguous")
        if child.quantity <= 0 or child.quantity % state.parent.lot_size != 0:
            raise ParentExecutionCorruption(
                "child quantity is not positive/parent-lot-aligned"
            )
        if child.filled_quantity < 0 or child.filled_quantity > child.quantity:
            raise ParentExecutionCorruption(
                "child filled quantity is outside [0, quantity]"
            )
        _parse_time(child.scheduled_at)


class ParentExecutionStore:
    """One durable parent-state snapshot with cross-process compare/update."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._thread_lock = RLock()

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+b") as handle:
            if _fcntl is not None:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)
                return
            if _msvcrt is not None:  # pragma: no cover - Windows
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                    os.fsync(handle.fileno())
                handle.seek(0)
                _msvcrt.locking(handle.fileno(), _msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    _msvcrt.locking(handle.fileno(), _msvcrt.LK_UNLCK, 1)
                return
            raise ParentExecutionError("no supported cross-process lock backend")

    def _read_unlocked(self) -> ParentExecutionState | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            state = ParentExecutionState.from_dict(payload)
            _verify_state(state)
            return state
        except ParentExecutionError:
            raise
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ParentExecutionCorruption(
                f"cannot parse parent execution state {self.path}: {exc}"
            ) from exc

    def read(self) -> ParentExecutionState | None:
        with self._thread_lock, self._exclusive_lock():
            return self._read_unlocked()

    def ensure(self, parent: ParentOrderSpec) -> ParentExecutionState:
        canonical = parent.canonical()
        with self._thread_lock, self._exclusive_lock():
            existing = self._read_unlocked()
            if existing is not None:
                if existing.parent != canonical:
                    raise ParentExecutionConflict(
                        "persisted parent economics differ from requested parent"
                    )
                return existing
            timestamp = _now()
            state = _with_digest(
                ParentExecutionState(
                    schema_version=PARENT_EXECUTION_SCHEMA,
                    parent=canonical,
                    children=(),
                    last_observed_cumulative_volume=0,
                    revision=0,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
            self._write_unlocked(state)
            return state

    def transact(
        self,
        parent: ParentOrderSpec,
        mutator: Callable[[ParentExecutionState], ParentExecutionState],
    ) -> ParentExecutionState:
        canonical = parent.canonical()
        with self._thread_lock, self._exclusive_lock():
            state = self._read_unlocked()
            if state is None:
                timestamp = _now()
                state = _with_digest(
                    ParentExecutionState(
                        schema_version=PARENT_EXECUTION_SCHEMA,
                        parent=canonical,
                        children=(),
                        last_observed_cumulative_volume=0,
                        revision=0,
                        created_at=timestamp,
                        updated_at=timestamp,
                    )
                )
            elif state.parent != canonical:
                raise ParentExecutionConflict(
                    "persisted parent economics differ from requested parent"
                )
            updated = mutator(state)
            if updated.parent != canonical:
                raise ParentExecutionConflict("mutator changed immutable parent economics")
            updated = _with_digest(
                replace(
                    updated,
                    schema_version=PARENT_EXECUTION_SCHEMA,
                    revision=state.revision + 1,
                    created_at=state.created_at,
                    updated_at=_now(),
                )
            )
            _verify_state(updated)
            self._write_unlocked(updated)
            return updated

    def _write_unlocked(self, state: ParentExecutionState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        encoded = json.dumps(
            state.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ) + "\n"
        try:
            with tmp.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
            if os.name == "posix":
                fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        finally:
            if tmp.exists():
                tmp.unlink()


def _allocate_lot_quantities(
    total_quantity: int,
    weights: Sequence[float],
    lot_size: int,
) -> list[int]:
    if not weights:
        raise ParentExecutionError("allocation requires at least one weight")
    total_lots = total_quantity // lot_size
    raw = [float(weight) * total_lots for weight in weights]
    base = [int(value) for value in raw]
    remainder = total_lots - sum(base)
    order = sorted(
        range(len(raw)),
        key=lambda index: (raw[index] - base[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        base[index] += 1
    return [lots * lot_size for lots in base]


def _static_children(parent: ParentOrderSpec) -> tuple[ChildExecution, ...]:
    if parent.algorithm is ExecutionAlgorithm.TWAP:
        weights = [1.0 / len(parent.schedule_times)] * len(parent.schedule_times)
    elif parent.algorithm is ExecutionAlgorithm.VWAP:
        weights = list(parent.volume_profile)
    else:
        raise ParentExecutionError(
            "static child planning is only valid for TWAP/VWAP"
        )
    quantities = _allocate_lot_quantities(
        parent.total_quantity,
        weights,
        parent.lot_size,
    )
    children: list[ChildExecution] = []
    for index, (quantity, scheduled_at) in enumerate(
        zip(quantities, parent.schedule_times)
    ):
        if quantity <= 0:
            continue
        children.append(
            ChildExecution(
                child_id=_content_child_id(
                    parent.parent_id, index, quantity, scheduled_at
                ),
                sequence=len(children),
                quantity=quantity,
                scheduled_at=scheduled_at,
                last_updated_at=_now(),
            )
        )
    if sum(child.quantity for child in children) != parent.total_quantity:
        raise ParentExecutionError(
            "static child allocation does not sum to parent quantity"
        )
    return tuple(children)


class ParentChildExecutionEngine:
    """Deterministic child-release state machine backed by durable state."""

    def __init__(self, parent: ParentOrderSpec, store: ParentExecutionStore) -> None:
        self.parent = parent.canonical()
        self.store = store
        self.store.ensure(self.parent)
        if self.parent.algorithm in {ExecutionAlgorithm.TWAP, ExecutionAlgorithm.VWAP}:
            self._ensure_static_plan()

    def state(self) -> ParentExecutionState:
        state = self.store.read()
        if state is None:
            raise ParentExecutionCorruption("parent execution state disappeared")
        if state.parent != self.parent:
            raise ParentExecutionConflict(
                "parent state belongs to different economics"
            )
        return state

    def released_children(self) -> tuple[ChildExecution, ...]:
        """Return full-size child intents eligible for idempotent downstream delivery.

        PARTIAL children are intentionally excluded because replaying the original
        quantity would over-submit their residual.  A broker/OMS reconciliation
        layer must handle a partial child explicitly.
        """

        return tuple(
            child
            for child in self.state().children
            if child.status is ChildStatus.RELEASED
        )

    def _ensure_static_plan(self) -> ParentExecutionState:
        def mutate(state: ParentExecutionState) -> ParentExecutionState:
            if state.children:
                return state
            return replace(state, children=_static_children(self.parent))

        return self.store.transact(self.parent, mutate)

    def release_due(
        self,
        *,
        now: str,
        cumulative_market_volume: int | None = None,
    ) -> tuple[ChildExecution, ...]:
        timestamp = _parse_time(now)

        def mutate(state: ParentExecutionState) -> ParentExecutionState:
            if state.complete:
                return state
            if self.parent.algorithm in {
                ExecutionAlgorithm.TWAP,
                ExecutionAlgorithm.VWAP,
            }:
                children = tuple(
                    replace(
                        child,
                        status=ChildStatus.RELEASED,
                        last_updated_at=timestamp.isoformat(),
                    )
                    if child.status is ChildStatus.PLANNED
                    and _parse_time(child.scheduled_at) <= timestamp
                    else child
                    for child in state.children
                )
                return replace(state, children=children)
            if self.parent.algorithm is ExecutionAlgorithm.POV:
                return self._release_pov(
                    state,
                    timestamp=timestamp,
                    cumulative_market_volume=cumulative_market_volume,
                )
            if self.parent.algorithm is ExecutionAlgorithm.ICEBERG:
                return self._release_iceberg(state, timestamp=timestamp)
            raise ParentExecutionError(
                f"unsupported algorithm {self.parent.algorithm}"
            )

        self.store.transact(self.parent, mutate)
        return self.released_children()

    def _release_pov(
        self,
        state: ParentExecutionState,
        *,
        timestamp: datetime,
        cumulative_market_volume: int | None,
    ) -> ParentExecutionState:
        if cumulative_market_volume is None:
            raise ParentExecutionError(
                "POV release requires cumulative_market_volume"
            )
        observed = int(cumulative_market_volume)
        if observed < 0:
            raise ParentExecutionError(
                "cumulative market volume must be non-negative"
            )
        if observed < state.last_observed_cumulative_volume:
            raise ParentExecutionError(
                "cumulative market volume moved backwards; feed reset/restart "
                "must be reconciled"
            )

        # Conservative POV: do not stack multiple live child orders.  The next
        # residual slice is released only after the active child reaches a
        # terminal state.  We still advance observed volume for restart safety.
        if any(child.status in _ACTIVE_CHILD_STATUSES for child in state.children):
            return replace(state, last_observed_cumulative_volume=observed)

        target = min(
            self.parent.total_quantity,
            int(observed * float(self.parent.participation_rate)),
        )
        target = (target // self.parent.lot_size) * self.parent.lot_size
        required = max(0, target - state.filled_quantity)
        if self.parent.max_child_quantity is not None:
            required = min(required, self.parent.max_child_quantity)
        required = (required // self.parent.lot_size) * self.parent.lot_size
        children = state.children
        if required > 0:
            sequence = len(children)
            scheduled_at = timestamp.isoformat()
            children = (
                *children,
                ChildExecution(
                    child_id=_content_child_id(
                        self.parent.parent_id,
                        sequence,
                        required,
                        scheduled_at,
                    ),
                    sequence=sequence,
                    quantity=required,
                    scheduled_at=scheduled_at,
                    status=ChildStatus.RELEASED,
                    last_updated_at=scheduled_at,
                ),
            )
        return replace(
            state,
            children=children,
            last_observed_cumulative_volume=observed,
        )

    def _release_iceberg(
        self,
        state: ParentExecutionState,
        *,
        timestamp: datetime,
    ) -> ParentExecutionState:
        if any(child.status in _ACTIVE_CHILD_STATUSES for child in state.children):
            return state
        remaining = state.remaining_quantity
        if remaining <= 0:
            return state
        quantity = min(remaining, int(self.parent.display_quantity or 0))
        if self.parent.max_child_quantity is not None:
            quantity = min(quantity, self.parent.max_child_quantity)
        quantity = (quantity // self.parent.lot_size) * self.parent.lot_size
        if quantity <= 0:
            raise ParentExecutionError(
                "iceberg remaining quantity is below executable lot"
            )
        sequence = len(state.children)
        scheduled_at = timestamp.isoformat()
        child = ChildExecution(
            child_id=_content_child_id(
                self.parent.parent_id,
                sequence,
                quantity,
                scheduled_at,
            ),
            sequence=sequence,
            quantity=quantity,
            scheduled_at=scheduled_at,
            status=ChildStatus.RELEASED,
            last_updated_at=scheduled_at,
        )
        return replace(state, children=(*state.children, child))

    def acknowledge(
        self,
        child_id: str,
        *,
        status: ChildStatus | str,
        filled_quantity: int,
    ) -> ParentExecutionState:
        new_status = ChildStatus(status)
        if new_status is ChildStatus.PLANNED:
            raise ParentExecutionError(
                "acknowledgement cannot move a child back to planned"
            )
        filled = int(filled_quantity)

        def mutate(state: ParentExecutionState) -> ParentExecutionState:
            children = list(state.children)
            matches = [
                index
                for index, child in enumerate(children)
                if child.child_id == child_id
            ]
            if len(matches) != 1:
                raise ParentExecutionError(
                    f"unknown/non-unique child id {child_id!r}"
                )
            index = matches[0]
            child = children[index]
            if child.status is ChildStatus.PLANNED:
                raise ParentExecutionError(
                    "PLANNED child cannot receive broker acknowledgement before release"
                )
            if filled < child.filled_quantity:
                raise ParentExecutionError(
                    "child cumulative filled quantity moved backwards"
                )
            if filled < 0 or filled > child.quantity:
                raise ParentExecutionError(
                    "child cumulative fill exceeds child quantity"
                )
            if new_status is ChildStatus.FILLED and filled != child.quantity:
                raise ParentExecutionError(
                    "FILLED child must have filled_quantity == quantity"
                )
            if new_status is ChildStatus.PARTIAL and not 0 < filled < child.quantity:
                raise ParentExecutionError(
                    "PARTIAL child requires 0 < filled_quantity < quantity"
                )
            if child.status in _TERMINAL_CHILD_STATUSES:
                if new_status is child.status and filled == child.filled_quantity:
                    return state
                raise ParentExecutionError(
                    f"terminal child state {child.status.value} is immutable"
                )

            children[index] = replace(
                child,
                status=new_status,
                filled_quantity=filled,
                last_updated_at=_now(),
            )
            updated = replace(state, children=tuple(children))
            if updated.filled_quantity > self.parent.total_quantity:
                raise ParentExecutionError(
                    "acknowledgement would overfill parent"
                )
            if updated.committed_quantity > self.parent.total_quantity:
                raise ParentExecutionError(
                    "acknowledgement would over-commit parent"
                )
            return updated

        return self.store.transact(self.parent, mutate)


__all__ = [
    "ChildExecution",
    "ChildStatus",
    "ExecutionAlgorithm",
    "PARENT_EXECUTION_SCHEMA",
    "ParentChildExecutionEngine",
    "ParentExecutionConflict",
    "ParentExecutionCorruption",
    "ParentExecutionError",
    "ParentExecutionState",
    "ParentExecutionStore",
    "ParentOrderSpec",
    "child_lineage_id",
]
