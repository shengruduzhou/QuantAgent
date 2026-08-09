"""Durable pending-signal evidence for the daily paper/shadow loop.

A target produced from session-T close information is *not* an executed paper
trade.  Under the canonical A-share contract it may only be attempted on the
next observed market session.  This module records that pending economic intent
without inventing a future bar or mutating portfolio state.

The artifact is immutable-by-identity: re-running the same signal with identical
weights is idempotent; the same signal date with different economics is a hard
conflict.  A stored SHA-256 covers the canonical payload so an edited artifact is
refused before it can be consumed by a later execution stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS


PENDING_SIGNAL_SCHEMA_VERSION = "paper_pending_signal_v1"
PENDING_SIGNAL_STATUS = "pending_next_observed_session"


class PendingSignalConflict(RuntimeError):
    """The same signal-date identity was reused with different economics."""


class PendingSignalCorruption(RuntimeError):
    """A persisted pending artifact no longer matches its own digest."""


@dataclass(frozen=True)
class PendingPaperSignal:
    schema_version: str
    status: str
    signal_date: str
    execution_timing_semantics: str
    target_weights: dict[str, float]
    target_weights_sha256: str
    source_lineage: dict[str, str]
    created_at: str
    payload_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalised_weights(weights: pd.DataFrame, signal_date: str) -> dict[str, float]:
    """Return one deterministic signal-date target vector.

    The daily loop is intentionally one signal per run.  Ambiguous duplicate
    rows or multiple signal dates are refused rather than silently selecting one.
    """

    if weights is None or weights.empty:
        return {}
    frame = weights.copy()
    if "trade_date" in frame.columns:
        dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date.astype("string")
        valid_dates = sorted(set(str(value) for value in dates.dropna()))
        if valid_dates != [signal_date]:
            raise ValueError(
                "pending paper signal requires exactly the current signal date; "
                f"got {valid_dates!r}, expected {[signal_date]!r}"
            )
        frame = frame.drop(columns=["trade_date"])
    elif len(frame.index) == 1 and isinstance(frame.index, pd.DatetimeIndex):
        observed = pd.Timestamp(frame.index[0]).date().isoformat()
        if observed != signal_date:
            raise ValueError(
                f"target index signal date {observed} does not match {signal_date}"
            )
    if len(frame) != 1:
        raise ValueError(f"pending paper signal requires one target row; got {len(frame)}")

    row = frame.iloc[0]
    result: dict[str, float] = {}
    for symbol, value in sorted(row.items(), key=lambda item: str(item[0])):
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            continue
        number = float(numeric)
        if not np.isfinite(number):
            raise ValueError(f"non-finite target weight for {symbol}: {number}")
        if number < -1e-12:
            raise ValueError(
                f"cash-account pending target cannot contain negative stock weight: {symbol}={number}"
            )
        if abs(number) <= 1e-15:
            number = 0.0
        result[str(symbol)] = number
    return result


def _weights_sha(weights: Mapping[str, float]) -> str:
    payload = json.dumps(
        {str(key): round(float(value), 15) for key, value in sorted(weights.items())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _payload_digest(payload: Mapping[str, object]) -> str:
    material = dict(payload)
    material.pop("payload_sha256", None)
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def verify_pending_signal(signal: PendingPaperSignal) -> None:
    payload = signal.to_dict()
    if signal.schema_version != PENDING_SIGNAL_SCHEMA_VERSION:
        raise PendingSignalCorruption(
            f"unsupported pending-signal schema {signal.schema_version!r}"
        )
    if signal.status != PENDING_SIGNAL_STATUS:
        raise PendingSignalCorruption(f"unexpected pending status {signal.status!r}")
    if signal.execution_timing_semantics != EXECUTION_TIMING_SEMANTICS:
        raise PendingSignalCorruption(
            "pending signal timing semantics do not match strict execution contract"
        )
    if _weights_sha(signal.target_weights) != signal.target_weights_sha256:
        raise PendingSignalCorruption("pending target-weight digest mismatch")
    expected = _payload_digest(payload)
    if expected != signal.payload_sha256:
        raise PendingSignalCorruption("pending signal payload digest mismatch")


class PendingPaperSignalStore:
    """One immutable pending-signal artifact per market signal date."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, signal_date: str) -> Path:
        safe = pd.Timestamp(signal_date).date().isoformat()
        return self.root / f"{safe}.json"

    def read(self, signal_date: str) -> PendingPaperSignal | None:
        path = self.path_for(signal_date)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            signal = PendingPaperSignal(**payload)
        except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
            raise PendingSignalCorruption(
                f"cannot parse pending paper signal {path}: {exc}"
            ) from exc
        verify_pending_signal(signal)
        return signal

    def record(
        self,
        *,
        signal_date: str,
        target_weights: pd.DataFrame,
        source_lineage: Mapping[str, str],
        created_at: str | None = None,
    ) -> tuple[PendingPaperSignal, Path]:
        signal_date = pd.Timestamp(signal_date).date().isoformat()
        canonical_weights = _normalised_weights(target_weights, signal_date)
        weights_digest = _weights_sha(canonical_weights)
        lineage = {str(key): str(value) for key, value in sorted(source_lineage.items())}
        timestamp = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        base: dict[str, object] = {
            "schema_version": PENDING_SIGNAL_SCHEMA_VERSION,
            "status": PENDING_SIGNAL_STATUS,
            "signal_date": signal_date,
            "execution_timing_semantics": EXECUTION_TIMING_SEMANTICS,
            "target_weights": canonical_weights,
            "target_weights_sha256": weights_digest,
            "source_lineage": lineage,
            "created_at": timestamp,
            "payload_sha256": "",
        }
        base["payload_sha256"] = _payload_digest(base)
        candidate = PendingPaperSignal(**base)
        verify_pending_signal(candidate)

        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(signal_date)
        existing = self.read(signal_date)
        if existing is not None:
            # created_at is not economic identity. A deterministic re-run with
            # the same target and lineage returns the original artifact rather
            # than rewriting history; different economics/lineage is a conflict.
            if (
                existing.target_weights_sha256 == candidate.target_weights_sha256
                and existing.source_lineage == candidate.source_lineage
                and existing.execution_timing_semantics == candidate.execution_timing_semantics
            ):
                return existing, path
            raise PendingSignalConflict(
                f"pending signal already exists for {signal_date} with different economics or lineage"
            )

        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        data = json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        with temp.open("x", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        return candidate, path


__all__ = [
    "PENDING_SIGNAL_SCHEMA_VERSION",
    "PENDING_SIGNAL_STATUS",
    "PendingSignalConflict",
    "PendingSignalCorruption",
    "PendingPaperSignal",
    "PendingPaperSignalStore",
    "verify_pending_signal",
]
