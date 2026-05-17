"""Holding-period tracker for the V7 target-weights builder.

Walk-forward folds, paper backtests and live trading all need to know
how long a name has been held: a position that's only one day old has
not yet earned its keep, and unwinding it immediately throws away the
information the alpha used to open it. The tracker exposes a small
state machine:

* ``begin_session(initial_weights)`` — seed the tracker with weights at
  ``t0`` (or empty for a cold start). Each seeded name's ``entry_date``
  becomes ``None`` until the first observed update sets it.
* ``record_session(date, weights, expected_horizons)`` — update the
  registry: new names get an ``entry_date``; names whose weight drops to
  zero leave the registry; names whose weight is non-zero increment
  their ``days_held``.
* ``snapshot()`` — return a ``DataFrame`` of the live registry.

The output is used by ``build_v7_target_weights`` to constrain
``|Δw|`` for under-aged names. State is persisted to a parquet file so
it survives walk-forward fold boundaries — without persistence, every
fold would look like fresh entries and the holding-period constraint
would never bind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass
class PositionRecord:
    symbol: str
    entry_date: pd.Timestamp | None
    last_seen: pd.Timestamp | None
    weight: float
    expected_horizon_days: int | None
    days_held: int = 0


class PositionAgeTracker:
    """Mutable per-symbol registry; not thread-safe."""

    def __init__(self, state_path: Path | None = None) -> None:
        self._records: dict[str, PositionRecord] = {}
        self._state_path = Path(state_path) if state_path is not None else None

    @classmethod
    def from_state(cls, state_path: Path) -> "PositionAgeTracker":
        tracker = cls(state_path=state_path)
        if state_path.exists():
            frame = pd.read_parquet(state_path)
            for _, row in frame.iterrows():
                symbol = str(row["symbol"])
                tracker._records[symbol] = PositionRecord(
                    symbol=symbol,
                    entry_date=pd.to_datetime(row["entry_date"]) if pd.notna(row.get("entry_date")) else None,
                    last_seen=pd.to_datetime(row["last_seen"]) if pd.notna(row.get("last_seen")) else None,
                    weight=float(row.get("weight", 0.0)),
                    expected_horizon_days=(
                        int(row["expected_horizon_days"]) if pd.notna(row.get("expected_horizon_days")) else None
                    ),
                    days_held=int(row.get("days_held", 0)),
                )
        return tracker

    def persist(self) -> Path | None:
        if self._state_path is None:
            return None
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot().to_parquet(self._state_path, index=False)
        return self._state_path

    def record_session(
        self,
        date: pd.Timestamp,
        weights: Mapping[str, float],
        expected_horizons: Mapping[str, int | None] | None = None,
    ) -> None:
        ts = pd.Timestamp(date)
        expected_horizons = dict(expected_horizons or {})
        seen = set()
        for symbol, weight in weights.items():
            weight_f = float(weight)
            seen.add(str(symbol))
            record = self._records.get(str(symbol))
            if record is None:
                if abs(weight_f) < 1e-9:
                    continue
                record = PositionRecord(
                    symbol=str(symbol),
                    entry_date=ts,
                    last_seen=ts,
                    weight=weight_f,
                    expected_horizon_days=expected_horizons.get(symbol),
                    days_held=0,
                )
                self._records[str(symbol)] = record
                continue
            if abs(weight_f) < 1e-9:
                # Position fully closed.
                self._records.pop(str(symbol), None)
                continue
            # Update existing — increment days_held only when calendar moves forward.
            if record.last_seen is None or ts > record.last_seen:
                record.days_held += 1
            record.last_seen = ts
            record.weight = weight_f
            if symbol in expected_horizons and expected_horizons[symbol] is not None:
                record.expected_horizon_days = int(expected_horizons[symbol])

        # Names not seen on this date keep their entry_date but get a stale flag via last_seen.
        for symbol in list(self._records.keys()):
            if symbol not in seen:
                self._records[symbol].weight = 0.0

    def snapshot(self) -> pd.DataFrame:
        rows = [
            {
                "symbol": rec.symbol,
                "entry_date": rec.entry_date,
                "last_seen": rec.last_seen,
                "weight": rec.weight,
                "expected_horizon_days": rec.expected_horizon_days,
                "days_held": rec.days_held,
            }
            for rec in self._records.values()
        ]
        return pd.DataFrame(rows)

    def age_for(self, symbol: str, today: pd.Timestamp) -> int:
        record = self._records.get(str(symbol))
        if record is None or record.entry_date is None:
            return 0
        delta = pd.Timestamp(today) - record.entry_date
        return max(int(delta.days), int(record.days_held))

    def is_locked(
        self,
        symbol: str,
        today: pd.Timestamp,
        force_close: bool = False,
    ) -> bool:
        if force_close:
            return False
        record = self._records.get(str(symbol))
        if record is None or record.expected_horizon_days is None:
            return False
        return self.age_for(symbol, today) < int(record.expected_horizon_days)

    def reset(self) -> None:
        self._records.clear()


__all__ = ["PositionRecord", "PositionAgeTracker"]
