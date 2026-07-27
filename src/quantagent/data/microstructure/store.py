"""Immutable raw event journal for A-share microstructure data.

The journal is the record of what a feed actually delivered. Everything
downstream -- normalisation, features, replay, backtests -- is reproducible
*from* it, which only holds if the journal itself is append-only and refuses
manufactured content.

Layout::

    <root>/provider=<provider>/family=<event_family>/exchange=<EX>/
        trade_date=<YYYY-MM-DD>/symbol=<canonical>/part-<seq>-<hash>.parquet

Hive-style partitioning is deliberate: it lets DuckDB/Polars/pyarrow prune to a
single symbol-day without a catalogue service, and it makes a missing partition
a visible directory absence rather than a silent empty scan.

Three invariants are enforced at write time, all fail-closed:

1. **Declared semantics.** ``data_class`` must be one of
   :data:`~quantagent.data.microstructure.contracts.DATA_CLASSES`, and must not
   be in ``NON_AUTHORITATIVE_CLASSES`` -- generated ticks and bar-derived ticks
   belong in a research sandbox, not in the journal.
2. **Schema completeness.** Every column the family's contract declares must be
   present. Missing columns are an error, not a null-fill.
3. **Immutability.** A part file is written once. Re-writing the same content
   is a no-op; writing *different* content to an existing partition raises
   unless the caller explicitly supersedes it, which records a tombstone rather
   than deleting history.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd

from quantagent.data.microstructure import contracts

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.=-]+$")


class ImmutableStoreError(RuntimeError):
    """Raised when a write would violate the journal's invariants."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe(token: str, *, what: str) -> str:
    token = str(token)
    if not token or not _SAFE_TOKEN.match(token):
        raise ImmutableStoreError(f"unsafe {what} partition token: {token!r}")
    return token


def frame_hash(frame: pd.DataFrame) -> str:
    """Content hash of a frame, stable across row order and column order."""
    if frame.empty:
        return hashlib.sha256(b"empty").hexdigest()[:16]
    ordered = frame.reindex(sorted(frame.columns), axis=1)
    ordered = ordered.sort_values(list(ordered.columns), kind="mergesort")
    payload = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class PartitionKey:
    provider: str
    family: str
    exchange: str
    trade_date: str
    symbol: str

    def relative_path(self) -> Path:
        return Path(
            f"provider={_safe(self.provider, what='provider')}",
            f"family={_safe(self.family, what='family')}",
            f"exchange={_safe(self.exchange, what='exchange')}",
            f"trade_date={_safe(self.trade_date, what='trade_date')}",
            f"symbol={_safe(self.symbol, what='symbol')}",
        )


@dataclass
class WriteReceipt:
    """What a single append actually did. Persisted next to the part file."""

    partition: str
    part_file: str
    rows: int
    content_hash: str
    data_class: str
    family: str
    ingest_sequence_start: int
    ingest_sequence_end: int
    written_at: str
    status: str  # WRITTEN | DEDUPLICATED | SUPERSEDED
    superseded_part: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RawEventStore:
    """Append-only partitioned journal of canonical microstructure events."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    # -- writing ----------------------------------------------------------
    def append(
        self,
        frame: pd.DataFrame,
        *,
        provider: str,
        family: str,
        data_class: str,
        supersede: bool = False,
    ) -> list[WriteReceipt]:
        """Validate and append ``frame``, one part file per symbol-day.

        Returns one receipt per partition touched. Raises
        :class:`ImmutableStoreError` before writing anything if the frame
        violates a journal invariant -- validation is whole-frame and happens
        first, so a bad batch never lands half-written.
        """
        contract = contracts.contract_for(family)
        self._validate(frame, contract=contract, data_class=data_class)
        if frame.empty:
            return []

        receipts: list[WriteReceipt] = []
        stamped = frame.copy()
        stamped["source_provider"] = provider
        stamped["data_class"] = data_class

        for (exchange, trade_date, symbol), part in stamped.groupby(
            ["exchange", "trade_date", "symbol"], sort=True
        ):
            key = PartitionKey(
                provider=provider,
                family=family,
                exchange=str(exchange),
                trade_date=str(trade_date),
                symbol=str(symbol),
            )
            receipts.append(
                self._write_partition(part, key=key, data_class=data_class, supersede=supersede)
            )
        return receipts

    def _validate(
        self, frame: pd.DataFrame, *, contract: contracts.EventContract, data_class: str
    ) -> None:
        if data_class not in contracts.DATA_CLASSES:
            raise ImmutableStoreError(
                f"data_class {data_class!r} is not a declared class; "
                f"known: {list(contracts.DATA_CLASSES)}"
            )
        if data_class in contracts.NON_AUTHORITATIVE_CLASSES:
            raise ImmutableStoreError(
                f"refusing to journal non-authoritative data_class {data_class!r}; "
                "generated, bar-derived and replayed events are research inputs, "
                "not a record of what the market did"
            )
        if frame.empty:
            return

        missing = [c for c in contract.columns if c not in frame.columns]
        if missing:
            raise ImmutableStoreError(
                f"frame is missing {contract.family} contract columns: {missing}"
            )
        if "ingest_sequence" in frame.columns and frame["ingest_sequence"].isna().any():
            raise ImmutableStoreError("ingest_sequence must be populated on every row")

        # A published side without a stated method is exactly the fabrication
        # this module exists to prevent.
        if "side" in frame.columns and "side_method" in frame.columns:
            claimed = frame["side"].notna() & (
                frame["side_method"].isna()
                | ~frame["side_method"].isin(contracts.SIDE_METHODS)
            )
            if bool(claimed.any()):
                raise ImmutableStoreError(
                    f"{int(claimed.sum())} rows carry a trade side with no valid "
                    "side_method; an inferred direction must name its rule"
                )

    def _write_partition(
        self,
        part: pd.DataFrame,
        *,
        key: PartitionKey,
        data_class: str,
        supersede: bool,
    ) -> WriteReceipt:
        directory = self.root / key.relative_path()
        directory.mkdir(parents=True, exist_ok=True)
        digest = frame_hash(part)
        existing = sorted(directory.glob("part-*.parquet"))

        for candidate in existing:
            if candidate.stem.endswith(digest):
                return WriteReceipt(
                    partition=str(key.relative_path()),
                    part_file=str(candidate.relative_to(self.root)),
                    rows=len(part),
                    content_hash=digest,
                    data_class=data_class,
                    family=key.family,
                    ingest_sequence_start=int(part["ingest_sequence"].min()),
                    ingest_sequence_end=int(part["ingest_sequence"].max()),
                    written_at=_utc_now(),
                    status="DEDUPLICATED",
                )

        superseded: str | None = None
        if existing and not supersede:
            raise ImmutableStoreError(
                f"partition {key.relative_path()} already holds "
                f"{len(existing)} part file(s) with different content; pass "
                "supersede=True to record a tombstone instead of silently "
                "rewriting history"
            )
        if existing and supersede:
            superseded = str(existing[-1].relative_to(self.root))
            tombstone = directory / f"{existing[-1].stem}.superseded.json"
            tombstone.write_text(
                json.dumps(
                    {
                        "superseded_at": _utc_now(),
                        "superseded_part": superseded,
                        "replacement_hash": digest,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        part_file = directory / f"part-{len(existing):04d}-{digest}.parquet"
        part.to_parquet(part_file, index=False)
        receipt = WriteReceipt(
            partition=str(key.relative_path()),
            part_file=str(part_file.relative_to(self.root)),
            rows=len(part),
            content_hash=digest,
            data_class=data_class,
            family=key.family,
            ingest_sequence_start=int(part["ingest_sequence"].min()),
            ingest_sequence_end=int(part["ingest_sequence"].max()),
            written_at=_utc_now(),
            status="SUPERSEDED" if superseded else "WRITTEN",
            superseded_part=superseded,
        )
        (directory / f"{part_file.stem}.receipt.json").write_text(
            json.dumps(receipt.to_dict(), indent=2), encoding="utf-8"
        )
        return receipt

    # -- reading ----------------------------------------------------------
    def partitions(
        self,
        *,
        provider: str | None = None,
        family: str | None = None,
        trade_date: str | None = None,
        symbol: str | None = None,
    ) -> Iterator[Path]:
        """Yield partition directories matching the given filters."""
        pattern = Path(
            f"provider={provider or '*'}",
            f"family={family or '*'}",
            "exchange=*",
            f"trade_date={trade_date or '*'}",
            f"symbol={symbol or '*'}",
        )
        yield from sorted(self.root.glob(str(pattern)))

    def read(
        self,
        *,
        provider: str | None = None,
        family: str | None = None,
        trade_date: str | None = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """Read matching partitions back into one frame, ordered by ingest."""
        frames: list[pd.DataFrame] = []
        for directory in self.partitions(
            provider=provider, family=family, trade_date=trade_date, symbol=symbol
        ):
            for part_file in sorted(directory.glob("part-*.parquet")):
                if (directory / f"{part_file.stem}.superseded.json").exists():
                    continue
                frames.append(pd.read_parquet(part_file))
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if "ingest_sequence" in combined.columns:
            combined = combined.sort_values("ingest_sequence", kind="mergesort")
        return combined.reset_index(drop=True)

    def inventory(self) -> pd.DataFrame:
        """One row per stored partition: provider, family, symbol-day, rows."""
        records: list[dict[str, Any]] = []
        for receipt_file in sorted(self.root.rglob("part-*.receipt.json")):
            payload = json.loads(receipt_file.read_text(encoding="utf-8"))
            tokens = dict(
                piece.split("=", 1)
                for piece in Path(payload["partition"]).parts
                if "=" in piece
            )
            records.append({**tokens, **payload})
        return pd.DataFrame(records)


def assign_ingest_sequence(
    frame: pd.DataFrame, *, start: int = 0, order_by: Sequence[str] | None = None
) -> pd.DataFrame:
    """Attach the storage-ordering counter this repository owns.

    This is explicitly **not** an exchange sequence number. It records the order
    in which we received and journalled events so a replay is deterministic; it
    carries no claim about the order in which the exchange matched them. When
    the vendor does publish its own sequence, keep it in ``sequence`` and leave
    the two independent.
    """
    result = frame.copy()
    if order_by:
        result = result.sort_values(list(order_by), kind="mergesort")
    result["ingest_sequence"] = range(start, start + len(result))
    return result.reset_index(drop=True)
