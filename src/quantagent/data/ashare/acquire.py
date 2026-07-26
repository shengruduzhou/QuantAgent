"""Resumable, provenance-preserving full-universe acquisition.

The acquisition worker is deliberately boring: one symbol at a time, written to
its own partition file, with a durable ledger row per attempt. That makes the
job resumable after any interruption, bounded in memory regardless of universe
size, and auditable — every partition records which provider answered and what
the other providers said.

Design rules enforced here:

* a provider is tried in the configured order and the FIRST provider that
  returns rows wins the symbol; the losing providers' retry classes are kept in
  the ledger so "we never asked" is distinguishable from "it refused";
* a symbol is never stitched together from two providers inside one partition —
  mixing is only possible at the panel level and then only with an explicit
  :class:`~quantagent.data.ashare.contracts.SourceBoundary` record;
* a permanent/entitlement failure is not retried on resume, a transient one is;
* a cancel file makes a long job stoppable from the workstation without losing
  completed partitions.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from quantagent.data.ashare.http import (
    RETRY_EMPTY,
    RETRY_ENTITLEMENT,
    RETRY_OK,
    RETRY_PERMANENT,
)
from quantagent.data.ashare.symbols import identify

LEDGER_COLUMNS = (
    "symbol", "board", "provider", "status", "retry_class", "rows",
    "first_date", "last_date", "attempts", "detail", "recorded_at",
)

#: Retry classes that will not be re-attempted on a later resume.
TERMINAL_CLASSES = frozenset({RETRY_PERMANENT, RETRY_ENTITLEMENT})


@dataclass
class ProviderSpec:
    """One provider in the fallback chain."""

    name: str
    fetch: Callable[[str], Any]      # symbol -> SourceResult
    min_interval_s: float = 0.0      # vendor pacing, enforced by the worker
    _last_call: float = field(default=0.0, repr=False)

    def pace(self) -> float:
        if self.min_interval_s <= 0:
            return 0.0
        wait = max(0.0, self.min_interval_s - (time.monotonic() - self._last_call))
        if wait:
            time.sleep(wait)
        self._last_call = time.monotonic()
        return wait


@dataclass
class AcquisitionReport:
    attempted: int = 0
    written: int = 0
    skipped_existing: int = 0
    failed: int = 0
    by_provider: dict[str, int] = field(default_factory=dict)
    by_retry_class: dict[str, int] = field(default_factory=dict)
    rate_limit_wait_s: float = 0.0
    runtime_s: float = 0.0
    stopped_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted, "written": self.written,
            "skipped_existing": self.skipped_existing, "failed": self.failed,
            "by_provider": self.by_provider, "by_retry_class": self.by_retry_class,
            "rate_limit_wait_s": round(self.rate_limit_wait_s, 1),
            "runtime_s": round(self.runtime_s, 1), "stopped_reason": self.stopped_reason,
        }


def partition_path(staging: Path, symbol: str) -> Path:
    return staging / f"sym_{symbol.replace('.', '_')}.parquet"


def completed_symbols(staging: Path) -> set[str]:
    """Symbols that already have a partition on disk (resume state)."""
    return {p.stem.replace("sym_", "").replace("_", ".")
            for p in staging.glob("sym_*.parquet")}


def terminal_failures(ledger_path: Path) -> set[str]:
    """Symbols whose last recorded attempt failed permanently — do not retry."""
    if not ledger_path.exists():
        return set()
    try:
        frame = pd.read_csv(ledger_path)
    except Exception:  # noqa: BLE001 - a corrupt ledger must not block a resume
        return set()
    if frame.empty or "retry_class" not in frame.columns:
        return set()
    last = frame.groupby("symbol").tail(1)
    return set(last.loc[last["retry_class"].isin(TERMINAL_CLASSES), "symbol"].astype(str))


class BarAcquisition:
    """Fetches one dataset family for a symbol list with provider fallback."""

    def __init__(self, staging: Path, ledger_path: Path,
                 providers: Sequence[ProviderSpec],
                 cancel_file: Path | None = None,
                 yield_check: Callable[[], str | None] | None = None,
                 log: Callable[[str], None] = print) -> None:
        self.staging = staging
        self.ledger_path = ledger_path
        self.providers = list(providers)
        self.cancel_file = cancel_file
        self.yield_check = yield_check
        self.log = log
        self.staging.mkdir(parents=True, exist_ok=True)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    # -- ledger -------------------------------------------------------------
    def _append_ledger(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        frame = pd.DataFrame(rows, columns=list(LEDGER_COLUMNS))
        frame.to_csv(self.ledger_path, mode="a", index=False,
                     header=not self.ledger_path.exists())

    # -- main loop ----------------------------------------------------------
    def run(self, symbols: Iterable[str], boards: dict[str, str] | None = None,
            max_minutes: float = 60.0, checkpoint_every: int = 20,
            skip_existing: bool = True) -> AcquisitionReport:
        started = time.monotonic()
        deadline = started + max_minutes * 60
        report = AcquisitionReport()
        done = completed_symbols(self.staging) if skip_existing else set()
        skip_terminal = terminal_failures(self.ledger_path)
        pending: list[dict[str, Any]] = []
        boards = boards or {}

        for index, symbol in enumerate(symbols):
            if self.cancel_file is not None and self.cancel_file.exists():
                report.stopped_reason = "cancelled"
                break
            if time.monotonic() > deadline:
                report.stopped_reason = f"time_budget_{max_minutes}min"
                break
            if symbol in done:
                report.skipped_existing += 1
                continue
            if symbol in skip_terminal:
                report.skipped_existing += 1
                continue
            if self.yield_check is not None:
                while (reason := self.yield_check()):
                    self.log(f"yielding ({reason}); sleeping 60s")
                    time.sleep(60)
                    if time.monotonic() > deadline:
                        break

            report.attempted += 1
            board = boards.get(symbol, "")
            written = False
            for spec in self.providers:
                report.rate_limit_wait_s += spec.pace()
                try:
                    result = spec.fetch(symbol)
                except Exception as exc:  # noqa: BLE001 - never let one symbol kill the run
                    pending.append(self._ledger_row(symbol, board, spec.name, "ERROR",
                                                    RETRY_PERMANENT, 0, None, None, 1,
                                                    f"{type(exc).__name__}: {str(exc)[:160]}"))
                    report.by_retry_class[RETRY_PERMANENT] = \
                        report.by_retry_class.get(RETRY_PERMANENT, 0) + 1
                    continue
                rows = int(result.rows)
                first = last = None
                date_col = next((c for c in ("trade_date", "bar_time")
                                 if c in result.frame.columns), None)
                if rows and date_col:
                    first = str(result.frame[date_col].min())
                    last = str(result.frame[date_col].max())
                report.by_retry_class[result.retry_class] = \
                    report.by_retry_class.get(result.retry_class, 0) + 1
                if result.retry_class == RETRY_OK and rows:
                    result.frame.to_parquet(partition_path(self.staging, symbol), index=False)
                    report.written += 1
                    report.by_provider[spec.name] = report.by_provider.get(spec.name, 0) + 1
                    pending.append(self._ledger_row(symbol, board, spec.name, "WRITTEN",
                                                   RETRY_OK, rows, first, last, 1, ""))
                    written = True
                    break
                pending.append(self._ledger_row(
                    symbol, board, spec.name, "NO_DATA", result.retry_class, rows,
                    first, last, 1, result.error or ""))
            if not written:
                report.failed += 1

            if len(pending) >= checkpoint_every:
                self._append_ledger(pending)
                pending = []
                self.log(f"  {index + 1} processed · written={report.written} "
                         f"failed={report.failed} skipped={report.skipped_existing} "
                         f"{time.monotonic() - started:.0f}s")
        self._append_ledger(pending)
        report.runtime_s = time.monotonic() - started
        return report

    @staticmethod
    def _ledger_row(symbol: str, board: str, provider: str, status: str, retry_class: str,
                    rows: int, first: str | None, last: str | None, attempts: int,
                    detail: str) -> dict[str, Any]:
        return {
            "symbol": symbol, "board": board, "provider": provider, "status": status,
            "retry_class": retry_class, "rows": rows, "first_date": first, "last_date": last,
            "attempts": attempts, "detail": detail[:200],
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


def load_partitions(staging: Path, columns: Sequence[str] | None = None,
                    chunk: int = 400) -> pd.DataFrame:
    """Read every partition into one frame in bounded-memory chunks."""
    files = sorted(staging.glob("sym_*.parquet"))
    frames: list[pd.DataFrame] = []
    buffer: list[pd.DataFrame] = []
    for index, path in enumerate(files, start=1):
        buffer.append(pd.read_parquet(path, columns=list(columns) if columns else None))
        if index % chunk == 0:
            frames.append(pd.concat(buffer, ignore_index=True))
            buffer = []
    if buffer:
        frames.append(pd.concat(buffer, ignore_index=True))
    if not frames:
        return pd.DataFrame(columns=list(columns) if columns else None)
    return pd.concat(frames, ignore_index=True)


def write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def trading_day_cutoff(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Latest date whose close is certainly published (16:00 CST margin).

    Fetching before the close would persist a partial in-progress bar, which is
    the failure mode that corrupted an earlier forward-paper window.
    """
    cst = (now or pd.Timestamp.now(tz="Asia/Shanghai")).tz_localize(None) \
        if (now is None or now.tzinfo is None) else now.tz_convert("Asia/Shanghai").tz_localize(None)
    if now is None:
        cst = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    return cst.normalize() if cst.hour >= 16 else cst.normalize() - pd.Timedelta(days=1)


def board_map(master: pd.DataFrame) -> dict[str, str]:
    """symbol -> board, tolerating a master that predates the canonical classifier."""
    out: dict[str, str] = {}
    for symbol in master["symbol"].astype(str):
        try:
            out[symbol] = identify(symbol).board
        except Exception:  # noqa: BLE001
            out[symbol] = "OTHER"
    return out


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
