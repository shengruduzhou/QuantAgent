"""Cross-process serialization for one durable paper account.

Signal generation and pending-signal execution touch different artifacts but
share one economic source of truth: the canonical ledger. A process-local
``threading.RLock`` cannot prevent a scheduler process from appending fills while
another process is freezing a target from an older account snapshot.

This module provides a small standard-library-only advisory lock keyed by the
canonical-ledger path. The operating system releases the lock if the process
exits, avoiding stale lock-directory recovery. POSIX uses ``fcntl.flock``;
Windows uses a one-byte ``msvcrt.locking`` region because QMT deployments may
run on Windows.

The lock is deliberately scoped to paper/shadow account orchestration. It is not
used by vectorized research/backtest ledgers, so research throughput is not
serialized by an operational safety primitive.
"""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import tempfile
import time
from typing import Iterator


class PaperAccountLockTimeout(RuntimeError):
    """Another process held the paper-account mutation/freeze lock too long."""


def paper_account_lock_path(canonical_ledger_path: str | os.PathLike[str]) -> Path:
    # Lock identity follows filesystem identity, not a caller's spelling of the
    # ledger path. ``strict=False`` resolves existing symlink/junction components
    # while still supporting a first-run ledger file that does not yet exist.
    path = Path(canonical_ledger_path).resolve(strict=False)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return path.with_name(f"{path.name}.account.lock")
    identity = f"{stat.st_dev}:{stat.st_ino}".encode("ascii")
    digest = sha256(identity).hexdigest()
    return (
        Path(tempfile.gettempdir())
        / "quantagent-paper-account-locks"
        / f"{digest}.lock"
    )


@contextmanager
def paper_account_lock(
    canonical_ledger_path: str | os.PathLike[str],
    *,
    timeout_seconds: float = 30.0,
    poll_seconds: float = 0.05,
) -> Iterator[Path]:
    """Acquire the inter-process lock for a canonical paper ledger.

    Acquisition is bounded and fail-closed. A caller that cannot establish an
    atomic account boundary must abort instead of silently falling back to an
    unlocked snapshot.
    """

    canonical_path = Path(canonical_ledger_path).resolve(strict=False)
    lock_path = paper_account_lock_path(canonical_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    with lock_path.open("a+b") as handle:
        # Windows byte-range locking requires the byte to exist.
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)

        if os.name == "nt":  # pragma: no cover - exercised on Windows CI/QMT hosts
            import msvcrt

            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise PaperAccountLockTimeout(
                            f"timed out acquiring paper account lock {lock_path}"
                        ) from exc
                    time.sleep(max(0.001, float(poll_seconds)))
            try:
                yield lock_path
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise PaperAccountLockTimeout(
                        f"timed out acquiring paper account lock {lock_path}"
                    ) from exc
                time.sleep(max(0.001, float(poll_seconds)))
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


__all__ = [
    "PaperAccountLockTimeout",
    "paper_account_lock",
    "paper_account_lock_path",
]
