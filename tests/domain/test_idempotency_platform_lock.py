from __future__ import annotations

from pathlib import Path

import quantagent.domain.idempotency as idem


class _FakeMsvcrt:
    LK_LOCK = 1
    LK_UNLCK = 2

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        self.calls.append((fd, mode, nbytes))


def test_windows_lock_backend_preserves_claim_once_protocol(tmp_path: Path, monkeypatch) -> None:
    """Linux CI still exercises the Windows-specific branch contractually.

    The fake only substitutes the OS locking primitive; JSONL persistence,
    fsync-before-grant, duplicate detection, and lock-file byte management all
    run through the real IdempotencyStore implementation.
    """
    fake = _FakeMsvcrt()
    monkeypatch.setattr(idem, "_fcntl", None)
    monkeypatch.setattr(idem, "_msvcrt", fake)

    path = tmp_path / "claims.jsonl"
    store = idem.IdempotencyStore(path)
    first = store.claim("economic-key", payload={"clientOrderId": "cid-1"})
    duplicate = store.claim("economic-key", payload={"clientOrderId": "cid-1"})

    assert first.granted is True
    assert duplicate.granted is False
    assert [mode for _, mode, _ in fake.calls] == [
        fake.LK_LOCK,
        fake.LK_UNLCK,
        fake.LK_LOCK,
        fake.LK_UNLCK,
    ]
    assert all(nbytes == 1 for _, _, nbytes in fake.calls)
    lock_path = path.with_suffix(path.suffix + ".lock")
    assert lock_path.read_bytes() == b"\0"


def test_durable_store_reloads_claim_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "claims.jsonl"
    first = idem.IdempotencyStore(path)
    assert first.claim("restart-key", payload={"value": 1}).granted is True

    restarted = idem.IdempotencyStore(path)
    replay = restarted.claim("restart-key", payload={"value": 1})
    assert replay.granted is False
    assert replay.record.payload["value"] == 1
