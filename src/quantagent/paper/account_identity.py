"""Immutable identity for one persistent paper/shadow economic account.

A canonical ledger contains economic events, but replay still needs two genesis
inputs supplied by its caller: ``portfolio_id`` and ``initial_cash``.  If two
workers supply different values, the same valid ledger can be reconstructed as
two different accounts/NAVs.  That is unacceptable at a production-style
boundary.

The first worker touching a *new/empty* account creates one durable identity
file beside its canonical ledger.  Every later target/execution/recovery worker
must present exactly the same identity.  A non-empty canonical ledger that
predates this contract is never silently bound to caller-supplied genesis
values: it requires an explicit, separately audited migration.  The identity is
immutable, hash-bound, fsync'd and created under a cross-process lock;
mismatches and corruptions fail closed.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import RLock
from typing import Iterator

from quantagent.domain.ledger import CanonicalLedger

try:  # POSIX research/CI hosts
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None

try:  # QMT/MiniQMT Windows hosts
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - Unix
    _msvcrt = None


ACCOUNT_IDENTITY_SCHEMA = "quantagent.paper.account_identity.v1"


class PaperAccountIdentityError(RuntimeError):
    """Base account identity failure."""


class PaperAccountIdentityMismatch(PaperAccountIdentityError):
    """A worker attempted to reinterpret an existing economic account."""


class PaperAccountIdentityCorruption(PaperAccountIdentityError):
    """The persisted identity cannot be trusted."""


class PaperAccountIdentityMigrationRequired(PaperAccountIdentityError):
    """A legacy non-empty ledger needs explicit genesis migration evidence."""


def _normalise_initial_cash(value: object) -> str:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise PaperAccountIdentityError(f"invalid initial_cash {value!r}") from exc
    if not amount.is_finite() or amount <= 0:
        raise PaperAccountIdentityError("initial_cash must be finite and > 0")
    # Fixed two-decimal CNY genesis value; callers supplying sub-cent values are
    # not allowed to create a second semantic identity through float noise.
    quantized = amount.quantize(Decimal("0.01"))
    if quantized != amount:
        raise PaperAccountIdentityError(
            "initial_cash must be representable to CNY cents exactly"
        )
    return format(quantized, "f")


def _normalise_portfolio_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        raise PaperAccountIdentityError("portfolio_id must be non-empty")
    if len(text) > 128:
        raise PaperAccountIdentityError("portfolio_id exceeds 128 characters")
    return text


def _digest(payload: dict[str, object]) -> str:
    material = dict(payload)
    material.pop("payload_sha256", None)
    canonical = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PaperAccountIdentityCorruption(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class PaperAccountIdentity:
    schema_version: str
    portfolio_id: str
    initial_cash_cny: str
    created_at: str
    payload_sha256: str

    @property
    def initial_cash(self) -> float:
        return float(Decimal(self.initial_cash_cny))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def verify(self) -> None:
        if self.schema_version != ACCOUNT_IDENTITY_SCHEMA:
            raise PaperAccountIdentityCorruption(
                f"unsupported account identity schema {self.schema_version!r}"
            )
        if _normalise_portfolio_id(self.portfolio_id) != self.portfolio_id:
            raise PaperAccountIdentityCorruption("paper account portfolio_id is not canonical")
        if _normalise_initial_cash(self.initial_cash_cny) != self.initial_cash_cny:
            raise PaperAccountIdentityCorruption("paper account initial_cash is not canonical")
        if _digest(self.to_dict()) != self.payload_sha256:
            raise PaperAccountIdentityCorruption("paper account identity digest mismatch")


class PaperAccountIdentityStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._thread_lock = RLock()

    @contextmanager
    def _exclusive_file_lock(self) -> Iterator[None]:
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
            if _msvcrt is not None:  # pragma: no cover - Windows host/CI
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
            raise PaperAccountIdentityError(
                "no supported cross-process file locking primitive"
            )

    def read(self) -> PaperAccountIdentity | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(
                self.path.read_text(encoding="utf-8"),
                object_pairs_hook=_strict_object,
            )
            identity = PaperAccountIdentity(**payload)
            identity.verify()
            return identity
        except PaperAccountIdentityError:
            raise
        except (json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
            raise PaperAccountIdentityCorruption(
                f"cannot parse paper account identity {self.path}: {exc}"
            ) from exc

    def ensure(
        self,
        *,
        portfolio_id: str,
        initial_cash: object,
        created_at: str | None = None,
    ) -> PaperAccountIdentity:
        expected_id = _normalise_portfolio_id(portfolio_id)
        expected_cash = _normalise_initial_cash(initial_cash)
        with self._thread_lock, self._exclusive_file_lock():
            existing = self.read()
            if existing is not None:
                if existing.portfolio_id != expected_id:
                    raise PaperAccountIdentityMismatch(
                        "paper account portfolio_id mismatch: "
                        f"persisted={existing.portfolio_id!r}, requested={expected_id!r}"
                    )
                if existing.initial_cash_cny != expected_cash:
                    raise PaperAccountIdentityMismatch(
                        "paper account initial_cash mismatch: "
                        f"persisted={existing.initial_cash_cny}, requested={expected_cash}"
                    )
                return existing

            payload: dict[str, object] = {
                "schema_version": ACCOUNT_IDENTITY_SCHEMA,
                "portfolio_id": expected_id,
                "initial_cash_cny": expected_cash,
                "created_at": created_at
                or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "payload_sha256": "",
            }
            payload["payload_sha256"] = _digest(payload)
            identity = PaperAccountIdentity(**payload)
            identity.verify()

            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            encoded = (
                json.dumps(identity.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
            try:
                with tmp.open("x", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, self.path)
                # Best effort directory durability on POSIX. Windows does not
                # expose a portable directory fsync through Python.
                if os.name == "posix":
                    directory_fd = os.open(self.path.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            finally:
                if tmp.exists():
                    tmp.unlink()
            return identity


def account_identity_path_for_canonical(canonical_ledger_path: str | Path) -> Path:
    return Path(canonical_ledger_path).with_name("account_identity.json")


def _assert_identity_creation_is_safe(
    *,
    canonical_ledger_path: Path,
    identity_store: PaperAccountIdentityStore,
) -> None:
    """Refuse to invent genesis values for a legacy non-empty economic chain."""

    if identity_store.read() is not None:
        return
    if not canonical_ledger_path.exists() or canonical_ledger_path.stat().st_size == 0:
        return
    ledger = CanonicalLedger(canonical_ledger_path)
    verification = ledger.verify()
    if not verification.get("valid"):
        raise PaperAccountIdentityCorruption(
            f"cannot establish account identity over invalid canonical ledger: {verification}"
        )
    if len(ledger) <= 0:
        return
    # Re-check identity after reading the ledger so a concurrent first worker
    # that successfully created the identity does not cause a false migration
    # error.  Economic writers are required to pass this identity gate before
    # appending, so a compliant new account cannot race an append ahead of it.
    if identity_store.read() is not None:
        return
    raise PaperAccountIdentityMigrationRequired(
        "canonical ledger already contains economic records but has no immutable "
        "paper account identity; automatic binding to caller-supplied portfolio_id/"
        "initial_cash is prohibited. Perform an explicit audited migration."
    )


def ensure_paper_account_identity(
    *,
    canonical_ledger_path: str | Path,
    portfolio_id: str,
    initial_cash: object,
    identity_path: str | Path | None = None,
) -> PaperAccountIdentity:
    canonical_path = Path(canonical_ledger_path)
    path = (
        Path(identity_path)
        if identity_path is not None
        else account_identity_path_for_canonical(canonical_path)
    )
    store = PaperAccountIdentityStore(path)
    _assert_identity_creation_is_safe(
        canonical_ledger_path=canonical_path,
        identity_store=store,
    )
    return store.ensure(
        portfolio_id=portfolio_id,
        initial_cash=initial_cash,
    )


__all__ = [
    "ACCOUNT_IDENTITY_SCHEMA",
    "PaperAccountIdentityError",
    "PaperAccountIdentityMismatch",
    "PaperAccountIdentityCorruption",
    "PaperAccountIdentityMigrationRequired",
    "PaperAccountIdentity",
    "PaperAccountIdentityStore",
    "account_identity_path_for_canonical",
    "ensure_paper_account_identity",
]
