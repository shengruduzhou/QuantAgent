#!/usr/bin/env python3
"""Module One fault injection: kill it, break its disk, then check the money.

Every earlier round's "crash" test raised an exception. That covers the logical
window and not the physical one: an injected exception still unwinds the stack,
still runs `finally`, still flushes buffers. A real `SIGKILL` does none of that,
and the difference is exactly where a durable log gets torn.

So the process faults here are delivered with `signal.SIGKILL` to a real child
process at a chosen economic boundary, and the filesystem faults are injected at
the syscall the ledger actually depends on. After each one the parent re-opens the
files and checks the invariants that must hold no matter what was interrupted:

* one intent creates at most one economic order
* one execution id creates one economic effect
* no confirmed fill is lost
* the chain verifies, or reports a torn tail and stays replayable up to it
* `realised + unrealised == NAV - initial cash`
* recovery introduces no new order

Each experiment records steady state, hypothesis, blast radius, the fault, the
measurement and the verdict, per the program's chaos protocol.

Usage:
    python scripts/module1_fault_injection.py            # run all, exit 1 on any failure
    python scripts/module1_fault_injection.py --json     # machine-readable
    python scripts/module1_fault_injection.py --child ...  # internal: the victim process
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantagent.domain.ledger import (  # noqa: E402
    CanonicalLedger,
    LedgerCorruption,
    LedgerWriteUnavailable,
)
from quantagent.paper.broker import MarketSnapshot, PaperBroker  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "docs" / "architecture" / "module1_fault_injection.json"

SYMBOL = "600000.SH"
SESSION = "2026-08-04"
INITIAL_CASH = 1_000_000.0

#: Boundaries a child process can be killed at, in the order they occur.
KILL_POINTS: tuple[str, ...] = (
    "none",
    "after_claim",
    "after_first_ledger_append",
    "mid_ledger_append",
    "after_venue_fill",
    "after_execution_before_resolve",
)


def market_source(symbol: str, trade_date: str) -> MarketSnapshot | None:
    if symbol != SYMBOL:
        return None
    return MarketSnapshot(
        symbol=symbol, trade_date=trade_date, last_price=10.00, previous_close=10.00,
        session_volume=1e8, board="SH_Main",
    )


def order_payload(key: str = "k1") -> dict[str, Any]:
    return {
        "idempotencyKey": key, "runId": "run_fault", "symbol": SYMBOL, "side": "BUY",
        "quantity": 1_000, "limitPrice": 10.05, "tradeDate": SESSION, "signalId": key,
    }


# ---------------------------------------------------------------------------
# The victim
# ---------------------------------------------------------------------------
def _suicide() -> None:
    """A real SIGKILL to self.

    `os._exit` would be tidier and would prove less: it still lets the interpreter
    return from the current frame. SIGKILL cannot be caught, blocked or deferred,
    so whatever the process was part-way through stays part-way through.
    """
    os.kill(os.getpid(), signal.SIGKILL)


def run_child(root: Path, kill_at: str) -> dict[str, Any]:
    """Submit and drain one order, dying at `kill_at`. Runs in its own process."""
    from services.quant_api.services.paper_orders import PaperOrderService

    service = PaperOrderService(root, market_source=market_source, initial_cash=INITIAL_CASH)
    appends = {"n": 0}
    real_append = CanonicalLedger.append

    def counting_append(self, event, **kwargs):
        appends["n"] += 1
        if kill_at == "after_first_ledger_append" and appends["n"] == 1:
            record = real_append(self, event, **kwargs)
            _suicide()
            return record
        if kill_at == "mid_ledger_append" and appends["n"] == 1:
            # Half a line, then death: a torn trailing record produced by a signal
            # rather than by a test writing a truncated file itself.
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write('{"schemaVersion": "quantagent.canonical_led')
                    handle.flush()
                    os.fsync(handle.fileno())
            _suicide()
        return real_append(self, event, **kwargs)

    real_report = PaperBroker.apply_execution_report

    def killing_report(self, order, fill, *, trade_date=None):
        booked = real_report(self, order, fill, trade_date=trade_date)
        if kill_at == "after_venue_fill" and booked:
            _suicide()
        return booked

    real_resolve = type(service.claims).resolve

    def killing_resolve(self, key, *, outcome, payload=None):
        if kill_at == "after_execution_before_resolve" and str(key).startswith("req_"):
            _suicide()
        return real_resolve(self, key, outcome=outcome, payload=payload)

    if not service.writable:
        # Losing the writer lock is a legitimate outcome under contention, not a
        # crash. Reporting it keeps the experiment's measurement explicit rather
        # than inferred from a non-zero exit code.
        service.close()
        return {"survived": True, "wrote": False, "reason": "writer lock unavailable"}

    with (
        mock.patch.object(CanonicalLedger, "append", counting_append),
        mock.patch.object(PaperBroker, "apply_execution_report", killing_report),
        mock.patch.object(type(service.claims), "resolve", killing_resolve),
    ):
        service.submit(order_payload())
        if kill_at == "after_claim":
            _suicide()
        drained = service.drain()

    result = {"survived": True, "wrote": True, "drained": drained}
    service.close()
    return result


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------
@dataclass
class Measurement:
    ledger_records: int = 0
    torn_tail: bool = False
    chain_valid: bool = False
    orders: int = 0
    #: Orders sharing one intent's lineage. The invariant is "one intent, at most
    #: one order" — not a cap on the absolute count, which would be violated the
    #: moment an experiment legitimately issues a second intent.
    max_orders_per_intent: int = 0
    fills: int = 0
    distinct_execution_ids: int = 0
    identity_residual: float = 0.0
    orders_after_recovery: int = 0
    fills_after_recovery: int = 0
    #: False when recovery could not run. Without this, a skipped recovery leaves
    #: the "after" counts at zero and the invariant check reports a fill as lost.
    recovery_ran: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledgerRecords": self.ledger_records,
            "tornTail": self.torn_tail,
            "chainValid": self.chain_valid,
            "orders": self.orders,
            "maxOrdersPerIntent": self.max_orders_per_intent,
            "fills": self.fills,
            "distinctExecutionIds": self.distinct_execution_ids,
            "identityResidual": self.identity_residual,
            "ordersAfterRecovery": self.orders_after_recovery,
            "fillsAfterRecovery": self.fills_after_recovery,
            "recoveryRan": self.recovery_ran,
            "notes": self.notes,
        }


def measure(root: Path, *, recover: bool = True) -> Measurement:
    """Re-open everything from disk and read the economics off it."""
    from services.quant_api.services.paper_orders import PaperOrderService

    ledger_path = root / "canonical.jsonl"
    result = Measurement()
    if not ledger_path.exists():
        result.chain_valid = True
        result.notes.append("no ledger file: nothing economic was ever recorded")
        return result

    ledger = CanonicalLedger(ledger_path)
    verification = ledger.verify()
    result.ledger_records = int(verification["records"])
    result.torn_tail = bool(verification["tornTail"])
    result.chain_valid = bool(verification["valid"])
    if not result.chain_valid:
        result.notes.append(f"chain broken at record {verification.get('brokenAt')}")
        return result

    book, account = ledger.replay(initial_cash=INITIAL_CASH)
    result.orders = len(book.orders())
    per_intent: dict[str, int] = {}
    for order in book.orders():
        key = order.lineage.signal_id or order.order_id
        per_intent[key] = per_intent.get(key, 0) + 1
    result.max_orders_per_intent = max(per_intent.values(), default=0)
    result.fills = len(book.fills())
    result.distinct_execution_ids = len({f.execution_id for f in book.fills()})
    result.identity_residual = account.identity_residual({SYMBOL: 10.00})

    if recover:
        service = PaperOrderService(
            root, market_source=market_source, initial_cash=INITIAL_CASH
        )
        try:
            if service.writable:
                service.recover()
                # A second drain must not resurrect anything the fault interrupted.
                service.drain()
                after = CanonicalLedger(ledger_path).replay_book()
                result.orders_after_recovery = len(after.orders())
                result.fills_after_recovery = len(after.fills())
                result.recovery_ran = True
            else:
                result.notes.append("recovery skipped: writer lock unavailable")
        finally:
            service.close()
    return result


def check_invariants(measurement: Measurement, *, expect_fill: bool) -> list[str]:
    """Return the violated invariants. Empty means the fault was survived."""
    violations: list[str] = []
    if not measurement.chain_valid:
        violations.append("the chain does not verify")
    if measurement.max_orders_per_intent > 1:
        violations.append(
            f"one intent produced {measurement.max_orders_per_intent} economic orders"
        )
    if measurement.fills != measurement.distinct_execution_ids:
        violations.append(
            f"{measurement.fills} fills carry only "
            f"{measurement.distinct_execution_ids} distinct execution ids"
        )
    if abs(measurement.identity_residual) > 1e-6:
        violations.append(
            f"realised + unrealised does not reconcile with cash "
            f"(residual {measurement.identity_residual})"
        )
    if measurement.recovery_ran:
        if measurement.orders_after_recovery > measurement.orders:
            violations.append(
                f"recovery introduced an order: {measurement.orders} -> "
                f"{measurement.orders_after_recovery}"
            )
        if measurement.fills_after_recovery < measurement.fills:
            violations.append(
                f"a confirmed fill was lost in recovery: {measurement.fills} -> "
                f"{measurement.fills_after_recovery}"
            )
    if expect_fill and measurement.fills == 0:
        violations.append("the control run recorded no fill, so it proves nothing")
    return violations


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
@dataclass
class Experiment:
    experiment_id: str
    fault: str
    hypothesis: str
    blast_radius: str
    #: Real signal, or injected at a syscall. Stated so nobody reads an injected
    #: OSError as a proven hardware failure.
    delivery: str
    measurement: Measurement
    violations: list[str]
    child_signal: int | None = None

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "experimentId": self.experiment_id,
            "fault": self.fault,
            "hypothesis": self.hypothesis,
            "blastRadius": self.blast_radius,
            "delivery": self.delivery,
            "childSignal": self.child_signal,
            "measurement": self.measurement.to_dict(),
            "violations": self.violations,
            "passed": self.passed,
        }


def _spawn_victim(root: Path, kill_at: str) -> int | None:
    """Run the child and return the signal that killed it, or None if it survived."""
    completed = subprocess.run(
        [
            sys.executable, str(Path(__file__).resolve()),
            "--child", "--kill-at", kill_at, "--root", str(root),
        ],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=300,
    )
    if completed.returncode < 0:
        return -completed.returncode
    if completed.returncode != 0:
        raise RuntimeError(
            f"child for {kill_at} failed without a signal: {completed.stderr[-2000:]}"
        )
    return None


def experiment_sigkill(kill_at: str) -> Experiment:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "paper"
        killed_by = _spawn_victim(root, kill_at)
        expect_fill = kill_at in {"none", "after_execution_before_resolve"}
        measurement = measure(root)
        if kill_at != "none" and killed_by != signal.SIGKILL:
            measurement.notes.append(
                f"child was not killed by SIGKILL (signal={killed_by})"
            )
        return Experiment(
            experiment_id=f"sigkill.{kill_at}",
            fault=(
                "control run, no fault" if kill_at == "none"
                else f"SIGKILL to a live worker process at {kill_at}"
            ),
            hypothesis=(
                "the ledger stays replayable, at most one economic order exists, no "
                "confirmed fill is lost, and recovery adds nothing"
            ),
            blast_radius="one paper account in a temporary directory",
            delivery="control" if kill_at == "none" else "real signal.SIGKILL",
            measurement=measurement,
            violations=check_invariants(measurement, expect_fill=expect_fill),
            child_signal=killed_by,
        )


def _filesystem_experiment(
    experiment_id: str,
    fault: str,
    hypothesis: str,
    inject: Callable[[Path], list[str]],
    *,
    delivery: str,
    expect_fill: bool = False,
) -> Experiment:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "paper"
        notes = inject(root)
        measurement = measure(root)
        measurement.notes.extend(notes)
        return Experiment(
            experiment_id=experiment_id,
            fault=fault,
            hypothesis=hypothesis,
            blast_radius="one paper account in a temporary directory",
            delivery=delivery,
            measurement=measurement,
            violations=check_invariants(measurement, expect_fill=expect_fill),
        )


def _service(root: Path):
    from services.quant_api.services.paper_orders import PaperOrderService

    return PaperOrderService(root, market_source=market_source, initial_cash=INITIAL_CASH)


def _fsync_only_for(target_name: str, real_fsync):
    """An fsync that fails for one file and works for every other.

    Patching `os.fsync` wholesale is too blunt: the ledger and the idempotency
    store share the symbol, so a blanket failure hits whichever writes first — in
    practice the claim store, leaving the ledger untouched and the experiment
    proving something else entirely. The fd is resolved through /proc so the fault
    lands on the file the hypothesis is about.
    """

    def selective(fd: int) -> None:
        try:
            name = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            name = ""
        if name.endswith(target_name):
            raise OSError(5, f"simulated EIO writing {target_name}")
        return real_fsync(fd)

    return selective


def _seed_one_completed_order(service) -> list[str]:
    """Put real records on the chain before the fault.

    A fault injected against an *empty* ledger only shows that the error surfaces.
    The stronger claim — that everything already committed survives the failure —
    needs something to survive, so every storage experiment runs against a chain
    that already holds a filled order.
    """
    service.submit(order_payload("seed"))
    service.drain()
    return ["seeded one completed order before injecting the fault"]


def _inject_fsync_failure(root: Path) -> list[str]:
    service = _service(root)
    notes: list[str] = []
    try:
        notes += _seed_one_completed_order(service)
        before = len(CanonicalLedger(service.ledger_path))
        service.submit(order_payload("victim"))
        with mock.patch(
            "quantagent.domain.ledger.os.fsync",
            _fsync_only_for("canonical.jsonl", os.fsync),
        ):
            try:
                service.drain()
                notes.append("VIOLATION: the failed fsync did not surface to the caller")
            except OSError as exc:
                notes.append(f"append surfaced the failure: {exc}")
        # The latch: nothing may be written on top of a tail of unknown length.
        try:
            service.ledger.append(service.ledger.read()[0].event)
            notes.append("VIOLATION: a further append was accepted after a failed write")
        except LedgerWriteUnavailable:
            notes.append("further appends refused (fail-closed latch held)")
        notes.append(
            f"records before the fault {before}, on disk now "
            f"{len(CanonicalLedger(service.ledger_path))}"
        )
    finally:
        service.close()
    return notes


def _inject_enospc(root: Path) -> list[str]:
    service = _service(root)
    notes: list[str] = []
    try:
        notes += _seed_one_completed_order(service)
        before = len(CanonicalLedger(service.ledger_path))
        service.submit(order_payload("victim"))
        real_open = Path.open

        def full_disk(self, *args, **kwargs):
            handle = real_open(self, *args, **kwargs)
            if self.name == "canonical.jsonl" and "a" in str(
                args[0] if args else kwargs.get("mode", "")
            ):
                handle.write = mock.Mock(
                    side_effect=OSError(28, "No space left on device")
                )
            return handle

        with mock.patch.object(Path, "open", full_disk):
            try:
                service.drain()
                notes.append("VIOLATION: the full disk did not surface to the caller")
            except OSError as exc:
                notes.append(f"append surfaced the failure: {exc}")
        notes.append(
            f"records before the fault {before}, on disk now "
            f"{len(CanonicalLedger(service.ledger_path))}"
        )
    finally:
        service.close()
    return notes


def _inject_enospc_at_flush(root: Path) -> list[str]:
    """ENOSPC surfacing at `flush` rather than at `write`.

    A genuinely full filesystem does not always fail the `write` call: the bytes
    land in the buffer and the error appears when it is drained. Injecting only at
    `write` would leave the more likely shape of the real fault untested, and it is
    the more dangerous one — the caller has already been told the write succeeded.
    """
    service = _service(root)
    notes: list[str] = []
    try:
        notes += _seed_one_completed_order(service)
        before = len(CanonicalLedger(service.ledger_path))
        service.submit(order_payload("victim"))
        real_open = Path.open

        def full_disk(self, *args, **kwargs):
            handle = real_open(self, *args, **kwargs)
            if self.name == "canonical.jsonl" and "a" in str(
                args[0] if args else kwargs.get("mode", "")
            ):
                handle.flush = mock.Mock(
                    side_effect=OSError(28, "No space left on device (at flush)")
                )
            return handle

        with mock.patch.object(Path, "open", full_disk):
            try:
                service.drain()
                notes.append("VIOLATION: the failed flush did not surface to the caller")
            except OSError as exc:
                notes.append(f"append surfaced the failure: {exc}")
        try:
            service.ledger.append(service.ledger.read()[0].event)
            notes.append("VIOLATION: a further append was accepted after a failed flush")
        except LedgerWriteUnavailable:
            notes.append("further appends refused (fail-closed latch held)")
        notes.append(
            f"records before the fault {before}, on disk now "
            f"{len(CanonicalLedger(service.ledger_path))}"
        )
    finally:
        service.close()
    return notes


def _inject_read_only_path(root: Path) -> list[str]:
    service = _service(root)
    notes: list[str] = []
    try:
        notes += _seed_one_completed_order(service)
        before = len(CanonicalLedger(service.ledger_path))
        service.submit(order_payload("victim"))
        target = service.ledger_path
        original = target.stat().st_mode
        target.chmod(stat.S_IRUSR)
        try:
            service.drain()
            notes.append("VIOLATION: a read-only ledger accepted a write")
        except OSError as exc:
            notes.append(f"append surfaced the failure: {type(exc).__name__}")
        finally:
            target.chmod(original)
        notes.append(
            f"records before the fault {before}, on disk now "
            f"{len(CanonicalLedger(service.ledger_path))}"
        )
    finally:
        service.close()
    return notes


def _inject_missing_middle_record(root: Path) -> list[str]:
    service = _service(root)
    try:
        service.submit(order_payload())
        service.drain()
        path = service.ledger_path
    finally:
        service.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    removed = lines.pop(len(lines) // 2)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [f"removed one middle record ({len(removed)} bytes) from the chain"]


def _inject_corrupted_record(root: Path) -> list[str]:
    service = _service(root)
    try:
        service.submit(order_payload())
        service.drain()
        path = service.ledger_path
    finally:
        service.close()
    lines = path.read_text(encoding="utf-8").splitlines()
    index = next(
        i for i, line in reversed(list(enumerate(lines)))
        if json.loads(line)["event"].get("fill")
    )
    record = json.loads(lines[index])
    record["event"]["fill"]["quantity"] += 100
    lines[index] = json.dumps(record, sort_keys=True, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ["edited a filled quantity in place"]


def experiment_lock_contention() -> Experiment:
    """Four processes race for one queue entry. Exactly one order, and no deadlock."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "paper"
        service = _service(root)
        try:
            service.submit(order_payload())
        finally:
            service.close()  # hand the writer lock to the children

        children = [
            subprocess.Popen(
                [
                    sys.executable, str(Path(__file__).resolve()),
                    "--child", "--kill-at", "none", "--root", str(root),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                cwd=PROJECT_ROOT,
            )
            for _ in range(4)
        ]
        # The timeout *is* the deadlock check: a lock cycle would hang here.
        outcomes = [child.communicate(timeout=180) for child in children]
        codes = [child.returncode for child in children]

        measurement = measure(root)
        measurement.notes.append(f"child exit codes: {codes}")
        writers = sum(1 for out, _ in outcomes if '"wrote": true' in out.lower())
        measurement.notes.append(f"{writers} of 4 children held the writer lock")
        violations = check_invariants(measurement, expect_fill=False)
        if any(code is None for code in codes):
            violations.append("a child did not terminate: possible lock cycle")
        if any(code not in (0, None) for code in codes):
            violations.append(f"a child failed unexpectedly: exit codes {codes}")
        if writers == 0:
            violations.append("no child could ever write: the lock never handed over")
        return Experiment(
            experiment_id="lock.contention.four_processes",
            fault="four worker processes contend for one writer lock and one queue entry",
            hypothesis=(
                "exactly one economic order exists, every process terminates, and no "
                "lock cycle forms"
            ),
            blast_radius="one paper account in a temporary directory",
            delivery="real concurrent processes",
            measurement=measurement,
            violations=violations,
        )


def run_all() -> list[Experiment]:
    experiments = [experiment_sigkill(point) for point in KILL_POINTS]
    experiments.append(
        _filesystem_experiment(
            "storage.fsync_failure",
            "os.fsync raises EIO during a canonical append",
            "the failure surfaces to the caller, no further append is accepted, and a "
            "restart replays the file that is actually there",
            _inject_fsync_failure,
            delivery="injected at the os.fsync syscall, scoped to the ledger's fd",
            expect_fill=True,
        )
    )
    experiments.append(
        _filesystem_experiment(
            "storage.disk_full",
            "the ledger write fails with ENOSPC",
            "the failure surfaces and the chain remains replayable",
            _inject_enospc,
            delivery="injected at the file write (not a genuinely full filesystem)",
            expect_fill=True,
        )
    )
    experiments.append(
        _filesystem_experiment(
            "storage.disk_full_at_flush",
            "the ledger write succeeds but the flush fails with ENOSPC",
            "the failure still surfaces, the ledger latches closed, and everything "
            "already committed remains replayable",
            _inject_enospc_at_flush,
            delivery="injected at the file flush (the shape a full volume usually takes)",
            expect_fill=True,
        )
    )
    experiments.append(
        _filesystem_experiment(
            "storage.read_only_path",
            "the ledger file is made read-only under a running writer",
            "the append fails loudly rather than silently dropping an economic event",
            _inject_read_only_path,
            delivery="real chmod on the real file",
            expect_fill=True,
        )
    )
    experiments.append(
        _filesystem_experiment(
            "integrity.missing_record",
            "a record is removed from the middle of the chain",
            "verification fails and no projection is exposed",
            _inject_missing_middle_record,
            delivery="real edit to the file",
        )
    )
    experiments.append(
        _filesystem_experiment(
            "integrity.corrupted_record",
            "a filled quantity is edited in place",
            "verification fails and no projection is exposed",
            _inject_corrupted_record,
            delivery="real edit to the file",
        )
    )
    experiments.append(experiment_lock_contention())
    return experiments


#: Experiments whose whole point is that the chain must be *rejected*. For these a
#: failing verification is the pass condition, so the generic invariant check is
#: inverted rather than skipped.
MUST_BE_REJECTED = {"integrity.missing_record", "integrity.corrupted_record"}


def finalise(experiments: list[Experiment]) -> list[Experiment]:
    for experiment in experiments:
        # A note the injector marked VIOLATION is a failure, not commentary. Without
        # this the harness could print the word and still report a clean board.
        experiment.violations.extend(
            note for note in experiment.measurement.notes if note.startswith("VIOLATION")
        )
    for experiment in experiments:
        if experiment.experiment_id not in MUST_BE_REJECTED:
            continue
        if experiment.measurement.chain_valid:
            experiment.violations = [
                "a tampered chain verified: projections would be exposed from it"
            ]
        else:
            experiment.violations = []
            experiment.measurement.notes.append(
                "verification correctly refused the chain; this is the pass condition"
            )
    return experiments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kill-at", default="none", choices=KILL_POINTS)
    parser.add_argument("--root", default="")
    parser.add_argument("--report", default=str(REPORT_PATH))
    args = parser.parse_args()

    if args.child:
        print(json.dumps(run_child(Path(args.root), args.kill_at)))
        return 0

    experiments = finalise(run_all())
    payload = {
        "schemaVersion": "quantagent.module1_fault_injection.v1",
        "steadyState": {
            "description": (
                "one paper account, one queued buy of 1,000 shares at a bounded price"
            ),
            "invariants": [
                "one intent creates at most one economic order (measured per lineage, "
                "not as a cap on the absolute count)",
                "one execution id creates one economic effect",
                "no confirmed fill is lost",
                "the chain verifies or reports a torn tail and stays replayable",
                "realised + unrealised == NAV - initial cash",
                "recovery introduces no new order",
            ],
            "stopCondition": "any invariant violated by any experiment",
        },
        "experiments": [experiment.to_dict() for experiment in experiments],
        "experimentCount": len(experiments),
        "failed": [e.experiment_id for e in experiments if not e.passed],
        "clean": all(e.passed for e in experiments),
    }

    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if payload["clean"] else 1

    print(f"fault injection -> {destination.relative_to(PROJECT_ROOT)}")
    print(f"{len(experiments)} experiments, {len(payload['failed'])} failed\n")
    for experiment in experiments:
        mark = "ok  " if experiment.passed else "FAIL"
        measurement = experiment.measurement
        print(f"  [{mark}] {experiment.experiment_id}  ({experiment.delivery})")
        print(
            f"         records {measurement.ledger_records}  torn {measurement.torn_tail}  "
            f"valid {measurement.chain_valid}  orders {measurement.orders}"
            f"(max/intent {measurement.max_orders_per_intent})  "
            f"fills {measurement.fills}  residual {measurement.identity_residual:+.2e}"
        )
        for note in measurement.notes:
            print(f"         note: {note}")
        for violation in experiment.violations:
            print(f"         VIOLATION: {violation}")
    if not payload["clean"]:
        print("\nFAULT INJECTION FAILED")
        return 1
    print("\nall invariants held under every injected fault")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
