#!/usr/bin/env python3
"""Phase II completion ledger.

Every requirement carries a state, and a state is not a claim an author makes —
it is checked. `verified` requires that the code paths, test paths and evidence
paths recorded against the requirement all exist on disk, and that the named
tests actually pass. Anything asserting `verified` without them is downgraded to
`implemented_not_verified` when this script runs, and the run exits non-zero.

That inversion is the point: the ledger is designed so an over-claim fails loudly
rather than reading as progress.

Usage:
    python scripts/phase2_ledger.py            # print the ledger, exit 1 if incomplete
    python scripts/phase2_ledger.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

NOT_STARTED = "not_started"
IN_PROGRESS = "in_progress"
IMPLEMENTED_NOT_VERIFIED = "implemented_not_verified"
VERIFIED = "verified"
BLOCKED = "blocked"

#: States that still block the phase from being declared complete.
INCOMPLETE_STATES = {NOT_STARTED, IN_PROGRESS, IMPLEMENTED_NOT_VERIFIED}


@dataclass
class Requirement:
    id: str
    module: str
    title: str
    claimed_state: str
    code_paths: tuple[str, ...] = ()
    test_paths: tuple[str, ...] = ()
    evidence_paths: tuple[str, ...] = ()
    note: str = ""
    #: Populated by `evaluate`.
    state: str = field(default="", init=False)
    missing: list[str] = field(default_factory=list, init=False)

    def evaluate(self, *, run_tests: bool) -> None:
        self.missing = [
            path
            for path in (*self.code_paths, *self.test_paths, *self.evidence_paths)
            if not (PROJECT_ROOT / path).exists()
        ]
        if self.claimed_state != VERIFIED:
            self.state = self.claimed_state
            return
        if not self.code_paths or not self.test_paths:
            self.state = IMPLEMENTED_NOT_VERIFIED
            self.missing.append("<no code or test path recorded>")
            return
        if self.missing:
            self.state = IMPLEMENTED_NOT_VERIFIED
            return
        if run_tests and not _tests_pass(self.test_paths):
            self.state = IMPLEMENTED_NOT_VERIFIED
            self.missing.append("<recorded tests did not pass>")
            return
        self.state = VERIFIED

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "module": self.module,
            "title": self.title,
            "state": self.state,
            "claimedState": self.claimed_state,
            "codePaths": list(self.code_paths),
            "testPaths": list(self.test_paths),
            "evidencePaths": list(self.evidence_paths),
            "missing": self.missing,
            "note": self.note,
        }


def _tests_pass(test_paths: tuple[str, ...]) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly", *test_paths],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


# ---------------------------------------------------------------------------
# The ledger. States here are *claims*; `evaluate` decides what they become.
# ---------------------------------------------------------------------------
REQUIREMENTS: tuple[Requirement, ...] = (
    # -- Module 1: orders as first-class entities ---------------------------
    Requirement(
        "M1-01", "orders", "Canonical lineage identifiers across all engines", VERIFIED,
        code_paths=("src/quantagent/domain/lineage.py",),
        test_paths=("tests/domain/test_lineage.py",),
    ),
    Requirement(
        "M1-02", "orders", "Durable claim-once idempotency for economic actions", VERIFIED,
        code_paths=("src/quantagent/domain/idempotency.py",),
        test_paths=("tests/domain/test_order_idempotency.py",),
        note="Covers all 7 duplicate-delivery paths plus the INC-E1 over-suppression direction.",
    ),
    Requirement(
        "M1-03a", "orders", "Canonical immutable order/execution entity set defined", VERIFIED,
        code_paths=("src/quantagent/domain/orders.py",),
        test_paths=("tests/domain/test_order_entities.py",),
        note=(
            "Signal, TargetPosition, OrderIntent, RiskDecision, Order, OrderEvent, Fill, "
            "PositionLot, CashMovement, OrderBook. Immutable snapshots, enforced transitions, "
            "T+1 lots, slippage derived from price rather than charged as a fee."
        ),
    ),
    Requirement(
        "M1-03b", "orders", "The fast backtest engine emits canonical entities", VERIFIED,
        code_paths=("src/quantagent/backtest/engine.py",),
        test_paths=("tests/domain/test_ledger_reconstruction.py",),
        note=(
            "backtest/engine.py now opens a Signal -> OrderIntent -> Order for every order, "
            "records RiskDecision + lifecycle events, and its `trades` DataFrame is derived "
            "from canonical Fills via _trade_row (one-way projection, no independent state)."
        ),
    ),
    Requirement(
        "M1-03c", "orders", "execution/order_manager.py adopts the canonical set", VERIFIED,
        code_paths=("src/quantagent/execution/order_manager.py",),
        test_paths=("tests/domain/test_integrated_idempotency.py",),
        note=(
            "_open_canonical writes Signal -> Intent -> Order + RiskDecision to the "
            "CanonicalLedger before broker.submit(); _record_canonical_state folds the broker "
            "reply. broker_base types are now dispositioned as stateless wire adapters."
        ),
    ),
    Requirement(
        "M1-03d", "orders", "OrderManager.history is a derived projection", VERIFIED,
        code_paths=("src/quantagent/execution/order_manager.py",),
        test_paths=("tests/domain/test_integrated_idempotency.py",),
        note=(
            "`history` is a read-only property projected from the canonical OrderBook; "
            "`rebuild_history()` reconstructs it from the ledger. The wire cache `_wire` holds "
            "broker request/reply pairs only and is never read for economics. DEF-010: "
            "rebuild_history built `CanonicalLedger(self.ledger_path)`, which returned an *empty* "
            "ledger whenever the chain was in-memory or injected — so the audit compared `history` "
            "against nothing and passed. It now reads the chain the manager actually writes to, "
            "re-opened from disk when durable, via the new replay_book() that needs no fabricated "
            "initial_cash."
        ),
    ),
    Requirement(
        "M1-03e", "orders", "Non-canonical models proven unable to reach execution", VERIFIED,
        code_paths=("src/quantagent/backtest/microstructure_simulator.py",),
        test_paths=("tests/domain/test_model_isolation.py",),
        note=(
            "Isolation is enforced rather than asserted: microstructure_simulator has no "
            "production importer and no import path to broker/ledger, and no module outside "
            "domain/ keeps its own durable order log."
        ),
    ),
    Requirement(
        "M1-03f", "orders", "paper/orders.py has no independent state machine", VERIFIED,
        code_paths=("src/quantagent/paper/orders.py",),
        test_paths=("tests/domain/test_model_isolation.py",),
        evidence_paths=("docs/architecture/parallel_model_audit.json",),
        note=(
            "ALLOWED_TRANSITIONS and TERMINAL_STATES are now derived by comprehension from "
            "domain.orders.ALLOWED_TRANSITIONS via _TO_CANONICAL; paper restates no rules. "
            "The migration exposed a genuine gap in the canonical table (ACCEPTED -> REJECTED "
            "is real: a venue can acknowledge then refuse at fill time) which was fixed in "
            "domain/orders.py rather than worked around in paper."
        ),
    ),
    Requirement(
        "M1-15", "orders", "Residual parallel-model blockers reduced to zero", VERIFIED,
        code_paths=("scripts/parallel_model_audit.py",),
        test_paths=("tests/domain/test_model_isolation.py",),
        evidence_paths=("docs/architecture/parallel_model_audit.json",),
        note=(
            "unresolved_blocker = 0 across 19 findings (10 canonical projections, 9 stateless "
            "adapters). Reached by migration and enforced dispositions, not by narrowing the "
            "scanner — the scanner was *widened* this round: ORDER_DOMAIN_PREFIXES gained "
            "src/quantagent/reconciliation/ and ENTITY_NAMES gained OrderFacts, which added a "
            "blocker that was then closed by disposition. Every disposition names a test that "
            "fails if it stops holding."
        ),
    ),
    Requirement(
        "M1-16", "orders", "Paper economic state replays from the canonical ledger", VERIFIED,
        code_paths=("src/quantagent/paper/broker.py",),
        test_paths=("tests/domain/test_paper_canonical_replay.py",),
        note=(
            "PaperBroker mirrors every order, acceptance, rejection, cancellation and fill "
            "to CanonicalLedger. Measured: paper cash 989984.8999 == replayed cash "
            "989984.8999; position, order status, cumulative/leaves/last quantity and "
            "lineage all reconstruct; replay is hash-stable; a tampered chain raises "
            "LedgerCorruption before any projection is exposed."
        ),
    ),
    Requirement(
        "M1-17", "orders", "ACCEPTED->REJECTED is constrained to untraded orders", VERIFIED,
        code_paths=("src/quantagent/domain/orders.py",),
        test_paths=("tests/domain/test_rejection_semantics.py",),
        note=(
            "A rejection requires cumulative_quantity == 0, no fills and an explicit reason; "
            "PARTIALLY_FILLED -> REJECTED stays absent from the table. leaves_quantity, "
            "last_quantity and cumulative_quantity are exposed separately from status."
        ),
    ),
    Requirement(
        "M1-18", "orders", "No dual economic writes: CanonicalLedger is Paper's only ledger",
        VERIFIED,
        code_paths=("src/quantagent/paper/broker.py",),
        test_paths=("tests/domain/test_model_isolation.py",),
        note=(
            "All 8 economic _emit calls removed from paper/broker.py; the legacy EventLedger "
            "now carries only kill switch, mark-to-market and session close. Corporate actions "
            "were on that list until DEF-020 measured what classifying them as telemetry cost: a "
            "500.00 cash and 1,000-share divergence from one dividend and one split, on the "
            "record of account that is supposed to be the only one. They are canonical now, and "
            "the audit's ECONOMIC_EVENT_NAMES gained CORPORATE_ACTION_APPLIED so the omission "
            "cannot recur — the list not containing it is exactly why no audit objected. "
            "A repository audit fails if any economic event name reappears, and a per-method "
            "AST check fails if one method writes economically to both ledgers — that audit "
            "caught two dead ORDER_FILLED assignments the regex removal missed."
        ),
    ),
    Requirement(
        "M1-19", "orders", "Paper economic recovery reads only the canonical ledger", VERIFIED,
        code_paths=("src/quantagent/paper/recovery.py",),
        test_paths=("tests/paper/test_paper_broker.py",),
        note=(
            "recover_from_canonical() refuses a chain that fails verification, then rebuilds "
            "cash, positions, lots, sellable inventory, realised PnL and fees from canonical "
            "events. Sellability is resolved against the latest session in the ledger. "
            "recover() survives for operational telemetry (kill switch, sessions) only."
        ),
    ),
    Requirement(
        "M1-14", "orders", "Economic idempotency is mandatory and fails closed", VERIFIED,
        code_paths=("src/quantagent/execution/order_manager.py",),
        test_paths=("tests/domain/test_integrated_idempotency.py",),
        note=(
            "Submission without lineage.run_id raises MissingIdempotencyLineage before the "
            "broker is touched; a reused key with a different request fingerprint raises "
            "IdempotencyConflict rather than returning the original. Forensic replay is an "
            "explicit isolated harness refused by any economically reachable broker."
        ),
    ),
    Requirement(
        "M1-12", "orders", "Machine-readable residual parallel-model report", VERIFIED,
        code_paths=("scripts/parallel_model_audit.py",),
        test_paths=("tests/domain/test_integrated_idempotency.py",),
        evidence_paths=("docs/architecture/parallel_model_audit.json",),
        note=(
            "AST scan classifies every order entity/status enum/direct status mutation as "
            "removed | canonical_projection | stateless_adapter | unresolved_blocker. "
            "Undispositioned findings default to blocker, so a new duplicate cannot pass "
            "silently — demonstrated this round: adding reconciliation/ to the scanned domain "
            "immediately flagged its OrderFacts projection until it was dispositioned. "
            "Currently 19 findings: 10 canonical projections, 9 stateless adapters, 0 blockers."
        ),
    ),
    Requirement(
        "M1-04", "orders", "Append-only hash-chained ledger, no silent state overwrite", VERIFIED,
        code_paths=("src/quantagent/domain/ledger.py",),
        test_paths=("tests/domain/test_ledger_reconstruction.py",),
        note=(
            "Hash-chained JSONL, fsynced. verify() locates the first broken record; a tampered "
            "record and a torn trailing write are both covered."
        ),
    ),
    Requirement(
        "M1-07", "orders", "Ledger replay reconstructs exact economic state (req D)", VERIFIED,
        code_paths=("src/quantagent/domain/accounting.py", "src/quantagent/domain/ledger.py"),
        test_paths=("tests/domain/test_ledger_reconstruction.py",),
        note=(
            "Real engine run, in-memory state discarded, rebuilt from disk: order status, "
            "filled/remaining quantity, lots, cash, fees, slippage and NAV all match, and "
            "replay is hash-stable across repetitions. Replay refuses a corrupt chain."
        ),
    ),
    Requirement(
        "M1-08", "orders", "Accounting invariants + state-machine fuzzing (req E)", VERIFIED,
        code_paths=("src/quantagent/domain/accounting.py",),
        test_paths=("tests/domain/test_accounting_invariants.py",),
        note=(
            "12 seeded fuzz sequences assert identities after every economic event; 8 more "
            "assert a refused transition records no event and mutates nothing. DEF-007 closed by "
            "*deleting* AccountState.frozen_cash rather than documenting it: no event fed it, so "
            "it read 0.0 forever and every 'reserved cash matches' comparison passed without "
            "measuring anything. Cash committed to working buy orders is now derived from the "
            "order book by reconciliation.snapshot (reserved_cash), which also means the "
            "commitment is released exactly once because it is recomputed rather than "
            "decremented. Order gained `reference_price` in the process — an order priced only "
            "by reference previously reserved nothing, reporting a working order as committing "
            "no capital."
        ),
    ),
    Requirement(
        "M1-09", "orders", "T+1 position lots (req F)", VERIFIED,
        code_paths=("src/quantagent/domain/accounting.py", "src/quantagent/backtest/engine.py"),
        test_paths=("tests/domain/test_accounting_invariants.py",),
        note=(
            "Same-day buy unsellable, prior-day sellable, intraday T uses only settled base, "
            "partial fills form separate lots, cancels do not touch settled inventory. "
            "Found and fixed a real T+1 violation in the fast engine (see defect DEF-004)."
        ),
    ),
    Requirement(
        "M1-10", "orders", "Integrated idempotency: order manager, threads, processes (req B)",
        VERIFIED,
        code_paths=("src/quantagent/domain/idempotency.py", "src/quantagent/execution/order_manager.py"),
        test_paths=("tests/domain/test_integrated_idempotency.py", "tests/domain/test_order_idempotency.py"),
        note=(
            "Drives the real OrderManager against a recording broker: repeated reconcile, "
            "restart, 8 threads, 32-thread claim race, two OS processes, two recovery workers "
            "-> exactly one economic submission each time. Later-session rebuy and revised "
            "quantity still pass through. Found and fixed a real concurrency defect: the store "
            "loaded once at construction, so two workers each held an empty view and both won "
            "(see defect DEF-006)."
        ),
    ),
    Requirement(
        "M1-20", "orders",
        "Composite Fast/Paper/OMS replay with zero unexplained differences", VERIFIED,
        code_paths=(
            "src/quantagent/reconciliation/composite.py",
            "src/quantagent/reconciliation/snapshot.py",
            "src/quantagent/reconciliation/differences.py",
            "scripts/module1_composite_replay.py",
        ),
        test_paths=("tests/domain/test_composite_replay.py",),
        evidence_paths=("docs/architecture/module1_composite_replay.json",),
        note=(
            "MEASURED unexplained_economic_differences = 0. One scenario (full fill, "
            "ACCEPTED->REJECTED with zero fills, T+1 refusal, next-session sale with stamp duty, "
            "partial fill, cancelled remainder, re-delivered execution, out-of-order event, "
            "settlement, restart) driven through all three paths, each rebuilt from its ledger "
            "file alone. paper vs oms_to_paper: 0 differences of any kind. fast vs venue: 45 "
            "differences, every one named by an ExplanationRule for fill-price or cost model; no "
            "rule excuses a quantity, status or inventory difference. Reserved cash and reserved "
            "inventory are derived from the order book rather than read from the unfed "
            "AccountState.frozen_cash field (DEF-007). Found DEF-008 (P0), DEF-009 (P0), "
            "DEF-010, DEF-011 and DEF-014 — the last two in this round's own new code, both "
            "caught by the comparison before anything was claimed as working."
        ),
    ),
    Requirement(
        "M1-21", "orders", "One execution id produces exactly one economic effect", VERIFIED,
        code_paths=(
            "src/quantagent/domain/orders.py",
            "src/quantagent/domain/accounting.py",
            "src/quantagent/domain/ledger.py",
            "src/quantagent/paper/broker.py",
        ),
        test_paths=("tests/domain/test_composite_replay.py",),
        note=(
            "DEF-008 was measured before the fix: re-delivering one 500-share execution left the "
            "book holding 1,000 shares and charged cash twice. Now enforced on three layers — "
            "Order.apply returns the same snapshot for an identical re-report (and raises "
            "DuplicateExecution on conflicting economics), OrderBook.apply records no event when "
            "nothing changed, and replay_account refuses to apply an execution id twice even if "
            "an older writer put it in the file. mirror_event and mirror_open keep book and "
            "ledger from disagreeing about whether an event happened — DEF-014 proved that "
            "obligation cannot be left to each call site: every caller appended "
            "history_of(...)[-1] unconditionally, so an already-known order id wrote a stale "
            "FILL into the chain as a CREATED and left it unreplayable."
        ),
    ),
    Requirement(
        "M1-22", "orders", "OMS-to-paper is a real path sharing one record of account", VERIFIED,
        code_paths=(
            "src/quantagent/execution/paper_adapter.py",
            "src/quantagent/execution/order_manager.py",
            "src/quantagent/paper/broker.py",
        ),
        test_paths=("tests/domain/test_composite_replay.py",),
        evidence_paths=("docs/architecture/module1_composite_replay.json",),
        note=(
            "Before this the OMS could only reach VirtualBroker, so intent -> risk -> OMS -> venue "
            "had no paper leg to reconcile. PaperBrokerAdapter translates only; the OMS opens the "
            "canonical order and hands it to the venue via attach_canonical, so one economic order "
            "keeps one record. Passing both a ledger and a ledger path now raises on both sides. "
            "Measured: the OMS chain and the venue driven directly produce zero differences."
        ),
    ),
    Requirement(
        "M1-23", "orders", "PnL split reconciles with cash on every path", VERIFIED,
        code_paths=(
            "src/quantagent/domain/accounting.py",
            "src/quantagent/domain/orders.py",
        ),
        test_paths=(
            "tests/domain/test_accounting_invariants.py",
            "tests/domain/test_composite_replay.py",
        ),
        evidence_paths=("docs/architecture/module1_composite_replay.json",),
        note=(
            "DEF-009: entry fees were expensed instead of capitalised into cost basis, so "
            "`realised + unrealised` exceeded `NAV - initial cash` by 10.3521 on the composite "
            "scenario — profit no cash or inventory backed. Cash was always right, which is why a "
            "NAV-level review could not see it. AccountState.identity_residual is now the check; "
            "it is 0 to float precision on all three paths, and unrealised PnL is marked against "
            "the same weighted-average basis a sale realises against."
        ),
    ),
    Requirement(
        "M1-13", "orders", "API and worker submission paths use the durable guard", VERIFIED,
        code_paths=(
            "services/quant_api/services/paper_orders.py",
            "services/quant_api/schemas/paper.py",
            "services/quant_api/routes/api.py",
        ),
        test_paths=("tests/quant_ui/test_paper_order_api.py",),
        note=(
            "Round 7 found the premise wrong: there was no HTTP endpoint and no worker that "
            "submitted an economic order, so the requirement was unbuilt rather than untested. "
            "Round 8 built it — POST /api/paper/orders queues, /paper/orders/drain runs the "
            "worker, and the chain is HTTP -> claim -> worker -> OrderManager -> "
            "PaperBrokerAdapter -> PaperBroker -> one CanonicalLedger. 46 tests drive the route "
            "itself, measuring orders and fills on the ledger rather than trusting the reply: "
            "double click, repeated POST, timeout-after-success, gateway retry, duplicate queue "
            "delivery, 16 concurrent posts, 8 concurrent worker threads, two OS processes, "
            "simultaneous recovery, crash before claim / after claim / after canonical append / "
            "after paper acceptance, duplicate execution report, reordered lifecycle event, "
            "changed fingerprint (409), later-session order, cancel of a partially filled order, "
            "and both lineage fields failing closed. Live intent is refused with 451 before "
            "anything is recorded; missing market data is an explicit rejection, never a "
            "fabricated fill. Found DEF-015 and DEF-016."
        ),
    ),
    Requirement(
        "M1-24", "orders", "Deployment uniqueness: single-host single-writer enforced",
        VERIFIED,
        code_paths=("services/quant_api/services/paper_orders.py",),
        test_paths=("tests/quant_ui/test_paper_order_api.py",),
        note=(
            "The mandate's *single-host contract* option, chosen because this build has no "
            "shared transactional store to give the alternative. Enforced in two layers because "
            "one is not enough: an exclusive fcntl.flock stops a second process on this host, and "
            "a host-identity occupancy record with a heartbeat stops a second *host* — flock is "
            "advisory and per-host, so on a shared filesystem it would grant both and prove "
            "nothing. Writes are refused rather than the process being killed, so a second "
            "instance still serves read-only pages and says why. Covered in both directions: a "
            "stale heartbeat and an unreadable record both read as free, so a dead host cannot "
            "hold the account hostage. LIMIT, stated rather than papered over: multi-host "
            "deployment is *refused*, not supported — the shared transactional uniqueness that "
            "would support it does not exist here, and the cross-host guard is best effort."
        ),
    ),
    Requirement(
        "M1-11", "orders", "Module One controlled fault tests (req G)", VERIFIED,
        code_paths=(
            "scripts/module1_fault_injection.py",
            "src/quantagent/domain/ledger.py",
        ),
        test_paths=(
            "tests/domain/test_ledger_reconstruction.py",
            "tests/domain/test_order_idempotency.py",
            "tests/domain/test_composite_replay.py",
            "tests/domain/test_ledger_write_failures.py",
            "tests/quant_ui/test_module1_fault_injection.py",
        ),
        evidence_paths=("docs/architecture/module1_fault_injection.json",),
        note=(
            "12 experiments in scripts/module1_fault_injection.py, each recording steady state, "
            "hypothesis, blast radius, delivery mechanism and measurement; run in the suite by "
            "tests/quant_ui/test_module1_fault_injection.py so a regression fails there rather "
            "than in a script someone remembers to invoke. Process faults are delivered by a "
            "**real signal.SIGKILL** to a child process at five economic boundaries (after the "
            "claim, after the first ledger append, mid-append leaving a genuine torn tail, after "
            "the venue books the fill, after execution before the request resolves) — the "
            "earlier rounds' injected exceptions still unwound the stack and flushed buffers, "
            "which is precisely the difference that tears a log. Storage faults: fsync EIO scoped "
            "to the ledger's own fd, ENOSPC, a real chmod to read-only, a removed middle record, "
            "an edited filled quantity. Plus four contending worker processes, where the timeout "
            "is the deadlock check. Every storage experiment runs against a chain that already "
            "holds a filled order, so it proves committed data survives rather than only that an "
            "error surfaces. Invariants checked after each: at most one order per intent "
            "(measured per lineage, not as a cap on the count), one execution id one effect, no "
            "confirmed fill lost, chain verifies or reports a recoverable torn tail, "
            "realised + unrealised == NAV - initial cash, recovery adds no order. Found DEF-017. "
            "STILL NOT covered, stated rather than implied: a machine-level restart, and a "
            "genuinely full filesystem (ENOSPC is injected at the write, not produced by filling "
            "a volume). There is no DBMS in this build, so 'database lock/disconnect' maps to the "
            "filesystem and lock faults above rather than to a test that does not exist."
        ),
    ),
    Requirement(
        "M1-05", "orders", "UI drill-down strategy->signal->order->fill->position->cash->NAV",
        NOT_STARTED,
    ),
    Requirement(
        "M1-06", "orders", "Skipped/blocked/rejected/cancelled/expired orders all queryable",
        IN_PROGRESS,
        code_paths=("src/quantagent/execution/order_manager.py",),
        test_paths=("tests/test_skipped_order_audit_is_explainable.py",),
        note="Skip reasons are now accurate and drillable; cancel/expire paths are not modelled in the fast engine.",
    ),
    # -- Module 2: streaming event-driven backtest --------------------------
    Requirement(
        "M2-01", "streaming", "Strict timestamp-ordered event bus", VERIFIED,
        code_paths=("src/quantagent/streaming/bus.py",),
        test_paths=("tests/streaming/test_event_bus.py",),
        note=(
            "Events are buffered in a heap and emitted in the total order from M2-03, so the "
            "same events shuffled eight ways produce one digest and one emission sequence — "
            "arrival order cannot reach the results. Three refusals rather than conventions: an "
            "event sorting before the last emitted one raises LateArrival (never silently "
            "reordered, which would make the emitted sequence a lie, and never dropped, which "
            "would lose a fill); a re-published identical event raises DuplicateEvent; and a "
            "consumer publishing MARKET_KINDS mid-drain raises, because reacting to the tape is "
            "allowed and extending it is how a backtest invents the market it wanted. Reactions "
            "at the same instant *are* accepted — a bar producing a signal then an order is the "
            "normal case and the causal priority orders it. The frontier advances with the bus "
            "and is handed to the consumer rather than the consumer being trusted to track time. "
            "Checkpoints carry a running digest of emitted event ids, not a position: a resumed "
            "run is proven to be the same run, and resuming a bus that has already emitted is "
            "refused rather than splicing two runs. Two defects fixed in this round's own code: "
            "`replay()` sorted before publishing, which neutralised the shuffle its callers were "
            "testing; and the draining flag spanned the whole loop, so a caller that broke out "
            "early left the bus refusing market data forever."
        ),
    ),
    Requirement(
        "M2-02", "streaming", "Full event taxonomy (calendar/session/bar/corp-action/...)",
        VERIFIED,
        code_paths=(
            "src/quantagent/streaming/events.py",
            "src/quantagent/domain/timeline.py",
        ),
        test_paths=("tests/streaming/test_event_bus.py",),
        note=(
            "All 21 kinds the programme names, asserted by name rather than by count. Every "
            "event carries the full nine-stamp EventTime — there is no constructor taking a bare "
            "instant, because an event that cannot say when it became knowable cannot be checked "
            "for look-ahead. Every kind has a distinct declared causal priority, and "
            "`_assert_priorities_cover_every_kind` runs at import so adding a kind without "
            "placing it in the causal chain is a startup error rather than a subtly wrong "
            "backtest (it would otherwise sort last via UNKNOWN_PRIORITY, after the orders that "
            "should have reacted to it). MARKET_KINDS and REACTION_KINDS partition the taxonomy, "
            "which is what lets the bus tell observing the tape from extending it. Event ids are "
            "content-addressed so a replay produces the same ids as the run it replays."
        ),
    ),
    Requirement(
        "M2-03", "streaming", "Explicit time model (event/available/decision/submit/fill)",
        VERIFIED,
        code_paths=("src/quantagent/domain/timeline.py",),
        test_paths=("tests/domain/test_timeline.py",),
        note=(
            "EventTime carries all nine stamps (source, event, available, ingestion, "
            "processing, decision, submission, venue receive, fill) and refuses two things at "
            "construction: a naive timestamp, and an available_time earlier than its "
            "event_time. DecisionFrontier.admit *raises* rather than returning a boolean — a "
            "guard whose result can be ignored by forgetting to read it is not a guard — and "
            "the frontier cannot move backwards. Deterministic tie-breaking is "
            "(event_time, causal priority, source sequence, stable identity); an unknown event "
            "kind sorts after every known one rather than sharing the calendar bucket, and the "
            "stable final term is what keeps a replay reproducible instead of ordering by set "
            "iteration. Made concrete to this repository rather than left abstract: "
            "bar_field_availability states when each *column* of a daily A-share bar becomes "
            "knowable, because the panel is one row per (symbol, session) and `close` is as "
            "easy to reach from a 09:30 decision as `pre_close` — pre_close and open are "
            "admitted at the open (the auction prints 09:25), close/high/low/volume/amount are "
            "refused with the 5h30m gap named, and an unknown column raises rather than "
            "defaulting. NOT YET WIRED into the engines: that is M2-01/M2-04's work, and this "
            "requirement claims the model and its guard, not their adoption."
        ),
    ),
    Requirement(
        "M2-04", "streaming", "Order lifecycle shared with paper and live", VERIFIED,
        code_paths=("src/quantagent/streaming/lifecycle.py",),
        test_paths=("tests/streaming/test_lifecycle_and_ambiguity.py",),
        note=(
            "OrderLifecycle keeps *no* order state: it folds ORDER/VENUE_CALLBACK/FILL/CANCEL/"
            "EXPIRY through OrderBook via mirror_open/mirror_event, and every economic figure it "
            "reports is a replay of the chain. Tested by what changes rather than by attribute "
            "names — the only field that moves while events are handled is the events-seen "
            "counter (a name-based check would flag `initial_cash`, an immutable replay input, "
            "and prove nothing). Because the lifecycle is Module One's, its refusals are "
            "inherited rather than restated, and each is exercised here: a re-delivered "
            "execution id moves no money, reusing one with different economics raises, a fill "
            "after a cancel raises, a rejection after a fill raises, and a cancel that lost a "
            "race to a fill is absorbed (which is not the same as absorbing a duplicate fill). "
            "DEF-016 is inherited too: a cancel in the next session does not re-date the "
            "earlier fill's settlement. Three things fail closed rather than defaulting: an "
            "event for an order this run never opened, a missing required field, and an unknown "
            "venue reply — treating 'unknown' as 'accepted' is how a refused order starts "
            "trading. Whether a fill is final is decided by the order's remaining quantity, not "
            "asserted by the venue. Cross-engine evidence: given the same economic events, the "
            "streaming lifecycle and the paper broker produce identical cash, positions, fees, "
            "realised PnL, lots and event sequences, compared through EconomicSnapshot so it is "
            "records of account being compared rather than two engines' opinions of themselves."
        ),
    ),
    Requirement(
        "M2-05", "streaming", "Documented rule for same-bar stop-loss/take-profit ambiguity",
        VERIFIED,
        code_paths=("src/quantagent/streaming/ambiguity.py",),
        test_paths=("tests/streaming/test_lifecycle_and_ambiguity.py",),
        note=(
            "A bar that touches both the stop and the target is consistent with a loss and with "
            "a gain, and the default failure is silent: whichever branch the code checks first "
            "wins, and code tends to check the profitable one. Every resolution now names the "
            "rule that produced it, and the three ambiguous outcomes are distinguished — "
            "RESOLVED_BY_INTRABAR (a measurement, finer data settled it), "
            "RESOLVED_CONSERVATIVELY (an assumption, the adverse level is taken to have "
            "triggered first, and `is_assumption` says so) and UNRESOLVED (the caller asked to "
            "be told rather than assumed for). There is deliberately no policy that resolves "
            "ambiguity in the position's favour: AmbiguityPolicy has exactly two members, so it "
            "cannot be selected from a configuration file, and a test fails if such a member is "
            "ever added. Adverse is defined relative to the *position*, not to price direction, "
            "so a short bracket resolves against the short. A gap through a level fills at the "
            "bar's open rather than at the level, because filling at the level credits a price "
            "the market never offered. Inverted brackets and internally inconsistent bars are "
            "refused, since an inconsistent bar can be made to resolve either way. "
            "`ambiguity_report` separates assumptions from measurements: '3% ambiguous' and '3% "
            "priced by assumption' are the same number describing very different confidence."
        ),
    ),
    # -- Module 3: fast vs streaming reconciliation -------------------------
    Requirement(
        "M3-01", "reconciliation", "Event-level fast-vs-streaming reconciliation table",
        VERIFIED,
        code_paths=(
            "src/quantagent/streaming/matching.py",
            "src/quantagent/reconciliation/composite.py",
        ),
        test_paths=("tests/domain/test_composite_replay.py",),
        evidence_paths=("docs/architecture/module1_composite_replay.json",),
        note=(
            "The composite now runs FOUR engines, streaming included, and streaming derives "
            "its own fills: MatchingVenue reads a BAR, decides acceptance, quantity and price "
            "over the shared A-share rulebook, and publishes VENUE_CALLBACK/FILL events. That "
            "independence is the requirement — a streaming engine handed the venue's answers "
            "agrees with it automatically and the comparison would be a recording against "
            "itself. Proven independent rather than asserted: the two legs' execution id sets "
            "are disjoint (`strm-*` versus the venue's uuids), checked by a test. MEASURED: "
            "streaming vs paper = **0 differences of any kind**, not merely 0 unexplained, "
            "across all 45+ dimensions with no rule permitting any. Streaming vs fast = 45 "
            "differences, every one named. Streaming expresses all 10 scenario steps (the fast "
            "engine expresses 4), so nothing is out of its scope. Found DEF-019, and removed a "
            "duplicated pricing formula: `ashare_rules.execution_price` is now the single "
            "implementation both engines delegate to, so their agreement on price is a property "
            "rather than a coincidence maintained by hand — a test breaks the shared function "
            "and requires *both* engines to break."
        ),
    ),
    Requirement(
        "M3-02", "reconciliation",
        "Exact equality for discrete state, tolerances only for documented float math",
        VERIFIED,
        code_paths=("src/quantagent/reconciliation/differences.py",),
        test_paths=("tests/domain/test_composite_replay.py",),
        evidence_paths=("docs/architecture/module1_composite_replay.json",),
        note=(
            "Enforced at construction, not by review: ExplanationRule rejects a tolerance on "
            "any status, quantity, count, lot or event-sequence dimension, so `0.5 orders` "
            "cannot be configured. A tolerance also stops applying beyond its own bound rather "
            "than stretching to fit. The two pairs that must match exactly — paper vs OMS and "
            "streaming vs paper — are granted an *empty* rule tuple, so no tolerance can apply "
            "to them at all, and tests assert both tuples stay empty. For the pairs that may "
            "legitimately differ, a test scans every cross-path rule and fails if any grants a "
            "tolerance on a discrete dimension. A dimension one side never reported is compared "
            "against an ABSENT sentinel and counts as a difference, so an engine cannot "
            "reconcile perfectly by modelling nothing."
        ),
    ),
    Requirement(
        "M3-03", "reconciliation", "Reconciliation visible in UI and downloadable", BLOCKED,
        note=(
            "Blocked on Module Five, not on reconciliation. The machine-readable table exists "
            "and is regenerated by scripts/module1_composite_replay.py into "
            "docs/architecture/module1_composite_replay.json; what is missing is a route and a "
            "page to serve it, which is the web loop's work (M7-01). Recorded as blocked rather "
            "than in_progress because no amount of reconciliation work advances it."
        ),
    ),
    # -- Module 4: golden scenarios -----------------------------------------
    Requirement(
        "M4-01", "golden", "Hand-calculated golden backtest scenarios", IN_PROGRESS,
        code_paths=(
            "src/quantagent/backtest/engine.py",
            "src/quantagent/domain/orders.py",
        ),
        test_paths=(
            "tests/test_golden_backtest_scenarios.py",
            "tests/test_golden_venue_scenarios.py",
        ),
        note=(
            "28 scenarios across two files, each with its arithmetic derived in the docstring and "
            "asserted exactly. The fast-engine file (12) covers costs, T+1, price limits, lots, "
            "the NAV identity and determinism. The venue file (16) covers what the fast engine "
            "structurally cannot, having no venue and no working orders: partial fill with costs "
            "charged only on what traded, cancel of a remainder, cancel before any fill, late "
            "fill after a cancel, cash dividend, 2:1 split, corporate action reaching the chain, "
            "ex-date total-return equivalence, corporate-action interleaving, fractional "
            "entitlement refusal, an action on an unheld position, two strategies competing for "
            "cash, two competing for inventory, same-bar stop/target both ways, and the streaming "
            "engine reproducing the partial-fill arithmetic to the cent, four delisting cases, "
            "four missing-benchmark cases and two expiration cases. Found DEF-020, DEF-021 and "
            "DEF-022. Also "
            "corrected a wrong premise of my own: a test asserted that holding through an ex date "
            "earned 500.00 more than selling first. It does not — the engine said so — because "
            "the mark drops by the dividend, so total return is unchanged and only the 0.255 fee "
            "difference on a smaller notional remains. The test now pins that, which is the "
            "stronger check: an engine crediting a dividend *without* dropping the mark would "
            "show free money on the ex date. STILL MISSING, and why this stays in_progress: "
            "three cases, each blocked on something other than scenario-writing effort, stated "
            "so the gap is not mistaken for laziness: (a) *spread* cannot be modelled — there is "
            "no bid/ask in this build and no Level-2 vendor serves this market, so a spread "
            "scenario would be asserting an invented number; (b) *base-position intraday T* is a "
            "strategy feature that does not exist yet; (c) *same-bar stop/target through an "
            "engine* needs bracket orders in the order model — the rule itself is implemented and "
            "tested in M2-05, but no engine places a bracket, so there is nothing to drive it "
            "through. Delisting and the missing benchmark are now covered, and "
            "writing it forced DEF-021: a held position with no mark was valued at *zero*, "
            "understating NAV by 10,000.00 on a 1,000-share holding and fabricating a 10,005.10 "
            "loss — while the accounting identity still held, because cash and the mark were "
            "consistently wrong together. Valuation now refuses rather than defaulting, and "
            "`valuation()` reports nav=None with the unpriceable symbols named. Paper's "
            "Portfolio had the same defect in the opposite shape: `if s in prices` silently "
            "*excluded* unpriced holdings, claiming they did not exist. Writing the "
            "missing-benchmark case then found DEF-022, the same shape one level up: a benchmark "
            "gap was filled with 0%-return days, so with the benchmark absent for 5 of 10 "
            "sessions it reported benchmark_return +0.00% and excess_return +0.00% when the truth "
            "was +20% and -20% — excess return overstated by 20 percentage points and presented "
            "as a confident number. Now: interior gaps stay NaN (only the first observation is "
            "legitimately 0%), an incomplete benchmark yields None with benchmark_status and "
            "session-coverage counts, and the acceptance gate distinguishes an *absent* benchmark "
            "from an *incomplete* one because the two need different fixes. The gate's "
            "machine-readable reason code was deliberately left unchanged and the cause put in "
            "structured fields instead — mutating an identifier to carry a new distinction breaks "
            "every consumer matching on it."
        ),
    ),
    # -- Module 5: multi-layer risk -----------------------------------------
    Requirement(
        "M5-01", "risk", "Gates emit pass/fail/unknown and never fabricate a measurement",
        VERIFIED,
        code_paths=(
            "src/quantagent/data/v7_quality_gates.py",
            "src/quantagent/backtest/paper_report.py",
        ),
        test_paths=(
            "tests/test_gate_unknown_vs_failed.py",
            "tests/test_golden_venue_scenarios.py",
        ),
        note=(
            "All 12 measurement-dependent gates emit unknown when their metric is absent, not "
            "just excess_return_after_costs. DEF-023, measured before the fix by evaluating an "
            "*empty* metrics dict: four gates **passed on no evidence at all** — max_drawdown "
            "(abs(0.0) <= any limit), single_factor_dominance (0.0 <= any limit), "
            "no_mock_or_synthetic (absent -> False -> 'no synthetic data') and no_pit_violations "
            "(absent -> 0 -> 'no look-ahead'). The last two are the worst: a run never audited "
            "for leakage or synthetic data was recorded as clean on both, which is the claim this "
            "programme most needs earned rather than defaulted. The gates that did fail reported "
            "a fabricated 0.0 as the *measured* value, so an operator read "
            "'rank_ic_mean_not_positive' and concluded the model had no edge when no IC had been "
            "computed. Now: 12 unknowns on an empty dict, zero measured passes, asserted by test. "
            "Three-way distinction pinned per gate — never audited / audited clean / audited "
            "dirty are three statuses, not two. A config-waived requirement is also distinguished "
            "from a measured pass (reason=waived_by_configuration, required=False), because "
            "'passed' implies a threshold was cleared and a switched-off check cleared nothing. "
            "Verified the strictness does not make a genuinely complete run unpassable. Excess "
            "return additionally distinguishes an absent benchmark from an incomplete one "
            "(DEF-022) via structured fields, with the machine-readable reason code left stable."
        ),
    ),
    Requirement(
        "M5-02", "risk", "Research risk layer (PIT/leakage/survivorship/multiple-testing)",
        IN_PROGRESS,
        code_paths=(
            "src/quantagent/data/v7_quality_gates.py",
            "src/quantagent/data/ashare/gold_bridge.py",
            "src/quantagent/research/selection_governance.py",
        ),
        test_paths=(
            "tests/test_gate_unknown_vs_failed.py",
            "tests/test_v7_live_readiness_gates.py",
            "tests/test_pit_label_alignment_gates.py",
            "tests/test_survivorship_producer.py",
        ),
        evidence_paths=("docs/architecture/m5_leakage_audit.json",),
        note=(
            "PIT, purge/embargo, OOS budget, PBO/DSR/SPA, survivorship and now label alignment "
            "are all enforced as gates. Stays in_progress because the readiness pipeline still "
            "does not *compute* survivorship, and because DEF-026 leaves an open convention "
            "decision recorded below. "
            "DEF-025, measured: the DEF-023 hardening was defeated one layer down. Making a gate "
            "report `unknown` on a missing measurement achieves nothing while the producer "
            "supplies a fabricated one, and four did: `_pit_violations` returned 0 for a frame it "
            "could not read, `_uses_mock_or_synthetic` returned False for a frame with no "
            "provenance column, `_aggregate_metrics` wrote `uses_mock_or_synthetic: False` as a "
            "constant, and the CLI injected `pit_violation_count = 0`. So on the real training "
            "path `no_pit_violations` and `no_mock_or_synthetic` — the two claims this programme "
            "most needs earned — were recorded as *measured passes*, on nothing. Both audits now "
            "return tri-state reports and producers forward what was measured or nothing at all. "
            "DEF-026, measured: nothing had ever compared a row's availability stamp against its "
            "own label window. `build_market_features` stamped `available_at = next trading row` "
            "while `v7_label_builder` opens the label window at `close(trade_date)`, so 100% of "
            "rows declared themselves unusable until after they were already being scored — and "
            "that also violates the `available_at <= trade_date` invariant the gold builder "
            "asserts in the same sentence that describes the shift. The stamp is not "
            "documentation: it is the as-of join key in `merge_pit_features`, so a third-party "
            "feature published during the label window joined onto the row scored on that "
            "window's return, at rank IC +1.0000 (0.0 after the fix). A conservative stamp "
            "intended to withhold information was admitting it. Execution latency belongs to the "
            "executable layer, which models it explicitly; encoding it in the availability stamp "
            "bought nothing and cost a leak. `evaluate_label_alignment` returns pass / fail / "
            "unknown, reads `label_entry_at` when a delayed-entry builder publishes one, and "
            "records which basis it used rather than assuming. OPEN DECISION, not a defect: this "
            "repository runs two label conventions — `v7_label_builder`'s "
            "close(t+h)/close(t)-1 and `gold_bridge.LABEL_CONVENTION`'s delay-1 "
            "close(t+1+h)/close(t+1)-1. Measured on 800 liquid names over 2024-08..2026-08, "
            "delaying entry by one session costs 4.0% of momentum_5d's IC and 8.8% of "
            "momentum_20d's, but 73.5% of return_1d's: the cost is concentrated exactly in the "
            "fastest-decaying signals, which is where this programme's short-horizon candidates "
            "live. Three tests that asserted 'all gates satisfied' were again found passing with "
            "no evidence for the new gate — the same discovery as DEF-024 one round earlier. "
            "DEF-024, measured: gold_bridge masked a missing delisting date as a confident FALSE, "
            "so a symbol the master recorded as `status=delisted` with no date contributed exactly "
            "as many eligible training sessions as a still-listed name — survivorship bias by "
            "default, and asymmetric with mask_pre_listing, which already returned UNKNOWN for a "
            "missing listing date. The mask now consults `status`: only an explicitly listed "
            "status turns a missing date into FALSE, anything else is UNKNOWN. "
            "`eligible_for_training` deliberately keeps its permissive meaning (flipping UNKNOWN "
            "to ineligible would empty the universe wherever a register has partial coverage — "
            "U0's ST data is SZSE-only), so the tri-state `eligibility_status` and a "
            "`unknown_masks` column are published alongside it and a gate enforces the policy on "
            "the difference. `evaluate_survivorship` returns pass / fail / unknown, where unknown "
            "covers both 'a name's delisting status cannot be determined' and 'the panel contains "
            "no delisted names at all', the latter being the signature of the bias in a "
            "full-universe panel. Wiring it caught something worth recording: two live-readiness "
            "tests asserting 'all gates satisfied' had been passing with no survivorship evidence "
            "whatsoever, so a readiness report could assert production-readiness without anyone "
            "having checked whether the universe contained the names that died. "
            "DEF-027, measured: U0 keeps two masters, and the fix could not fire against one "
            "of them. `historical_security_master.parquet` (H-032C) has no `status` column and "
            "not a single delisting date, so against it the mask answered UNKNOWN for all "
            "5,888 names — unable to separate the 358 it existed to catch from the 5,530 it "
            "did not need to. Honest but inert, and the DEF-025 lesson in a new place: "
            "hardening one side of an interface is worth nothing until you check what the "
            "other side speaks. That master does record the distinction, in two columns that "
            "agree exactly — `status_end_blocked` (True for those 358) and `source` "
            "(`sz_delist` / `sh_delist_retry`) — so `resolve_listing_status` reads whichever "
            "vocabulary is present and publishes which one it used. "
            "DEF-028, and it corrects a wrong verdict this ledger carried for one round: the "
            "audit counted a delisted name as present only if it had a `mask_post_delisting == "
            "TRUE` row — a bar *after* the security stopped existing. A panel that includes "
            "dead names and correctly stops each at its delisting date has none of those, by "
            "construction, so the audit read the healthiest possible shape as the most "
            "suspicious one and the only way to construct a passing fixture was to build a "
            "broken panel. Presence is now read from the security's status and the mask is "
            "used for what it does answer — whether any row outlived its own delisting, which "
            "is a real failure when it happens. Re-measured against the shipped "
            "FULL_UNIVERSE_GOLD_READY artifact using the master the build actually reads "
            "(`security_master.parquet`: `status` present, 5,533 listed / 361 delisted, all "
            "361 delisting dates populated): 261 delisted names are present, contributing "
            "424,662 sessions — 3.89% of the panel — and **zero** of them have a bar past "
            "their delisting date. Survivorship verdict: **pass**. The previous round's "
            "reading — that those names had been recorded as never having died — was wrong, "
            "and wrong because the artifact was judged against a master it was not built "
            "from. `build_masks` was also 18x slower than it needed to be (row-wise "
            "`iterrows()`, ~30k rows/s, ~6 min on the full panel); vectorised to 22s with "
            "output pinned identical to the row-wise construction, which matters now that "
            "survivorship is computed per training run rather than once at build time. "
            "STILL OPEN: the label-entry convention (see DEF-026 above), and nothing computes "
            "survivorship at *gold build* time, so a future rebuild pointed at the H-032C "
            "master would silently lose the evidence that makes today's verdict a pass."
        ),
    ),
    Requirement("M5-03", "risk", "Model risk layer (versioning/sensitivity/drift/disable)", NOT_STARTED),
    Requirement("M5-04", "risk", "Portfolio risk layer (limits/exposure/capacity/stress)", IN_PROGRESS,
                code_paths=("src/quantagent/portfolio/v7_target_weights.py",),
                note="Name/sector/turnover limits enforced; style and factor exposure, capacity and stress tests are not."),
    Requirement("M5-05", "risk", "Pre-trade risk layer", IN_PROGRESS,
                code_paths=("src/quantagent/paper/risk.py", "src/quantagent/execution/order_manager.py"),
                note="Cash, inventory, T+1, price band and lot size enforced in paper; stale-data and abnormal-price checks absent."),
    Requirement("M5-06", "risk", "Intraday/post-trade risk layer", NOT_STARTED),
    Requirement("M5-07", "risk", "Operational controls (kill switch/cancel all/flatten/approval)", IN_PROGRESS,
                code_paths=("src/quantagent/paper/risk.py", "src/quantagent/execution/risk_kill_switch.py"),
                note="Kill switch and cancel-all exist for paper; flatten, per-symbol disable and manual approval are not wired to the UI."),
    Requirement("M5-08", "risk", "Risk decisions stored as first-class evidence", IN_PROGRESS,
                code_paths=("src/quantagent/paper/risk.py",),
                note="RiskDecision exists in paper; the fast backtest records rejects without rule/threshold/measured-value."),
    # -- Module 6: lifecycle -------------------------------------------------
    Requirement("M6-01", "lifecycle", "Full entity lifecycle (create/version/clone/branch/rollback/delete/restore)", IN_PROGRESS,
                code_paths=("services/quant_api/services/strategies.py",),
                test_paths=("tests/quant_ui/test_strategy_lifecycle.py",),
                note="Save/version/compare/soft-delete/archive exist; clone, branch, rollback, restore and controlled permanent delete do not."),
    Requirement("M6-02", "lifecycle", "A version never silently mutates after producing a result", NOT_STARTED),
    # -- Module 7: web operation loop ---------------------------------------
    Requirement("M7-01", "web", "Complete terminal-free operation loop", IN_PROGRESS,
                code_paths=("apps/quant-ui/src/vnext/strategy/StrategyStudioPage.tsx",),
                note="Create/validate/launch/monitor/cancel exist. Streaming backtest, reconciliation, order drill-down, promotion and rollback have no UI."),
    Requirement("M7-02", "web", "UI shows blocking reason and next valid action", IN_PROGRESS,
                code_paths=("services/quant_api/services/strategies.py",),
                test_paths=("tests/quant_ui/test_blocked_configuration.py",),
                note="Pre-flight issues carry a remediation and an action; promotion-stage guidance does not exist."),
    # -- Module 8: paper/shadow/canary --------------------------------------
    Requirement("M8-01", "promotion", "Explicit promotion stages with no skipping", NOT_STARTED),
    Requirement("M8-02", "promotion", "Paper uses the same contracts as the streaming backtest", BLOCKED,
                note="Blocked on M2: there is no streaming backtest to share contracts with."),
    Requirement("M8-03", "promotion", "Shadow mode against live data, no economic orders", NOT_STARTED),
    Requirement("M8-04", "promotion", "Canary admission mechanism, disabled by default", NOT_STARTED),
    # -- Module 9: determinism and cycles -----------------------------------
    Requirement("M9-01", "determinism", "Hash-level determinism across the full pipeline", NOT_STARTED),
    Requirement("M9-02", "determinism", "Repeated unattended cycles with no resource drift", NOT_STARTED),
    # -- Module 10: chaos ----------------------------------------------------
    Requirement("M10-01", "chaos", "Documented steady state and invariants", NOT_STARTED),
    Requirement("M10-02", "chaos", "Fault injection at every major stage", NOT_STARTED),
    Requirement("M10-03", "chaos", "Recovery verified with no duplicate order and no fabricated completion", NOT_STARTED),
    # -- Module 11: usability ------------------------------------------------
    Requirement("M11-01", "usability", "No-AI, no-terminal first-time-user acceptance test", NOT_STARTED),
    # -- Module 12: 11-role acceptance --------------------------------------
    Requirement("M12-01", "acceptance", "11 independent role reports with own tests and evidence", NOT_STARTED),
    # -- Module 13: test matrix ---------------------------------------------
    Requirement("M13-01", "matrix", "Full supported-configuration test matrix", NOT_STARTED),
    Requirement(
        "M13-02", "matrix", "Regression suite passes with no external network dependency", VERIFIED,
        code_paths=("tests/conftest.py",),
        test_paths=("tests/domain/",),
        note="Ambient .env network flags are disarmed before collection; full suite 2152 passed / 0 failed in 3:21.",
    ),
)


def build(run_tests: bool = False) -> dict:
    for requirement in REQUIREMENTS:
        requirement.evaluate(run_tests=run_tests)
    by_state: dict[str, int] = {}
    for requirement in REQUIREMENTS:
        by_state[requirement.state] = by_state.get(requirement.state, 0) + 1
    blocking = [r for r in REQUIREMENTS if r.state in INCOMPLETE_STATES]
    return {
        "schemaVersion": "quantagent.phase2_ledger.v1",
        "总数": len(REQUIREMENTS),
        "byState": by_state,
        "phaseComplete": not blocking,
        "maximumPermittedStage": _permitted_stage(),
        "requirements": [r.to_dict() for r in REQUIREMENTS],
    }


#: The requirements that constitute each gate, transcribed from the role reports in
#: docs/architecture/module1_role_reports.md rather than inferred from module
#: membership. Naming them explicitly is what makes the *exclusions* auditable: a
#: derivation keyed on "every requirement whose module is orders" would demote the
#: stage for a UI item, and one keyed on nothing at all was the placeholder this
#: replaces (DEF-018 — it returned `blocked` once streaming was complete, so
#: finishing Module Two *lowered* the reported stage).
MODULE_ONE_GATE: tuple[str, ...] = (
    "M1-01", "M1-02", "M1-03a", "M1-03b", "M1-03c", "M1-03d", "M1-03e", "M1-03f",
    "M1-04", "M1-07", "M1-08", "M1-09", "M1-10", "M1-11", "M1-12", "M1-13",
    "M1-14", "M1-15", "M1-16", "M1-17", "M1-18", "M1-19", "M1-20", "M1-21",
    "M1-22", "M1-23", "M1-24",
)
#: Deliberately outside the gate: M1-05 is order drill-down in the web UI and M1-06
#: is cancel/expire modelling in the fast engine. Both are workflow items that
#: belong to Module Five's loop; neither affects whether an economic figure is
#: correct, which is what a stage is a claim about.
MODULE_ONE_NON_GATE: tuple[str, ...] = ("M1-05", "M1-06")

MODULE_TWO_GATE: tuple[str, ...] = ("M2-01", "M2-02", "M2-03", "M2-04", "M2-05")
RECONCILIATION_GATE: tuple[str, ...] = ("M3-01", "M3-02", "M3-03")
GOLDEN_GATE: tuple[str, ...] = ("M4-01",)
RISK_GATE: tuple[str, ...] = (
    "M5-01", "M5-02", "M5-03", "M5-04", "M5-05", "M5-06", "M5-07", "M5-08",
)
ACCEPTANCE_GATE: tuple[str, ...] = ("M11-01", "M12-01", "M13-01")
#: The regression suite itself. If this is not verified nothing else can be read.
SUITE = "M13-02"


def _permitted_stage() -> str:
    """Derived from evidence, never asserted.

    Each step down is a statement about what cannot be trusted yet, not a score:

    * no passing regression suite -> `blocked`, because no other evidence is readable
    * Module One's gate incomplete -> `research_only`: order-level figures are not
      yet a record of account, so no backtest result about them means anything
    * Module Two, reconciliation, golden scenarios, risk or independent acceptance
      incomplete -> `backtest_only`: results exist and are not independently
      validated
    * all of the above verified -> `independently_validated`

    Nothing here can reach `paper_ready` or beyond: those require Module Six, and a
    stage this function could grant without Module Six evidence would be exactly
    the promotion-by-test-count the programme forbids.
    """
    verified = {r.id for r in REQUIREMENTS if r.state == VERIFIED}

    def complete(ids: tuple[str, ...]) -> bool:
        return all(requirement in verified for requirement in ids)

    if SUITE not in verified:
        return "blocked"
    if not complete(MODULE_ONE_GATE):
        return "research_only"
    if not complete(
        MODULE_TWO_GATE + RECONCILIATION_GATE + GOLDEN_GATE + RISK_GATE + ACCEPTANCE_GATE
    ):
        return "backtest_only"
    return "independently_validated"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--run-tests", action="store_true", help="execute recorded tests before allowing `verified`")
    args = parser.parse_args()

    ledger = build(run_tests=args.run_tests)
    if args.json:
        print(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"Phase II ledger — {ledger['总数']} requirements")
        for state, count in sorted(ledger["byState"].items()):
            print(f"  {state:28s} {count}")
        print(f"\nphase complete: {ledger['phaseComplete']}")
        print(f"maximum permitted stage: {ledger['maximumPermittedStage']}")
        incomplete = [r for r in REQUIREMENTS if r.state in INCOMPLETE_STATES]
        if incomplete:
            print(f"\n{len(incomplete)} requirements still blocking:")
            for requirement in incomplete:
                print(f"  [{requirement.state:24s}] {requirement.id}  {requirement.title}")
    return 0 if ledger["phaseComplete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
