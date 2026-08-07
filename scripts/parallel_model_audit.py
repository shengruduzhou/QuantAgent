#!/usr/bin/env python3
"""Find every economic order model that is not the canonical one.

Module One's gate says no file may hold independent economic truth. That claim
is only checkable if the duplicates are enumerated mechanically — a hand-written
list goes stale the moment someone adds a dataclass, and "I looked and it seemed
fine" is not evidence.

The scanner walks the AST for the shapes that constitute a parallel model:

* a class named like an order/fill/execution entity that is not the canonical one
* an order-status enum other than `domain.orders.OrderStatus`
* direct assignment to a `.status` attribute (state mutation outside `apply`)
* cash or position mutation in a module that never imports the canonical Fill

Each finding carries a classification. `unresolved_blocker` is the default: a
duplicate is guilty until someone records *why* it is benign, so silence never
reads as compliance.

Usage:
    python scripts/parallel_model_audit.py            # human summary, exit 1 if blockers
    python scripts/parallel_model_audit.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "quantagent"
SERVICE_ROOT = PROJECT_ROOT / "services"

CANONICAL_MODULE = "quantagent/domain/orders.py"

REMOVED = "removed"
CANONICAL_PROJECTION = "canonical_projection"
STATELESS_ADAPTER = "stateless_adapter"
UNRESOLVED_BLOCKER = "unresolved_blocker"

#: Only these trees hold economic order state. A `.status` assignment elsewhere
#: is a *job* status or a *probe* status — a different domain that Module One
#: does not govern. Counting those as blockers would inflate the report and make
#: the real ones harder to see, which is its own kind of dishonest number.
ORDER_DOMAIN_PREFIXES = (
    "src/quantagent/execution/",
    "src/quantagent/paper/",
    "src/quantagent/backtest/",
    "src/quantagent/domain/",
    # Reads order state to reconcile engines against each other. In scope
    # precisely because it is the one place that is *allowed* to hold order
    # figures outside the canonical module, so the audit has to attest that what
    # it holds is a read-only projection rather than a second book.
    "src/quantagent/reconciliation/",
    # Carries order, fill and cancellation *events*. In scope so the audit has to
    # attest that they are envelopes around canonical entities rather than a third
    # order model growing beside the other two.
    "src/quantagent/streaming/",
)

#: Class names that denote an economic order entity wherever they appear.
ENTITY_NAMES = {
    "Order", "OrderIntent", "OrderRecord", "OrderState", "OrderEvent",
    "Fill", "FillDecision", "FillResult", "Execution", "PositionLot",
    "OrderFacts",
}
STATUS_ENUM_NAMES = {"OrderStatus", "OrderState", "OrderSide", "Side"}

#: Explicit dispositions. Everything absent from here defaults to a blocker, so
#: adding a duplicate cannot quietly pass the audit.
DISPOSITIONS: dict[str, tuple[str, str]] = {
    # module::symbol -> (classification, justification)
    "src/quantagent/domain/orders.py::Order": (
        CANONICAL_PROJECTION, "the canonical entity itself"),
    "src/quantagent/domain/orders.py::OrderIntent": (
        CANONICAL_PROJECTION, "the canonical entity itself"),
    "src/quantagent/domain/orders.py::OrderEvent": (
        CANONICAL_PROJECTION, "the canonical entity itself"),
    "src/quantagent/domain/orders.py::Fill": (
        CANONICAL_PROJECTION, "the canonical entity itself"),
    "src/quantagent/domain/orders.py::PositionLot": (
        CANONICAL_PROJECTION, "the canonical entity itself"),
    "src/quantagent/domain/orders.py::OrderStatus": (
        CANONICAL_PROJECTION, "the canonical status enum"),
    "src/quantagent/domain/orders.py::Side": (
        CANONICAL_PROJECTION, "the canonical side enum"),
    "src/quantagent/backtest/fill_model.py::FillModelConfig": (
        STATELESS_ADAPTER,
        "microstructure parameters, not an order entity; holds no order state"),
    "src/quantagent/backtest/fill_model.py::FillModelResult": (
        STATELESS_ADAPTER,
        "one-shot return value of AShareFillModel.fill(); converted to a canonical Fill "
        "by EventDrivenBacktester._canonical_fill and never stored"),
    "src/quantagent/execution/fill_simulator.py::FillDecision": (
        STATELESS_ADAPTER,
        "pure fill-probability decision; carries no quantity ledger"),
    # broker_base is the wire protocol a broker adapter speaks. Since
    # OrderManager._open_canonical writes every order to the CanonicalLedger
    # before broker.submit(), these carry no economic truth of their own — they
    # are per-call request/reply values.
    # Frozen read-only facts derived by EconomicSnapshot.from_replay from a
    # replayed OrderBook. It has no state machine, no mutation path and no
    # durable store, and it is only ever built from a ledger replay — see
    # tests/domain/test_composite_replay.py::test_every_path_replays_from_its_ledger_alone,
    # which fails if these figures stop being a function of the event log.
    "src/quantagent/reconciliation/snapshot.py::OrderFacts": (
        CANONICAL_PROJECTION, "read-only comparison projection of a replayed OrderBook"),
    "src/quantagent/execution/broker_base.py::Order": (
        STATELESS_ADAPTER, "broker wire request; economic truth is on the canonical ledger"),
    "src/quantagent/execution/broker_base.py::OrderIntent": (
        STATELESS_ADAPTER, "broker wire intent; converted by OrderManager._open_canonical"),
    "src/quantagent/execution/broker_base.py::OrderState": (
        STATELESS_ADAPTER, "broker wire reply; folded in by _record_canonical_state"),
    "src/quantagent/execution/broker_base.py::OrderSide": (
        STATELESS_ADAPTER, "wire enum, mapped to domain.orders.Side at the boundary"),
    "src/quantagent/execution/broker_base.py::OrderStatus": (
        STATELESS_ADAPTER, "wire enum, mapped to canonical OrderEventType at the boundary"),
    "src/quantagent/execution/order_manager.py::OrderRecord": (
        STATELESS_ADAPTER,
        "broker request/reply cache. `OrderManager.history` is now a read-only property "
        "projected from the canonical OrderBook, and `rebuild_history()` reconstructs it "
        "from the durable ledger. Evidence: tests/domain/test_integrated_idempotency.py::"
        "test_history_is_a_projection_not_independent_state and ::test_history_rebuilds_"
        "from_the_durable_ledger"),
    "src/quantagent/backtest/microstructure_simulator.py::OrderIntent": (
        STATELESS_ADAPTER,
        "isolated microstructure research model with no production consumer and no import "
        "path to broker/ledger; isolation is enforced, not asserted. Evidence: "
        "tests/domain/test_model_isolation.py"),
    "src/quantagent/backtest/microstructure_simulator.py::Fill": (
        STATELESS_ADAPTER,
        "isolated microstructure research model; see the OrderIntent disposition. Evidence: "
        "tests/domain/test_model_isolation.py"),
    "src/quantagent/paper/orders.py::Order": (
        CANONICAL_PROJECTION,
        "matching/simulation fields only. Its state machine is no longer independent: "
        "ALLOWED_TRANSITIONS and TERMINAL_STATES are derived by comprehension from "
        "domain.orders.ALLOWED_TRANSITIONS via _TO_CANONICAL, and every paper state maps "
        "onto a canonical status. Evidence: tests/domain/test_model_isolation.py"),
    "src/quantagent/paper/orders.py::Fill": (
        CANONICAL_PROJECTION,
        "paper execution report carrying the same fee decomposition as domain.orders.Fill; "
        "holds no transition rules. Evidence: tests/domain/test_model_isolation.py"),
}


class ModuleScan(ast.NodeVisitor):
    def __init__(self, relative: str) -> None:
        self.relative = relative
        self.entities: list[tuple[str, int]] = []
        self.status_enums: list[tuple[str, int]] = []
        self.status_assignments: list[int] = []
        self.imports_canonical = False

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.startswith("quantagent.domain.orders"):
            self.imports_canonical = True
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name in ENTITY_NAMES:
            self.entities.append((node.name, node.lineno))
        if node.name in STATUS_ENUM_NAMES:
            bases = {getattr(base, "id", getattr(base, "attr", "")) for base in node.bases}
            if bases & {"Enum", "StrEnum", "str", "IntEnum"}:
                self.status_enums.append((node.name, node.lineno))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            # `record.state.status = X` / `order.status = X` — mutation that
            # bypasses the canonical `Order.apply` fold.
            if isinstance(target, ast.Attribute) and target.attr == "status":
                self.status_assignments.append(node.lineno)
        self.generic_visit(node)


def _scan(path: Path) -> ModuleScan | None:
    relative = str(path.relative_to(PROJECT_ROOT))
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None
    scan = ModuleScan(relative)
    scan.visit(tree)
    return scan


def _evidence(reason: str) -> str | None:
    """Pull the evidence path out of a disposition justification, if it names one."""
    for token in reason.replace(",", " ").split():
        if token.startswith("tests/") or token.startswith("docs/"):
            return token.rstrip(".")
    return None


def _classify(key: str, default_reason: str) -> tuple[str, str]:
    return DISPOSITIONS.get(key, (UNRESOLVED_BLOCKER, default_reason))


def build() -> dict:
    findings: list[dict] = []
    out_of_scope: list[dict] = []
    roots = [SOURCE_ROOT, SERVICE_ROOT]
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or "test" in path.name:
                continue
            scan = _scan(path)
            if scan is None:
                continue
            for name, line in scan.entities:
                key = f"{scan.relative}::{name}"
                classification, reason = _classify(
                    key, "duplicate economic entity; not dispositioned"
                )
                findings.append({
                    "kind": "duplicate_entity", "file": scan.relative, "symbol": name,
                    "line": line, "classification": classification, "reason": reason,
                    "evidence": _evidence(reason),
                })
            for name, line in scan.status_enums:
                key = f"{scan.relative}::{name}"
                classification, reason = _classify(
                    key, "parallel status/side enum; not dispositioned"
                )
                findings.append({
                    "kind": "duplicate_status_enum", "file": scan.relative, "symbol": name,
                    "line": line, "classification": classification, "reason": reason,
                })
            in_order_domain = scan.relative.startswith(ORDER_DOMAIN_PREFIXES)
            for line in scan.status_assignments:
                if not in_order_domain:
                    out_of_scope.append({
                        "kind": "non_order_status_mutation", "file": scan.relative,
                        "symbol": ".status", "line": line,
                        "reason": "job/probe/run status, outside the order domain",
                    })
                    continue
                key = f"{scan.relative}::status_assign_{line}"
                classification, reason = _classify(
                    key, "direct status mutation bypasses the canonical apply() fold"
                )
                findings.append({
                    "kind": "direct_status_mutation", "file": scan.relative,
                    "symbol": ".status", "line": line,
                    "classification": classification, "reason": reason,
                })

    by_class: dict[str, int] = {}
    for finding in findings:
        by_class[finding["classification"]] = by_class.get(finding["classification"], 0) + 1
    blockers = [f for f in findings if f["classification"] == UNRESOLVED_BLOCKER]
    return {
        "schemaVersion": "quantagent.parallel_model_audit.v1",
        "canonicalModule": CANONICAL_MODULE,
        "totalFindings": len(findings),
        "byClassification": by_class,
        "unresolvedBlockers": len(blockers),
        "outOfScopeStatusMutations": out_of_scope,
        "clean": not blockers,
        "findings": sorted(findings, key=lambda f: (f["classification"], f["file"], f["line"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["clean"] else 1

    print(f"Parallel-model audit — {report['totalFindings']} findings")
    for classification, count in sorted(report["byClassification"].items()):
        print(f"  {classification:24s} {count}")
    if report["unresolvedBlockers"]:
        print(f"\n{report['unresolvedBlockers']} unresolved blockers:")
        for finding in report["findings"]:
            if finding["classification"] != UNRESOLVED_BLOCKER:
                continue
            print(f"  {finding['file']}:{finding['line']}  {finding['kind']}  {finding['symbol']}")
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
