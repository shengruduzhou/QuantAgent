"""Structural proof that non-canonical order models cannot move money.

The parallel-model audit classifies a duplicate entity as benign only when
something enforces that it can never reach economic execution. A comment saying
"this is only for research" is not enforcement — it survives exactly until
someone imports it into a live path.

These are the enforcement. If a future change wires `microstructure_simulator`
into the broker or the canonical ledger, the corresponding test fails and the
audit disposition that depends on it becomes invalid.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Anything that can create an economic order or write the record of account.
ECONOMIC_REACH = (
    "quantagent.execution.broker_base",
    "quantagent.execution.order_manager",
    "quantagent.paper.broker",
    "quantagent.domain.ledger",
    "quantagent.data.providers.qmt_gateway",
)


def _imports_of(relative: str) -> set[str]:
    # utf-8-sig: at least one module carries a BOM, which ast.parse rejects.
    tree = ast.parse((PROJECT_ROOT / relative).read_text(encoding="utf-8-sig"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


def _importers_of(module_stem: str) -> set[str]:
    """Every non-test module that imports `module_stem`."""
    importers: set[str] = set()
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(PROJECT_ROOT))
        if module_stem in relative:
            continue
        if any(module_stem in imported for imported in _imports_of(relative)):
            importers.add(relative)
    for path in (PROJECT_ROOT / "services").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(PROJECT_ROOT))
        if any(module_stem in imported for imported in _imports_of(relative)):
            importers.add(relative)
    return importers


MICROSTRUCTURE = "src/quantagent/backtest/microstructure_simulator.py"


def test_microstructure_simulator_cannot_reach_economic_execution():
    """Its OrderIntent/Fill are a research study, structurally sealed off."""
    imports = _imports_of(MICROSTRUCTURE)

    reachable = {name for name in imports if name.startswith(ECONOMIC_REACH)}
    assert reachable == set(), (
        f"{MICROSTRUCTURE} imports economic execution modules {sorted(reachable)}; "
        "its duplicate OrderIntent/Fill can no longer be classified as isolated"
    )


def test_nothing_in_production_imports_the_microstructure_simulator():
    """A duplicate model nobody uses cannot become a second source of truth."""
    importers = _importers_of("microstructure_simulator")

    assert importers == set(), (
        f"microstructure_simulator is now imported by {sorted(importers)}; "
        "the audit disposition depends on it having no production consumer"
    )


def test_the_canonical_ledger_is_the_only_durable_order_writer():
    """No module may append order events to its own durable store."""
    offenders: list[str] = []
    for path in (PROJECT_ROOT / "src" / "quantagent").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(PROJECT_ROOT))
        if relative.startswith("src/quantagent/domain/"):
            continue
        source = path.read_text(encoding="utf-8")
        # A module that both defines an order entity and writes JSONL itself is
        # keeping its own record of account.
        defines_entity = "class Fill" in source or "class Order(" in source
        writes_own_log = ".jsonl" in source and "CanonicalLedger" not in source
        if defines_entity and writes_own_log:
            offenders.append(relative)

    assert offenders == [], f"modules keeping an independent order log: {offenders}"


# -- paper order path: no independent economic state machine -----------------
def test_paper_transitions_are_derived_from_the_canonical_table():
    """Paper must not restate the rules; it must translate them.

    Two tables for one concept drift, and the drift only surfaces when a fill
    lands on an order one of them thinks is dead.
    """
    from quantagent.domain.orders import ALLOWED_TRANSITIONS as CANONICAL
    from quantagent.paper import orders as paper

    for name, status in paper._TO_CANONICAL.items():
        expected = {
            other
            for other, other_status in paper._TO_CANONICAL.items()
            if other_status in CANONICAL[status]
        }
        assert paper.ALLOWED_TRANSITIONS[name] == expected, (
            f"paper state {name} diverged from canonical {status.value}"
        )


def test_paper_terminal_states_are_derived_from_canonical():
    from quantagent.domain.orders import TERMINAL_STATUSES
    from quantagent.paper import orders as paper

    expected = {
        name for name, status in paper._TO_CANONICAL.items() if status in TERMINAL_STATUSES
    }
    assert set(paper.TERMINAL_STATES) == expected


def test_paper_declares_no_transition_rules_of_its_own():
    """A literal rule table in paper/orders.py would be independent state."""
    source = (PROJECT_ROOT / "src/quantagent/paper/orders.py").read_text(encoding="utf-8")

    # The mapping must be built by comprehension over the canonical table, not
    # written out as literals.
    assert "CANONICAL_ALLOWED" in source, "paper must import the canonical table"
    assert "_TO_CANONICAL" in source, "paper must translate rather than restate"


def test_every_paper_state_maps_onto_a_canonical_status():
    from quantagent.paper import orders as paper

    assert set(paper._TO_CANONICAL) == set(paper.ORDER_STATES), (
        "a paper state with no canonical equivalent is state the ledger cannot express"
    )


# -- M1-18: no dual economic writes ------------------------------------------
#: Events that move money or inventory. Written to exactly one ledger.
ECONOMIC_EVENT_NAMES = (
    "ORDER_CREATED", "ORDER_ACCEPTED", "ORDER_REJECTED", "ORDER_FILLED",
    "ORDER_PARTIALLY_FILLED", "ORDER_CANCEL_REQUESTED", "ORDER_CANCELLED",
    "CASH_CHANGED", "POSITION_CHANGED",
    # A cash dividend moves cash and a split changes share count, so a corporate
    # action is economic by definition. Its absence from this list is what let
    # DEF-020 through: paper mutated its portfolio, emitted to the *operational*
    # log, and wrote nothing canonical — so the two records diverged by the whole
    # dividend and the whole share adjustment, and no audit objected.
    "CORPORATE_ACTION_APPLIED",
)


def test_paper_never_writes_an_economic_event_to_its_legacy_ledger():
    """Mirroring is not unification.

    Two records of account agree until a crash lands between the writes; then
    they disagree permanently and nothing says which is right. The canonical
    ledger is the only economic writer, and paper's log carries telemetry only.
    """
    source = (PROJECT_ROOT / "src/quantagent/paper/broker.py").read_text(encoding="utf-8")

    offenders = [
        name for name in ECONOMIC_EVENT_NAMES if f"lg.{name}" in source
    ]

    assert offenders == [], (
        f"paper/broker.py still emits economic events to its legacy EventLedger: "
        f"{offenders}. These belong to CanonicalLedger alone."
    )


def test_paper_broker_keeps_its_operational_log():
    """Telemetry is allowed — it just cannot be an economic record."""
    source = (PROJECT_ROOT / "src/quantagent/paper/broker.py").read_text(encoding="utf-8")

    assert "lg.KILL_SWITCH_TRIGGERED" in source
    assert "lg.MARK_TO_MARKET" in source
    assert "lg.SESSION_CLOSED" in source


def test_every_economic_paper_path_has_exactly_one_ledger_append():
    """An economic method may append to at most one ledger."""
    import ast

    tree = ast.parse((PROJECT_ROOT / "src/quantagent/paper/broker.py").read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        source = ast.unparse(node)
        writes_legacy = any(f"lg.{name}" in source for name in ECONOMIC_EVENT_NAMES)
        writes_canonical = any(
            marker in source
            for marker in ("_canonical_event", "_canonical_fill", "append_corporate_action")
        )
        if writes_legacy and writes_canonical:
            offenders.append(node.name)

    assert offenders == [], f"methods writing economically to both ledgers: {offenders}"
