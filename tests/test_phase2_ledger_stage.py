"""The promotion stage must be derived from evidence, and must not be gameable.

DEF-018: `_permitted_stage` was a placeholder with inverted logic — it returned
`backtest_only` while streaming was incomplete and `blocked` once it was done, so
*finishing Module Two lowered the reported stage*. A stage nobody checks is a
number that drifts, and this one is the programme's headline claim.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ledger_module():
    spec = importlib.util.spec_from_file_location(
        "phase2_ledger", PROJECT_ROOT / "scripts" / "phase2_ledger.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ledger():
    module = _ledger_module()
    for requirement in module.REQUIREMENTS:
        requirement.evaluate(run_tests=False)
    return module


def test_the_current_stage_is_backtest_only(ledger):
    """Module One and Module Two are done; reconciliation and risk are not."""
    assert ledger._permitted_stage() == "backtest_only"


def test_finishing_a_module_never_lowers_the_stage(ledger, monkeypatch):
    """The exact shape of DEF-018.

    Verifying more requirements must move the stage up or leave it, never down.
    """
    order = [
        "blocked", "research_only", "backtest_only", "independently_validated",
    ]
    baseline = order.index(ledger._permitted_stage())

    for requirement in ledger.REQUIREMENTS:
        monkeypatch.setattr(requirement, "state", ledger.VERIFIED, raising=False)

    assert order.index(ledger._permitted_stage()) >= baseline


def test_a_failing_regression_suite_blocks_everything(ledger, monkeypatch):
    """No readable evidence without a passing suite, whatever else is verified."""
    for requirement in ledger.REQUIREMENTS:
        monkeypatch.setattr(requirement, "state", ledger.VERIFIED, raising=False)
    suite = next(r for r in ledger.REQUIREMENTS if r.id == ledger.SUITE)
    monkeypatch.setattr(suite, "state", ledger.IN_PROGRESS, raising=False)

    assert ledger._permitted_stage() == "blocked"


def test_an_incomplete_module_one_gate_allows_research_only(ledger, monkeypatch):
    """Order figures that are not yet a record of account make backtests unreadable."""
    for requirement in ledger.REQUIREMENTS:
        monkeypatch.setattr(requirement, "state", ledger.VERIFIED, raising=False)
    gate_item = next(r for r in ledger.REQUIREMENTS if r.id == "M1-20")
    monkeypatch.setattr(gate_item, "state", ledger.IN_PROGRESS, raising=False)

    assert ledger._permitted_stage() == "research_only"


@pytest.mark.parametrize("requirement_id", ["M1-05", "M1-06"])
def test_a_non_gate_workflow_item_does_not_hold_the_stage_down(
    ledger, monkeypatch, requirement_id
):
    """The exclusion has to be real, not just documented.

    M1-05 is order drill-down in the UI and M1-06 is cancel/expire in the fast
    engine. Neither changes whether an economic figure is correct, which is what a
    stage claims — so neither may gate one.
    """
    for requirement in ledger.REQUIREMENTS:
        monkeypatch.setattr(requirement, "state", ledger.VERIFIED, raising=False)
    workflow_item = next(r for r in ledger.REQUIREMENTS if r.id == requirement_id)
    monkeypatch.setattr(workflow_item, "state", ledger.NOT_STARTED, raising=False)

    assert ledger._permitted_stage() == "independently_validated"


def test_reconciliation_alone_holds_the_stage_at_backtest_only(ledger, monkeypatch):
    """Unreconciled engines mean results exist but are not independently validated."""
    for requirement in ledger.REQUIREMENTS:
        monkeypatch.setattr(requirement, "state", ledger.VERIFIED, raising=False)
    reconciliation = next(r for r in ledger.REQUIREMENTS if r.id == "M3-01")
    monkeypatch.setattr(reconciliation, "state", ledger.NOT_STARTED, raising=False)

    assert ledger._permitted_stage() == "backtest_only"


def test_no_amount_of_verification_can_reach_paper_ready(ledger, monkeypatch):
    """Module Six evidence cannot be granted by this function, by construction.

    A derivation that could reach `paper_ready` from requirement states alone would
    be promotion by test count, which the programme forbids.
    """
    for requirement in ledger.REQUIREMENTS:
        monkeypatch.setattr(requirement, "state", ledger.VERIFIED, raising=False)

    assert ledger._permitted_stage() == "independently_validated"


def test_every_gate_id_names_a_real_requirement(ledger):
    """A typo in a gate list would silently exclude a criterion from the stage."""
    known = {r.id for r in ledger.REQUIREMENTS}
    for name in (
        "MODULE_ONE_GATE", "MODULE_ONE_NON_GATE", "MODULE_TWO_GATE",
        "RECONCILIATION_GATE", "GOLDEN_GATE", "RISK_GATE", "ACCEPTANCE_GATE",
    ):
        unknown = [rid for rid in getattr(ledger, name) if rid not in known]
        assert unknown == [], f"{name} names requirements that do not exist: {unknown}"
    assert ledger.SUITE in known


def test_the_gate_lists_partition_module_one(ledger):
    """Every Module One requirement is either a gate criterion or explicitly not."""
    module_one = {r.id for r in ledger.REQUIREMENTS if r.id.startswith("M1-")}
    classified = set(ledger.MODULE_ONE_GATE) | set(ledger.MODULE_ONE_NON_GATE)

    assert module_one == classified, (
        f"unclassified Module One requirements: {sorted(module_one ^ classified)}. A "
        "requirement that is neither in nor out of the gate is a silent exclusion."
    )
