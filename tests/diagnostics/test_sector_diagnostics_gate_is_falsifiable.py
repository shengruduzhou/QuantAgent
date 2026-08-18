"""The sector diagnostics gate must be able to fail.

Round 21 / R9 finding A-01.  `sector_usable_for_diagnostics` was the literal
`True` at its only producer, and the consumer defaulted it to `True` when the
manifest was absent.  Between them the gate could not go false for any input --
not for a map that failed validation, not for one covering nothing, not for a
manifest that did not exist.  The two sibling flags built in the same dict
(`sector_usable_for_optimization`, `st_usable_for_risk_filter`) both default
False.

AGENTS.md: "禁止把关卡写成常量 True，NOT_RUN 不得当作 PASS".  An unfalsifiable
gate records absence of evidence as a pass.
"""

from __future__ import annotations

import inspect

from quantagent.data.sector import sector_mapping
from quantagent.diagnostics import sector_audit


def test_producer_does_not_hardcode_the_gate() -> None:
    source = inspect.getsource(sector_mapping)
    assert '"sector_usable_for_diagnostics": True' not in source, (
        "the gate is a constant again; it cannot fail for any input"
    )


def test_consumer_defaults_a_missing_manifest_to_not_usable() -> None:
    """A missing manifest is missing evidence, never a pass."""
    status = sector_audit._gate_from_manifest(None)
    assert bool(status.get("sector_usable_for_diagnostics", False)) is False


def test_gate_is_false_when_validation_fails() -> None:
    source = inspect.getsource(sector_mapping)
    # The gate must read the validation result, not ignore it.
    assert 'validation["status"] == "passed"' in source


def test_zero_coverage_still_supports_diagnostics() -> None:
    """The gate must fail on invalidity, not on emptiness.

    An all-UNKNOWN map is a true exposure report, so coverage is deliberately
    not a condition here -- that is what separates this gate from the
    optimiser's, which does require coverage.
    """
    source = inspect.getsource(sector_mapping)
    gate = source.split("usable_for_diagnostics = ")[1].split("\n")[0]
    assert "coverage" not in gate
