"""Wiring inventory for the AFML validation helpers that remain.

Round 20 found CPCV (``combinatorial_purged_split``), triple-barrier labels,
uniqueness sample weights and ``meta_label`` correct but wired to nothing, and
pinned that status here so nobody would read their existence as "validation is
AFML-grade". Round 29 deleted them: no pipeline, script or document named an
intended consumer, and an unwired lookalike of a validation method is an audit
trap, not a capability. They are recoverable from git history (the last tree
that has them is this commit's parent) if a pipeline ever adopts them, and that
adoption must arrive with its own tests.

What remains is load-bearing, and this file keeps checking that it stays wired:
validation runs on single-path purged walk-forward plus purged k-fold and PBO.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Helpers that are genuinely load-bearing.
WIRED = {
    "purged_kfold_split",
    "probability_of_backtest_overfitting",
}

#: Deleted in round 29; must not reappear as unwired lookalikes.
REMOVED = {
    "combinatorial_purged_split",
    "triple_barrier_labels",
    "sample_weights_by_uniqueness",
    "daily_volatility",
    "meta_label",
}


def _production_callers(symbol: str) -> list[str]:
    """Call sites outside tests/ and outside the symbol's own definition."""
    proc = subprocess.run(
        ["grep", "-rn", symbol, "src", "services", "scripts", "--include=*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return [
        line
        for line in proc.stdout.splitlines()
        if f"def {symbol}" not in line
        and not line.startswith("src/quantagent/quant_math/purged_cv.py")
    ]


@pytest.mark.parametrize("symbol", sorted(WIRED))
def test_wired_helpers_have_production_callers(symbol):
    assert _production_callers(symbol), (
        f"{symbol} is listed as WIRED but has no production call site"
    )


def test_removed_helpers_are_not_defined_in_quant_math():
    """Re-adding one of these without a consumer recreates the audit trap."""
    proc = subprocess.run(
        ["grep", "-rnE", r"def (" + "|".join(sorted(REMOVED)) + r")\(", "src/quantagent/quant_math"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.stdout == "", proc.stdout
