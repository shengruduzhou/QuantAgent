"""An unevaluated concentration check must report `unknown`, never `pass`.

`RiskGate` already models three states (pass / unknown / blocked). The sector
concentration path did not use the middle one: with no risk snapshot AND no
sector series, nothing evaluated concentration, yet the gate returned
`passed=True, status="pass"` -- an affirmative claim that the book cleared a
check that never ran.

The live path is unaffected either way: `execution/live_session.py:155` is the
only caller and passes `production_mode=True`, which blocks outright on
`portfolio_risk_snapshot_missing`. This guard is for research callers, so a
promotion decision cannot read "risk passed" off an unevaluated book.
"""

from __future__ import annotations

import pandas as pd

from quantagent.risk.risk_gate import RiskGate
from quantagent.risk.risk_limits import V6RiskLimits


def _gate() -> RiskGate:
    return RiskGate(limits=V6RiskLimits())


def _book() -> pd.Series:
    # 10 names x 4% = 40% gross, all in one sector if a map is supplied.
    return pd.Series({f"S{i}": 0.04 for i in range(10)})


def _sectors() -> pd.Series:
    return pd.Series({f"S{i}": "tech" for i in range(10)})


class TestSectorConcentration:
    def test_supplied_sector_map_still_blocks_a_breach(self):
        """Guard against the fix disabling the real check."""
        result = _gate().check_target_weights(_book(), sector=_sectors())
        assert result.status == "blocked"
        assert not result.passed
        assert any(v.startswith("max_sector_weight:") for v in result.violations)

    def test_absent_sector_map_is_unknown_not_pass(self):
        result = _gate().check_target_weights(_book(), sector=None)
        assert result.status == "unknown", (
            "concentration was never evaluated, so the gate must not claim a pass"
        )
        assert not result.passed
        assert "sector_concentration_not_evaluated" in result.unknowns

    def test_production_mode_still_blocks_on_the_missing_snapshot(self):
        """Unknown must never soften an outright production block."""
        result = _gate().check_target_weights(_book(), sector=None, production_mode=True)
        assert result.status == "blocked"
        assert not result.passed
        assert "portfolio_risk_snapshot_missing" in result.violations

    def test_empty_book_does_not_manufacture_an_unknown(self):
        """Nothing held means nothing to concentrate; do not cry wolf."""
        result = _gate().check_target_weights(pd.Series(dtype=float), sector=None)
        assert "sector_concentration_not_evaluated" not in result.unknowns


class TestLimitsStillFire:
    """These already worked; pinned so the fix cannot regress them."""

    def test_single_name_concentration_rejects_the_symbol(self):
        result = _gate().check_target_weights(pd.Series({"A": 0.50, "B": 0.04}), sector=None)
        assert result.rejected_symbols.get("A") == "max_name_weight"
        assert result.status == "blocked"

    def test_gross_leverage_breach_blocks(self):
        heavy = pd.Series({f"S{i}": 0.04 for i in range(40)})  # gross 1.6 vs limit 1.0
        sectors = pd.Series({f"S{i}": f"sec{i % 8}" for i in range(40)})
        result = _gate().check_target_weights(heavy, sector=sectors)
        assert "max_leverage" in result.violations

    def test_non_finite_weights_are_rejected_by_symbol(self):
        book = pd.Series({"A": float("nan"), "B": float("inf"), "C": 0.04})
        result = _gate().check_target_weights(book, sector=None)
        assert result.rejected_symbols["A"] == "non_finite_target_weight"
        assert result.rejected_symbols["B"] == "non_finite_target_weight"

    def test_degraded_data_quality_blocks(self):
        result = _gate().check_target_weights(
            pd.Series({"A": 0.04}), sector=None, data_quality_score=0.10
        )
        assert "data_quality_below_threshold" in result.violations

    def test_model_drift_blocks(self):
        result = _gate().check_target_weights(
            pd.Series({"A": 0.04}), sector=None, model_drift_score=0.99
        )
        assert "model_drift_above_threshold" in result.violations
