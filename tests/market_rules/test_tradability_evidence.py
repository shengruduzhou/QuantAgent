"""Absent tradability evidence must not read as "tradeable".

Seven call sites independently wrote::

    for col in ("is_suspended", "is_st", "is_limit_up", "is_limit_down"):
        if col not in panel.columns:
            panel[col] = False

`False` on these columns is a positive claim -- definitely not suspended,
definitely not limit-locked, therefore freely tradeable at any size. A panel that
never carried the flags backtested as though every name traded freely every day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.market_rules.tradability_flags import (
    TRADABILITY_FLAG_COLUMNS,
    TradabilityEvidenceMissing,
    ensure_tradability_flags,
    tradability_evidence_note,
)


def _panel(**cols) -> pd.DataFrame:
    base = {
        "symbol": ["600000.SH", "600000.SH"],
        "trade_date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
        "close": [10.0, 10.5],
    }
    base.update(cols)
    return pd.DataFrame(base)


class TestUnverifiedIsReported:
    def test_all_flags_absent_are_reported(self):
        _, unverified = ensure_tradability_flags(_panel())
        assert set(unverified) == set(TRADABILITY_FLAG_COLUMNS)

    def test_fully_measured_panel_reports_nothing(self):
        panel = _panel(**{c: [False, False] for c in TRADABILITY_FLAG_COLUMNS})
        _, unverified = ensure_tradability_flags(panel)
        assert unverified == ()

    def test_nan_cell_counts_as_unverified(self):
        """A NaN state is unknown, not a measured 'no'."""
        cols = {c: [False, False] for c in TRADABILITY_FLAG_COLUMNS}
        cols["is_suspended"] = [False, np.nan]
        _, unverified = ensure_tradability_flags(_panel(**cols))
        assert "is_suspended" in unverified

    def test_partially_measured_reports_only_the_missing_ones(self):
        cols = {"is_suspended": [False, False], "is_st": [False, False]}
        _, unverified = ensure_tradability_flags(_panel(**cols))
        assert set(unverified) == {"is_limit_up", "is_limit_down"}


class TestFailClosedMode:
    def test_require_measured_raises_on_absent_flags(self):
        with pytest.raises(TradabilityEvidenceMissing, match="not measured"):
            ensure_tradability_flags(_panel(), require_measured=True)

    def test_require_measured_passes_when_everything_is_present(self):
        panel = _panel(**{c: [False, False] for c in TRADABILITY_FLAG_COLUMNS})
        out, unverified = ensure_tradability_flags(panel, require_measured=True)
        assert unverified == ()
        assert len(out) == 2


class TestOutputContract:
    def test_columns_are_boolean_and_panel_is_not_mutated(self):
        panel = _panel()
        out, _ = ensure_tradability_flags(panel)
        for column in TRADABILITY_FLAG_COLUMNS:
            assert out[column].dtype == bool
            assert column not in panel.columns, "input frame must not be mutated"

    def test_measured_true_values_survive(self):
        cols = {c: [False, False] for c in TRADABILITY_FLAG_COLUMNS}
        cols["is_limit_up"] = [True, False]
        out, _ = ensure_tradability_flags(_panel(**cols))
        assert out["is_limit_up"].tolist() == [True, False]

    def test_evidence_note_distinguishes_measured_from_assumed(self):
        assert tradability_evidence_note(())["tradability_measured"] is True
        note = tradability_evidence_note(("is_suspended",))
        assert note["tradability_measured"] is False
        assert note["tradability_unverified_columns"] == ["is_suspended"]


class TestCallSitesUseTheHelper:
    """The fabrication loop must not reappear by copy-paste."""

    def test_no_module_reimplements_the_flag_loop(self):
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        proc = subprocess.run(
            ["grep", "-rn", "--include=*.py", 'for col in ("is_suspended"', "src/quantagent"],
            cwd=root, capture_output=True, text=True,
        )
        offenders = [
            line for line in proc.stdout.splitlines()
            if "tradability_flags.py" not in line
        ]
        assert not offenders, (
            "these modules reimplement the tradability fabrication loop instead "
            "of calling ensure_tradability_flags:\n" + "\n".join(offenders)
        )
