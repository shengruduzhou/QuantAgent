"""Correctness + wiring inventory for the AFML research helpers.

Two separate jobs in one file, deliberately:

1. These helpers had ZERO callers and ZERO tests, so nothing had ever established
   that they work.  They do; these tests pin that.
2. Because they *look* like state-of-the-art validation machinery while being
   wired to nothing, they are an audit trap: a reader greps for CPCV or
   triple-barrier, finds both, and concludes validation is AFML-grade.  It is
   not -- validation currently runs on single-path walk-forward plus purged
   k-fold.  ``test_unwired_inventory_is_accurate`` makes that status a checked
   fact instead of folklore, so it cannot drift silently in either direction.

If you wire one of these into a pipeline, that test will fail and tell you to
move the name out of UNWIRED.  That failure is the feature.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantagent.quant_math.purged_cv import (
    combinatorial_purged_split,
    probability_of_backtest_overfitting,
    purged_kfold_split,
)
from quantagent.quant_math.triple_barrier import (
    daily_volatility,
    sample_weights_by_uniqueness,
    triple_barrier_labels,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Helpers that exist, are correct, and are called by NOTHING in the pipeline.
UNWIRED = {
    "combinatorial_purged_split",
    "triple_barrier_labels",
    "sample_weights_by_uniqueness",
    "daily_volatility",
}

#: Helpers that are genuinely load-bearing.
WIRED = {
    "purged_kfold_split",
    "probability_of_backtest_overfitting",
}


def _production_callers(symbol: str) -> list[str]:
    """Call sites outside tests/ and outside the symbol's own definition."""
    proc = subprocess.run(
        ["grep", "-rn", symbol, "src", "services", "scripts", "--include=*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    hits = []
    for line in proc.stdout.splitlines():
        if f"def {symbol}" in line:
            continue
        # the re-export/definition modules themselves are not "callers"
        if line.startswith(("src/quantagent/quant_math/purged_cv.py",
                            "src/quantagent/quant_math/triple_barrier.py")):
            continue
        hits.append(line)
    return hits


@pytest.fixture(scope="module")
def close() -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    rng = np.random.default_rng(0)
    return pd.Series(np.exp(np.cumsum(rng.standard_normal(200) * 0.01)), index=idx)


class TestWiringInventory:
    @pytest.mark.parametrize("symbol", sorted(WIRED))
    def test_wired_helpers_have_production_callers(self, symbol):
        assert _production_callers(symbol), (
            f"{symbol} is listed as WIRED but has no production call site"
        )

    @pytest.mark.parametrize("symbol", sorted(UNWIRED))
    def test_unwired_inventory_is_accurate(self, symbol):
        callers = _production_callers(symbol)
        assert not callers, (
            f"{symbol} is listed as UNWIRED but now has production callers:\n"
            + "\n".join(callers)
            + "\n\nThis is good news -- move it from UNWIRED to WIRED in this file, "
            "and make sure the pipeline that adopted it is covered by its own tests."
        )


class TestCombinatorialPurgedSplit:
    def test_generates_exactly_n_choose_k_paths(self, close):
        times = pd.Series(close.index, index=close.index)
        label_end = pd.Series(close.index.shift(5, freq="B"), index=close.index)
        splits = list(
            combinatorial_purged_split(times, label_end, n_splits=6, n_test_groups=2)
        )
        assert len(splits) == math.comb(6, 2) == 15

    def test_train_and_test_never_overlap(self, close):
        times = pd.Series(close.index, index=close.index)
        label_end = pd.Series(close.index.shift(5, freq="B"), index=close.index)
        for train, test in combinatorial_purged_split(
            times, label_end, n_splits=6, n_test_groups=2
        ):
            assert not set(train) & set(test)

    def test_purging_actually_removes_rows(self, close):
        """A label that ends inside the test window must not remain in train."""
        times = pd.Series(close.index, index=close.index)
        no_overlap = pd.Series(close.index, index=close.index)
        long_overlap = pd.Series(close.index.shift(20, freq="B"), index=close.index)

        def total_train(label_end):
            return sum(
                len(train)
                for train, _ in combinatorial_purged_split(
                    times, label_end, n_splits=6, n_test_groups=2, embargo_pct=0.0
                )
            )

        assert total_train(long_overlap) < total_train(no_overlap), (
            "longer labels must purge more training rows"
        )

    def test_embargo_removes_additional_rows(self, close):
        times = pd.Series(close.index, index=close.index)
        label_end = pd.Series(close.index, index=close.index)

        def total_train(embargo):
            return sum(
                len(train)
                for train, _ in combinatorial_purged_split(
                    times, label_end, n_splits=6, n_test_groups=2, embargo_pct=embargo
                )
            )

        assert total_train(0.10) < total_train(0.0)


class TestTripleBarrier:
    def test_labels_have_expected_shape_and_columns(self, close):
        sigma = daily_volatility(close, span=20).fillna(0.01)
        labels = triple_barrier_labels(close=close, sigma=sigma)
        assert list(labels.columns) == ["t1", "ret", "label", "barrier"]
        assert len(labels) == len(close)

    def test_labels_are_in_the_ternary_set(self, close):
        sigma = daily_volatility(close, span=20).fillna(0.01)
        labels = triple_barrier_labels(close=close, sigma=sigma)
        assert set(labels["label"].dropna().unique()) <= {-1, 0, 1}

    def test_barrier_reason_is_consistent_with_the_sign_of_the_return(self, close):
        sigma = daily_volatility(close, span=20).fillna(0.01)
        labels = triple_barrier_labels(close=close, sigma=sigma).dropna(subset=["ret"])
        pt = labels[labels["barrier"] == "pt"]
        sl = labels[labels["barrier"] == "sl"]
        # A profit-taking hit cannot have produced a negative return, and vice versa.
        assert (pt["ret"] >= 0).all()
        assert (sl["ret"] <= 0).all()

    def test_a_flat_series_never_touches_a_horizontal_barrier(self):
        idx = pd.date_range("2024-01-01", periods=60, freq="B")
        flat = pd.Series(100.0, index=idx)
        sigma = pd.Series(0.02, index=idx)
        labels = triple_barrier_labels(close=flat, sigma=sigma)
        # "vt" = vertical (time) barrier, i.e. the holding period expired without
        # either the profit-taking or stop-loss level being touched.
        assert (labels["barrier"].dropna() == "vt").all()
        assert (labels["label"].dropna() == 0).all()


class TestUniquenessWeights:
    def test_overlapping_labels_get_less_weight_than_disjoint_ones(self, close):
        """The whole point: an overlapping sample is worth less than a unique one."""
        idx = close.index[:50]
        overlapping = pd.DataFrame({"t1": idx.shift(10, freq="B")}, index=idx)
        disjoint = pd.DataFrame({"t1": idx.shift(1, freq="B")}, index=idx)

        w_overlap = sample_weights_by_uniqueness(overlapping, close.index)
        w_disjoint = sample_weights_by_uniqueness(disjoint, close.index)

        assert w_overlap.mean() < w_disjoint.mean(), (
            "10-day overlapping labels should carry lower uniqueness weight "
            "than 1-day labels"
        )

    def test_weights_are_positive_and_finite(self, close):
        idx = close.index[:50]
        events = pd.DataFrame({"t1": idx.shift(5, freq="B")}, index=idx)
        weights = sample_weights_by_uniqueness(events, close.index)
        assert np.isfinite(weights).all()
        assert (weights > 0).all()
