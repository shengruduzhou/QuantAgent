"""Numerical-equivalence guard for the vectorized Alpha101 fast paths.

``compute_alpha101`` is shared by training and forward inference, so the
vectorized per-symbol helpers (``ts_rank``, ``argmax/min``, ``product``,
``decay_linear``, rolling ``corr/cov``) must stay numerically identical to the
original ``groupby().rolling().apply(python_fn)`` reference implementations.

The reference implementations are preserved in ``alpha101`` behind the
``_REFERENCE_HELPERS`` switch; these tests flip that switch to compute the two
in-process and assert the outputs match (bit-for-bit in practice, ``<1e-9`` by
contract) including the exact NaN pattern. The synthetic panel deliberately
exercises the tricky paths: windows up to 250, constant-price stretches that
drive rolling ``corr`` to ``±inf`` (pandas treats those windows like NaN),
zero-volume rows (vwap fallback), and value ties (rank averaging).
"""

from __future__ import annotations

import multiprocessing

import numpy as np
import pandas as pd
import pytest

import quantagent.factors.alpha101 as alpha101
from quantagent.factors.alpha101 import compute_alpha101

TOL = 1e-9


def _panel() -> pd.DataFrame:
    """Synthetic multi-symbol OHLCV panel with edge cases for every helper."""
    rng = np.random.default_rng(20260614)
    dates = pd.date_range("2024-01-01", periods=320, freq="B")
    symbols = [f"S{i:02d}" for i in range(14)]
    rows = []
    for j, symbol in enumerate(symbols):
        close = 20.0 + np.cumsum(rng.normal(0.01, 0.4, len(dates))) + j
        close = np.clip(close, 1.0, None)
        volume = np.clip(1_000_000 + rng.normal(0, 50_000, len(dates)).cumsum() + j * 5_000, 1000, None)
        # Constant-price / constant-volume stretch -> degenerate rolling std ->
        # rolling corr produces inf/nan (exercises the ~isfinite window mask).
        if j % 5 == 0:
            close[40:75] = close[40]
            volume[40:75] = volume[40]
        # Value ties (rank averaging) + a couple of zero-volume rows (vwap fallback).
        close[120:126] = close[120]
        if j % 4 == 1:
            volume[200] = 0.0
            volume[201] = 0.0
        for i, date in enumerate(dates):
            c = float(close[i])
            v = float(volume[i])
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": c * 0.99,
                    "high": c * 1.015,
                    "low": c * 0.985,
                    "close": c,
                    "volume": v,
                    "amount": v * c,
                }
            )
    return pd.DataFrame(rows)


def _reference_wide(frame: pd.DataFrame) -> pd.DataFrame:
    alpha101._REFERENCE_HELPERS = True
    try:
        return compute_alpha101(frame, wide=True)
    finally:
        alpha101._REFERENCE_HELPERS = False


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
    """Return (max abs diff over jointly-finite cells, NaN-pattern-equal)."""
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    nan_equal = np.array_equal(nan_a, nan_b)
    mask = ~(nan_a | nan_b)
    diff = a[mask] - b[mask]
    diff = diff[np.isfinite(diff)]  # ignore inf-inf (==) cells
    return (float(np.abs(diff).max()) if diff.size else 0.0), nan_equal


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel()


def test_panel_actually_exercises_inf_corr_path(panel: pd.DataFrame):
    """Sanity: the synthetic panel must drive at least one rolling corr to inf,
    otherwise the ~isfinite window mask would go untested."""
    data, _, _ = alpha101._prepare_alpha_context(panel)
    corr = alpha101._corr(data, data["high"], data["volume"], 10)
    assert np.isinf(corr.to_numpy()).any(), "panel did not produce any inf corr"


def test_fast_matches_reference_wide(panel: pd.DataFrame):
    ref = _reference_wide(panel)
    fast = compute_alpha101(panel, wide=True)

    assert list(ref.columns) == list(fast.columns)
    assert ref[["trade_date", "symbol"]].equals(fast[["trade_date", "symbol"]])

    alpha_cols = [c for c in ref.columns if c.startswith("alpha")]
    assert len(alpha_cols) == 101
    worst = 0.0
    failures = []
    for col in alpha_cols:
        d, nan_equal = _max_abs_diff(ref[col].to_numpy(), fast[col].to_numpy())
        worst = max(worst, d)
        if d >= TOL or not nan_equal:
            failures.append((col, d, nan_equal))
    assert not failures, f"factors diverged: {failures} (global max {worst:.3e})"


def test_fast_matches_reference_long(panel: pd.DataFrame):
    alpha101._REFERENCE_HELPERS = True
    try:
        ref = compute_alpha101(panel)
    finally:
        alpha101._REFERENCE_HELPERS = False
    fast = compute_alpha101(panel)

    assert ref[["trade_date", "symbol", "factor_name"]].equals(
        fast[["trade_date", "symbol", "factor_name"]]
    )
    d, nan_equal = _max_abs_diff(
        ref["factor_value"].to_numpy(), fast["factor_value"].to_numpy()
    )
    assert nan_equal, "long-form NaN pattern diverged"
    assert d < TOL, f"long-form max abs diff {d:.3e} >= {TOL}"


def test_some_factors_are_non_trivial(panel: pd.DataFrame):
    """Guard against a degenerate all-NaN comparison passing vacuously."""
    fast = compute_alpha101(panel, wide=True)
    alpha_cols = [c for c in fast.columns if c.startswith("alpha")]
    finite_counts = {c: int(np.isfinite(fast[c].to_numpy()).sum()) for c in alpha_cols}
    # The long-window placeholders aside, most factors must carry real values.
    non_trivial = sum(1 for n in finite_counts.values() if n > 100)
    assert non_trivial >= 70, f"only {non_trivial} factors had >100 finite values"


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="factor-parallel path requires the POSIX fork start method",
)
def test_parallel_matches_serial(panel: pd.DataFrame):
    serial = compute_alpha101(panel, wide=True)
    parallel = compute_alpha101(panel, wide=True, workers=3)

    assert list(serial.columns) == list(parallel.columns)
    assert serial[["trade_date", "symbol"]].equals(parallel[["trade_date", "symbol"]])
    for col in (c for c in serial.columns if c.startswith("alpha")):
        a, b = serial[col].to_numpy(), parallel[col].to_numpy()
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{col} NaN pattern differs"
        mask = ~(np.isnan(a) | np.isnan(b))
        diff = a[mask] - b[mask]
        diff = diff[np.isfinite(diff)]
        assert (np.abs(diff).max() if diff.size else 0.0) == 0.0, f"{col} differs"


def test_workers_default_is_serial_behavior(panel: pd.DataFrame):
    """workers=1 (default) must not spawn processes and must match the subset API."""
    names = ["alpha003", "alpha050", "alpha071", "alpha088", "alpha096"]
    a = compute_alpha101(panel, names=names, wide=True)
    b = compute_alpha101(panel, names=names, wide=True, workers=1)
    assert a.equals(b)
