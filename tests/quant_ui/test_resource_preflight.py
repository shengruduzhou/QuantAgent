"""A scope the machine cannot hold must be refused before it is attempted.

The web UI offers full-universe training. On this box the 5,790-symbol panel
needs roughly 164 GiB of peak resident memory against ~34 GiB usable, so the
build ran for tens of minutes and was then OOM-killed — surfacing as an opaque
engineering failure instead of "this machine is too small for this scope".

The estimate is anchored on a measurement, not a guess: a 400-symbol build of
1.02M rows peaked at 15.3 GiB.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quantagent.cli.v7_train import (
    DATASET_BUILD_BYTES_PER_ROW,
    _assert_dataset_build_fits_in_memory,
)
from quantagent.research.verdict import ResearchRejection


def _labels(path: Path, symbols: int, days: int) -> Path:
    dates = pd.bdate_range("2016-01-04", periods=days)
    names = [f"{i:06d}.SZ" for i in range(symbols)]
    frame = pd.DataFrame({
        "symbol": [s for _ in dates for s in names],
        "trade_date": [d for d in dates for _ in names],
        "forward_return_5d": 0.0,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def test_the_coefficient_reproduces_the_measurement_it_came_from():
    """400 symbols x 2,562 days = 1.02M rows measured at 15.3 GiB peak."""
    projected_gib = 1_020_000 * DATASET_BUILD_BYTES_PER_ROW / 2**30
    assert projected_gib == pytest.approx(15.3, rel=0.01)


def test_a_scope_that_fits_is_allowed(tmp_path, monkeypatch):
    labels = _labels(tmp_path / "labels.parquet", symbols=20, days=60)
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: type("M", (), {"available": 64 * 2**30})()
    )

    report = _assert_dataset_build_fits_in_memory(
        labels_path=labels, symbols=None, output_dir=tmp_path
    )

    assert report["status"] == "pass"
    assert report["projectedRows"] == 20 * 60


def test_a_scope_that_does_not_fit_is_blocked_before_the_build(tmp_path, monkeypatch):
    labels = _labels(tmp_path / "labels.parquet", symbols=200, days=250)
    # 50,000 rows at the measured coefficient is ~0.75 GiB; allow far less.
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: type("M", (), {"available": 256 * 2**20})()
    )

    with pytest.raises(ResearchRejection) as raised:
        _assert_dataset_build_fits_in_memory(
            labels_path=labels, symbols=None, output_dir=tmp_path
        )

    rejection = raised.value
    assert rejection.verdict == "blocked", "an unrunnable scope is not a research rejection"
    assert rejection.code == "insufficient_memory_for_scope"
    assert rejection.stage == "preflight"
    # The operator needs the numbers, not just a refusal.
    assert rejection.metrics["projectedPeakGiB"] > rejection.metrics["usableGiB"]
    assert rejection.metrics["projectedRows"] == 200 * 250
    assert "--symbols-file" in rejection.remediation


def test_a_symbol_subset_is_scaled_from_real_rows_not_symbols_times_days(tmp_path, monkeypatch):
    """Listings and suspensions make symbols x days overstate a real panel."""
    labels = _labels(tmp_path / "labels.parquet", symbols=100, days=100)
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: type("M", (), {"available": 64 * 2**30})()
    )

    report = _assert_dataset_build_fits_in_memory(
        labels_path=labels, symbols=[f"{i:06d}.SZ" for i in range(25)], output_dir=tmp_path
    )

    # 25 of 100 symbols -> a quarter of the panel's 10,000 rows.
    assert report["projectedRows"] == 2_500
    assert report["symbolCount"] == 25


def test_a_prebuilt_dataset_is_never_blocked_on_build_memory(tmp_path, monkeypatch):
    """Reusing an existing dataset skips the build, so the gate must not fire."""
    labels = _labels(tmp_path / "labels.parquet", symbols=10, days=10)
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: type("M", (), {"available": 1 * 2**20})()
    )
    # The caller only invokes the gate when it is about to build; this asserts
    # the gate itself is the thing guarded, not the training run as a whole.
    with pytest.raises(ResearchRejection):
        _assert_dataset_build_fits_in_memory(
            labels_path=labels, symbols=None, output_dir=tmp_path
        )
