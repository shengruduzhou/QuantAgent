from __future__ import annotations

import json

import pandas as pd
import pytest

from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.paper.pending_signal import (
    PendingPaperSignalStore,
    PendingSignalConflict,
    PendingSignalCorruption,
)


def _weights(value: float = 0.5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-08-07")],
            "600000.SH": [value],
            "000001.SZ": [1.0 - value],
        }
    )


def test_pending_signal_is_tamper_evident_and_exact_timing_stamped(tmp_path) -> None:
    store = PendingPaperSignalStore(tmp_path / "pending")
    signal, path = store.record(
        signal_date="2026-08-07",
        target_weights=_weights(),
        source_lineage={"model_dir": "models/v7", "target_weights_path": "weights.parquet"},
        created_at="2026-08-07T07:00:00+00:00",
    )
    assert signal.execution_timing_semantics == EXECUTION_TIMING_SEMANTICS
    assert signal.status == "pending_next_observed_session"
    assert path.exists()
    assert store.read("2026-08-07") == signal

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["target_weights"]["600000.SH"] = 0.9
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PendingSignalCorruption, match="digest mismatch"):
        store.read("2026-08-07")


def test_same_signal_and_lineage_is_idempotent(tmp_path) -> None:
    store = PendingPaperSignalStore(tmp_path / "pending")
    first, first_path = store.record(
        signal_date="2026-08-07",
        target_weights=_weights(),
        source_lineage={"model": "v1"},
        created_at="2026-08-07T07:00:00+00:00",
    )
    second, second_path = store.record(
        signal_date="2026-08-07",
        target_weights=_weights(),
        source_lineage={"model": "v1"},
        created_at="2026-08-07T08:00:00+00:00",
    )
    assert second == first
    assert second_path == first_path


def test_same_signal_date_with_changed_economics_fails_closed(tmp_path) -> None:
    store = PendingPaperSignalStore(tmp_path / "pending")
    store.record(
        signal_date="2026-08-07",
        target_weights=_weights(0.5),
        source_lineage={"model": "v1"},
    )
    with pytest.raises(PendingSignalConflict, match="different economics"):
        store.record(
            signal_date="2026-08-07",
            target_weights=_weights(0.7),
            source_lineage={"model": "v1"},
        )


def test_negative_cash_account_target_is_rejected_before_persistence(tmp_path) -> None:
    store = PendingPaperSignalStore(tmp_path / "pending")
    weights = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-08-07")],
            "600000.SH": [1.1],
            "000001.SZ": [-0.1],
        }
    )
    with pytest.raises(ValueError, match="negative stock weight"):
        store.record(
            signal_date="2026-08-07",
            target_weights=weights,
            source_lineage={"model": "v1"},
        )
    assert not store.path_for("2026-08-07").exists()


def test_empty_target_is_not_reinterpreted_as_all_zero_liquidation(tmp_path) -> None:
    store = PendingPaperSignalStore(tmp_path / "pending")
    with pytest.raises(ValueError, match="target weights are empty"):
        store.record(
            signal_date="2026-08-07",
            target_weights=pd.DataFrame(),
            source_lineage={"model": "v1"},
        )
    assert not store.path_for("2026-08-07").exists()
