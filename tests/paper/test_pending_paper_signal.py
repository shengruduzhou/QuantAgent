from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
        source_lineage={
            "model_dir": "models/v7",
            "target_weights_path": "weights.parquet",
        },
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


def test_concurrent_identical_writers_converge_on_one_artifact(tmp_path) -> None:
    root = tmp_path / "pending"

    def write(index: int):
        store = PendingPaperSignalStore(root)
        signal, path = store.record(
            signal_date="2026-08-07",
            target_weights=_weights(0.5),
            source_lineage={"model": "v1"},
            created_at=f"2026-08-07T07:00:{index:02d}+00:00",
        )
        return signal.payload_sha256, path.read_bytes()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(write, range(16)))

    assert len({payload_sha for payload_sha, _ in results}) == 1
    assert len({data for _, data in results}) == 1
    persisted = PendingPaperSignalStore(root).read("2026-08-07")
    assert persisted is not None
    assert persisted.target_weights["600000.SH"] == pytest.approx(0.5)


def test_concurrent_conflicting_writers_never_last_writer_win(tmp_path) -> None:
    root = tmp_path / "pending"

    def write(value: float):
        store = PendingPaperSignalStore(root)
        try:
            signal, _ = store.record(
                signal_date="2026-08-07",
                target_weights=_weights(value),
                source_lineage={"model": "v1"},
            )
            return ("success", signal.target_weights["600000.SH"], signal.payload_sha256)
        except PendingSignalConflict:
            return ("conflict", value, "")

    values = [0.4, 0.6] * 8
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(write, values))

    successes = [item for item in outcomes if item[0] == "success"]
    conflicts = [item for item in outcomes if item[0] == "conflict"]
    assert successes
    assert conflicts
    assert len({round(float(item[1]), 12) for item in successes}) == 1
    assert len({item[2] for item in successes}) == 1

    persisted = PendingPaperSignalStore(root).read("2026-08-07")
    assert persisted is not None
    winning_weight = float(successes[0][1])
    assert persisted.target_weights["600000.SH"] == pytest.approx(winning_weight)


def test_duplicate_json_keys_are_rejected_as_corrupt_evidence(tmp_path) -> None:
    store = PendingPaperSignalStore(tmp_path / "pending")
    path = store.path_for("2026-08-07")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"schema_version":"paper_pending_signal_v1",'
        '"schema_version":"paper_pending_signal_v1"}\n',
        encoding="utf-8",
    )
    with pytest.raises(PendingSignalCorruption, match="duplicate JSON key"):
        store.read("2026-08-07")
