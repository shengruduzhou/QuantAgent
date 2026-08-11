from __future__ import annotations

import pandas as pd
import pytest

from scripts.rl_pit_train_eval import (
    _policy_contract,
    _require_matching_policy_contract,
    _training_input_fingerprint,
    _write_json,
)


def _inputs():
    dates = pd.date_range("2025-08-25", periods=3, freq="B")
    book = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
    predictions = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["A", "A", "A"],
            "alpha_score": [0.1, 0.2, 0.3],
        }
    )
    panel = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["A", "A", "A"],
            "close": [10.0, 10.1, 10.2],
            "is_limit_up": [False, False, False],
            "is_limit_down": [False, False, False],
            "is_suspended": [False, False, False],
        }
    )
    gaps = pd.DataFrame(columns=["trade_date", "symbol", "classification"])
    return book, predictions, panel, gaps


def test_training_input_fingerprint_changes_when_alpha_content_changes() -> None:
    book, predictions, panel, gaps = _inputs()
    first, first_parts = _training_input_fingerprint(
        book=book,
        predictions=predictions,
        panel=panel,
        session_gaps=gaps,
        train_end="2025-08-29",
    )
    changed = predictions.copy()
    changed.loc[1, "alpha_score"] = 99.0
    second, second_parts = _training_input_fingerprint(
        book=book,
        predictions=changed,
        panel=panel,
        session_gaps=gaps,
        train_end="2025-08-29",
    )

    assert first != second
    assert first_parts["predictions"] != second_parts["predictions"]
    assert first_parts["book"] == second_parts["book"]


def test_skip_train_contract_rejects_different_training_fingerprint(tmp_path) -> None:
    common = dict(
        reward_clock_semantics="clock-v1",
        target_index_semantics="signal",
        max_book=80,
        train_start="2025-01-01",
        train_end="2025-08-29",
        train_reward_end="2025-08-29",
        test_start="2026-01-02",
        warmup_start="2024-08-09",
        score_column="composite_score",
        predictions_path="preds.parquet",
        panel_path="panel.parquet",
        session_gaps_path="session_gaps.parquet",
        training_component_sha256={"book": "a", "predictions": "b", "panel": "c", "session_gaps": "d"},
        cost_bps=12.0,
        seed=1729,
    )
    old = _policy_contract(training_input_sha256="old", **common)
    new = _policy_contract(training_input_sha256="new", **common)
    path = tmp_path / "policy_contract.json"
    _write_json(path, old)

    with pytest.raises(RuntimeError, match="training inputs"):
        _require_matching_policy_contract(path, new)
