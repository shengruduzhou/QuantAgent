from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quantagent.backtest.execution_timing import (
    EXECUTION_TIMING_SEMANTICS,
    execution_trace_sha256,
    validate_execution_trace,
)
from quantagent.backtest.trace_proven_strict_v8 import (
    TRACE_PROVEN_STRICT_SEMANTICS,
    run_trace_proven_strict_backtest_v8,
)


def _market() -> pd.DataFrame:
    rows = []
    for index, date in enumerate(pd.bdate_range("2024-02-01", periods=5)):
        rows.append(
            {
                "trade_date": date,
                "symbol": "600000.SH",
                "close": 10.0 + index,
                "volume": 1_000_000.0,
                "amount": 10_000_000.0,
                "is_suspended": False,
                "is_st": False,
                "is_limit_up": False,
                "is_limit_down": False,
            }
        )
    return pd.DataFrame(rows)


def test_trace_proven_strict_bundle_writes_hash_bound_execution_trace(tmp_path: Path) -> None:
    market = _market()
    sessions = pd.DatetimeIndex(sorted(market["trade_date"].unique()))
    targets = pd.DataFrame({"600000.SH": [0.10, 0.10, 0.0]}, index=sessions[:3])

    artifact = run_trace_proven_strict_backtest_v8(targets, market)
    paths = artifact.write(tmp_path / "strict")

    assert paths["execution_trace"].exists()
    trace = pd.read_csv(paths["execution_trace"])
    validation = validate_execution_trace(trace)
    assert validation.ok, validation.reasons

    digest = execution_trace_sha256(trace)
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["execution_trace_sha256"] == digest
    assert metrics["execution_timing_semantics"] == EXECUTION_TIMING_SEMANTICS
    assert metrics["strict_evidence_semantics"] == TRACE_PROVEN_STRICT_SEMANTICS
    assert artifact.config["execution_trace_sha256"] == digest
    assert artifact.config["execution_timing_semantics"] == EXECUTION_TIMING_SEMANTICS

    schedules = trace[trace["record_type"] == "session_mapping"].copy()
    assert not schedules.empty
    assert (
        pd.to_datetime(schedules["execution_date"])
        > pd.to_datetime(schedules["signal_date"])
    ).all()


def test_trace_artifact_tamper_changes_canonical_digest_and_validation(tmp_path: Path) -> None:
    market = _market()
    sessions = pd.DatetimeIndex(sorted(market["trade_date"].unique()))
    targets = pd.DataFrame({"600000.SH": [0.10]}, index=[sessions[0]])
    artifact = run_trace_proven_strict_backtest_v8(targets, market)
    paths = artifact.write(tmp_path / "strict")

    trace = pd.read_csv(paths["execution_trace"])
    original = execution_trace_sha256(trace)
    schedule_idx = trace.index[trace["record_type"] == "session_mapping"][0]
    trace.loc[schedule_idx, "execution_date"] = trace.loc[schedule_idx, "signal_date"]
    assert execution_trace_sha256(trace) != original
    invalid = validate_execution_trace(trace)
    assert invalid.ok is False
    assert "execution_trace_same_or_prior_session_execution" in invalid.reasons
