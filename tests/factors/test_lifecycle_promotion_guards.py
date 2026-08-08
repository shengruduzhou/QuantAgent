from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.factors.lifecycle import (
    LifecycleThresholds,
    build_factor_lifecycle_report,
    recommend_factor_status,
)


def test_highly_redundant_factor_cannot_be_active() -> None:
    status = recommend_factor_status(
        rank_icir=0.8,
        positive_ratio=0.8,
        monotonicity=0.9,
        live_drift=0.2,
        max_existing_correlation=0.97,
    )
    assert status == "watch"


def test_unknown_drift_is_not_treated_as_zero_drift() -> None:
    status = recommend_factor_status(
        rank_icir=0.8,
        positive_ratio=0.8,
        monotonicity=0.9,
        live_drift=np.nan,
        max_existing_correlation=0.1,
    )
    assert status == "watch"


def test_optional_capacity_gate_blocks_when_enabled_and_missing() -> None:
    status = recommend_factor_status(
        rank_icir=0.8,
        positive_ratio=0.8,
        monotonicity=0.9,
        live_drift=0.2,
        max_existing_correlation=0.1,
        capacity_rmb=np.nan,
        thresholds=LifecycleThresholds(min_capacity_rmb_for_active=1_000_000.0),
    )
    assert status != "active"


def test_short_history_lifecycle_report_stays_watch() -> None:
    rows: list[dict[str, object]] = []
    for day in pd.date_range("2026-01-01", periods=3, freq="D"):
        for idx in range(8):
            rows.append(
                {
                    "trade_date": day,
                    "symbol": f"S{idx:02d}",
                    "factor": float(idx),
                    "ret": float(idx) / 100.0,
                    "amount": 10_000_000.0,
                }
            )
    report = build_factor_lifecycle_report(
        pd.DataFrame(rows),
        "factor",
        "ret",
        amount_column="amount",
    )
    assert np.isnan(report.live_drift)
    assert report.recommended_status == "watch"


def test_duplicate_factor_is_detected_from_frame() -> None:
    rows: list[dict[str, object]] = []
    for day_idx, day in enumerate(pd.date_range("2026-01-01", periods=10, freq="D")):
        for idx in range(8):
            signal = float(idx + day_idx * 0.01)
            rows.append(
                {
                    "trade_date": day,
                    "symbol": f"S{idx:02d}",
                    "factor": signal,
                    "existing": signal * 2.0,
                    "ret": float(idx) / 100.0,
                    "amount": 10_000_000.0,
                }
            )
    report = build_factor_lifecycle_report(
        pd.DataFrame(rows),
        "factor",
        "ret",
        existing_factor_columns=["existing"],
    )
    assert report.max_correlation_to_existing > 0.99
    assert report.recommended_status == "watch"
