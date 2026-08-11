from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.factors.lifecycle import build_factor_lifecycle_report


def _negative_alpha_panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day_idx, day in enumerate(pd.date_range("2025-01-02", periods=80, freq="B")):
        for idx in range(30):
            factor = float(idx) + 0.01 * np.sin(day_idx + idx)
            ret = -0.002 * factor + 0.0001 * np.cos(day_idx * 0.7 + idx * 0.3)
            amount = 100_000_000.0 if idx < 6 else 1_000_000.0
            rows.append(
                {
                    "trade_date": day,
                    "symbol": f"S{idx:03d}",
                    "factor": factor,
                    "ret": ret,
                    "amount": amount,
                }
            )
    return pd.DataFrame(rows)


def test_negative_direction_flips_all_sign_sensitive_research_metrics() -> None:
    frame = _negative_alpha_panel()

    wrong = build_factor_lifecycle_report(frame, "factor", "ret")
    aligned = build_factor_lifecycle_report(
        frame,
        "factor",
        "ret",
        expected_direction="negative",
    )

    assert wrong.expected_direction == "positive"
    assert aligned.expected_direction == "negative"
    assert wrong.rolling_rank_ic < -0.9
    assert aligned.rolling_rank_ic > 0.9
    assert wrong.positive_ic_ratio < 0.1
    assert aligned.positive_ic_ratio > 0.9
    assert wrong.monotonicity < -0.9
    assert aligned.monotonicity > 0.9


def test_negative_direction_uses_profitable_low_raw_tail_for_capacity() -> None:
    frame = _negative_alpha_panel()

    positive = build_factor_lifecycle_report(frame, "factor", "ret")
    negative = build_factor_lifecycle_report(
        frame,
        "factor",
        "ret",
        expected_direction="negative",
    )

    assert negative.capacity_proxy > positive.capacity_proxy * 20.0


def test_turnover_is_tail_consistent_after_direction_alignment() -> None:
    frame = _negative_alpha_panel()
    flipped = frame.copy()
    flipped["factor"] = -flipped["factor"]

    negative = build_factor_lifecycle_report(
        frame,
        "factor",
        "ret",
        expected_direction="negative",
    )
    equivalent_positive = build_factor_lifecycle_report(
        flipped,
        "factor",
        "ret",
        expected_direction="positive",
    )

    assert negative.turnover == pytest.approx(equivalent_positive.turnover)
    assert negative.capacity_proxy == pytest.approx(equivalent_positive.capacity_proxy)
    assert negative.rolling_rank_ic == pytest.approx(equivalent_positive.rolling_rank_ic)


def test_direction_does_not_change_sign_invariant_crowding_or_drift_guards() -> None:
    frame = _negative_alpha_panel()
    frame["existing"] = frame["factor"] * 0.9 + np.sin(np.arange(len(frame))) * 0.1

    positive = build_factor_lifecycle_report(
        frame,
        "factor",
        "ret",
        existing_factor_columns=["existing"],
        expected_direction="positive",
    )
    negative = build_factor_lifecycle_report(
        frame,
        "factor",
        "ret",
        existing_factor_columns=["existing"],
        expected_direction="negative",
    )

    assert negative.max_correlation_to_existing == pytest.approx(
        positive.max_correlation_to_existing
    )
    assert negative.crowding_proxy == pytest.approx(positive.crowding_proxy)
    assert negative.live_drift == pytest.approx(positive.live_drift)


def test_invalid_direction_fails_closed_instead_of_inferring_from_returns() -> None:
    with pytest.raises(ValueError, match="cannot be inferred"):
        build_factor_lifecycle_report(
            _negative_alpha_panel(),
            "factor",
            "ret",
            expected_direction="auto",  # type: ignore[arg-type]
        )
