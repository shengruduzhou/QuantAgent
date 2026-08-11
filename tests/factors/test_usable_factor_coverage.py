from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.factors.coverage import usable_factor_coverage
from quantagent.factors.governance_metrics import FactorGateConfig, evaluate_factor_candidate
from quantagent.factors.lifecycle import LifecycleThresholds, build_factor_lifecycle_report


def _dense_panel_sparse_factor() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2025-01-02", periods=80)
    for day_idx, date in enumerate(dates):
        for symbol_idx in range(30):
            usable = symbol_idx < 5
            factor = float(symbol_idx) + 0.01 * day_idx if usable else np.nan
            target = 0.001 * symbol_idx + 0.00001 * day_idx
            rows.append(
                {
                    "trade_date": date,
                    "symbol": f"S{symbol_idx:03d}",
                    "factor": factor,
                    "target": target,
                    "target_1d": target,
                    "adv20_cny": 100_000_000.0,
                    "amount": 100_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_usable_coverage_counts_factor_return_pairs_not_parent_panel() -> None:
    frame = _dense_panel_sparse_factor()

    coverage = usable_factor_coverage(
        frame,
        "factor",
        "target",
        min_symbols_per_date=20,
    )

    assert frame.groupby("trade_date")["symbol"].nunique().median() == 30
    assert coverage.median_symbols_per_date == 5.0
    assert coverage.coverage_dates == 0
    assert len(coverage.eligible_dates) == 0


def test_core_factor_gate_rejects_sparse_factor_on_dense_market_panel() -> None:
    frame = _dense_panel_sparse_factor()
    config = FactorGateConfig(
        min_dates=60,
        min_symbols_per_date=20,
        # Relax non-coverage gates so the test isolates coverage truth.
        min_mean_rank_ic=-1.0,
        min_ic_information_ratio=-1.0,
        min_newey_west_rank_t_stat=-1e9,
        min_positive_ic_ratio=0.0,
        max_losing_period_rate=1.0,
        max_recent_predictive_drift_z=1e9,
        max_library_abs_correlation=1.0,
        min_decay_retention=-1e9,
        max_decay_reversal=-1e9,
        target_book_cny=1.0,
        min_capacity_multiple=0.0,
    )

    report = evaluate_factor_candidate(
        frame,
        factor_name="factor",
        target_return_col="target",
        target_horizon_days=1,
        decay_return_columns={1: "target_1d"},
        config=config,
    )

    assert report.coverage_dates == 0
    assert report.median_symbols_per_date == 5.0
    assert report.passed is False
    assert any("coverage_dates=0" in reason for reason in report.rejection_reasons)
    assert any("median_symbols_per_date=5.0" in reason for reason in report.rejection_reasons)


def test_lifecycle_sparse_factor_cannot_borrow_dense_panel_coverage() -> None:
    frame = _dense_panel_sparse_factor()
    thresholds = LifecycleThresholds(
        min_effective_dates=60,
        min_median_symbols_per_date=20,
        min_newey_west_rank_t_stat=-1e9,
    )

    report = build_factor_lifecycle_report(
        frame,
        "factor",
        "target",
        thresholds=thresholds,
    )

    assert report.effective_dates == 0
    assert report.median_symbols_per_date == 5.0
    assert report.recommended_status == "watch"
