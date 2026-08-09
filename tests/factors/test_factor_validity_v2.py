from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.backtest.execution_timing import EXECUTION_TIMING_SEMANTICS
from quantagent.factors.governance_metrics import (
    FactorGateConfig,
    FactorPromotionContext,
    _ic_ir,
    evaluate_factor_candidate,
)


def _frame(days: int = 60, symbols: int = 20, *, deteriorate_recent: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2025-01-02", periods=days)
    for day_idx, date in enumerate(dates):
        recent = deteriorate_recent and day_idx >= int(days * 0.75)
        for symbol_idx in range(symbols):
            factor = float(symbol_idx)
            noise = np.sin(symbol_idx * 1.7 + day_idx * 0.31) * 3.0
            target = (-factor if recent else factor) + noise
            rows.append(
                {
                    "trade_date": date,
                    "symbol": f"S{symbol_idx:03d}",
                    "factor": factor,
                    "ret_1d": target / 100.0,
                    "ret_5d": (target + np.cos(symbol_idx + day_idx) * 0.5) / 100.0,
                    "adv20_cny": 100_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def _relaxed_config(**kwargs) -> FactorGateConfig:
    base = dict(
        min_dates=30,
        min_symbols_per_date=10,
        min_mean_rank_ic=0.05,
        min_ic_information_ratio=0.05,
        min_newey_west_rank_t_stat=0.0,
        min_positive_ic_ratio=0.50,
        max_losing_period_rate=1.0,
        max_recent_predictive_drift_z=50.0,
        max_library_abs_correlation=0.99,
        min_decay_retention=-10.0,
        max_decay_reversal=-10.0,
        target_book_cny=1_000_000.0,
        max_adv_participation=0.10,
        min_capacity_multiple=0.01,
        min_shadow_days=20,
    )
    base.update(kwargs)
    return FactorGateConfig(**base)


def _promotion(**kwargs) -> FactorPromotionContext:
    base = dict(
        label_semantics=EXECUTION_TIMING_SEMANTICS,
        preregistered=True,
        oos_only=True,
        cumulative_trials=100,
        multiple_testing_passed=True,
        pbo=0.10,
        dsr_probability=0.98,
        spa_pvalue=0.01,
        strict_long_only_backtest_passed=True,
        shadow_days=30,
        shadow_passed=True,
        pit_data_certified=True,
        evidence_digest="evidence-sha256",
    )
    base.update(kwargs)
    return FactorPromotionContext(**base)


def test_icir_is_mean_over_std_not_t_statistic_in_disguise() -> None:
    ic = pd.Series([0.00, 0.01, 0.02])
    expected = float(ic.mean() / ic.std(ddof=1))
    assert _ic_ir(ic) == pytest.approx(expected)
    assert _ic_ir(ic) != pytest.approx(expected * np.sqrt(len(ic)))


def test_valid_factor_still_cannot_be_promotion_candidate_without_context() -> None:
    report = evaluate_factor_candidate(
        _frame(),
        factor_name="factor",
        target_return_col="ret_1d",
        target_horizon_days=1,
        decay_return_columns={1: "ret_1d", 5: "ret_5d"},
        label_semantics=EXECUTION_TIMING_SEMANTICS,
        config=_relaxed_config(),
    )
    assert report.passed is True
    assert report.promotion_candidate_ready is False
    assert "promotion_context_missing" in report.promotion_blockers
    assert report.to_dict()["activation_authorized"] is False


def test_complete_context_can_only_make_core_valid_factor_a_promotion_candidate() -> None:
    report = evaluate_factor_candidate(
        _frame(),
        factor_name="factor",
        target_return_col="ret_1d",
        target_horizon_days=1,
        decay_return_columns={1: "ret_1d", 5: "ret_5d"},
        label_semantics=EXECUTION_TIMING_SEMANTICS,
        promotion_context=_promotion(),
        config=_relaxed_config(),
    )
    assert report.passed is True, report.rejection_reasons
    assert report.promotion_candidate_ready is True, report.promotion_blockers
    payload = report.to_dict()
    assert payload["activation_authorized"] is False
    assert payload["activation_authority"] == "hash_bound_factor_promotion_certificate_required"
    assert report.cumulative_trials == 100


def test_wrong_execution_label_semantics_blocks_promotion_candidacy_even_if_metrics_are_good() -> None:
    report = evaluate_factor_candidate(
        _frame(),
        factor_name="factor",
        target_return_col="ret_1d",
        target_horizon_days=1,
        decay_return_columns={1: "ret_1d", 5: "ret_5d"},
        label_semantics="same_close_t_to_t_plus_h",
        promotion_context=_promotion(label_semantics="same_close_t_to_t_plus_h"),
        config=_relaxed_config(),
    )
    assert report.promotion_candidate_ready is False
    assert any("label_semantics" in reason for reason in report.promotion_blockers)


def test_multiple_testing_failure_blocks_promotion_candidacy() -> None:
    report = evaluate_factor_candidate(
        _frame(),
        factor_name="factor",
        target_return_col="ret_1d",
        target_horizon_days=1,
        decay_return_columns={1: "ret_1d", 5: "ret_5d"},
        label_semantics=EXECUTION_TIMING_SEMANTICS,
        promotion_context=_promotion(
            multiple_testing_passed=False,
            pbo=0.40,
            dsr_probability=0.80,
            spa_pvalue=0.20,
        ),
        config=_relaxed_config(),
    )
    assert report.promotion_candidate_ready is False
    joined = "|".join(report.promotion_blockers)
    assert "multiple_testing_gate_not_passed" in joined
    assert "pbo=" in joined
    assert "dsr_probability=" in joined
    assert "spa_pvalue=" in joined


def test_recent_predictive_collapse_is_a_core_validity_failure() -> None:
    report = evaluate_factor_candidate(
        _frame(days=120, symbols=20, deteriorate_recent=True),
        factor_name="factor",
        target_return_col="ret_1d",
        target_horizon_days=1,
        decay_return_columns={1: "ret_1d", 5: "ret_5d"},
        label_semantics=EXECUTION_TIMING_SEMANTICS,
        promotion_context=_promotion(),
        config=_relaxed_config(
            min_dates=100,
            max_recent_predictive_drift_z=2.0,
            min_mean_rank_ic=-1.0,
            min_ic_information_ratio=-10.0,
            min_newey_west_rank_t_stat=-100.0,
            min_positive_ic_ratio=0.0,
        ),
    )
    assert report.recent_predictive_drift_z < -2.0
    assert any("recent_predictive_drift_z" in reason for reason in report.rejection_reasons)
    assert report.promotion_candidate_ready is False
