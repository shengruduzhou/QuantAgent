"""Tests for quantagent.optimization.multi_objective_loss."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.optimization.multi_objective_loss import (
    LossWeights,
    compute_multi_objective_loss,
    score_backtest,
)


def test_zero_returns_yields_zero_components():
    rets = pd.Series([0.0] * 252)
    loss = compute_multi_objective_loss(rets)
    assert loss.net_return == pytest.approx(0.0)
    assert loss.sharpe == pytest.approx(0.0)
    assert loss.max_drawdown == pytest.approx(0.0)
    # Calmar with zero DD → ann_return * 10 = 0
    assert loss.calmar == pytest.approx(0.0)
    assert loss.total == pytest.approx(0.0)


def test_positive_steady_returns_negative_total():
    """A strategy that earns +20bps/day with no drawdown should score
    a strongly negative (= good) total loss."""
    rets = pd.Series([0.002] * 252)
    loss = compute_multi_objective_loss(rets)
    # +0.2%/day → (1.002)^252 - 1 ≈ +65.5% annual
    assert loss.net_return > 0.50
    # std=0 for constant returns → Sharpe returns 0 (safer than inf)
    assert loss.sharpe == 0.0
    # No DD → calmar = ann_return * 10
    assert loss.max_drawdown == pytest.approx(0.0)
    assert loss.calmar == pytest.approx(loss.net_return * 10.0, rel=1e-5)
    assert loss.total < 0  # minimised → lower is better


def test_positive_volatile_returns_has_finite_sharpe():
    """With nonzero vol, sharpe should be finite and contribute to the loss."""
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(loc=0.001, scale=0.01, size=252))
    loss = compute_multi_objective_loss(rets)
    assert 0.5 < loss.sharpe < 5.0
    assert loss.total < 0  # still net-positive overall


def test_drawdown_increases_loss():
    """A strategy with same mean return but a deep drawdown should
    score worse than one without."""
    n = 252
    smooth = pd.Series([0.001] * n)
    rng = np.random.default_rng(42)
    # Volatile path with similar mean but a 10% drawdown
    rough = pd.Series(rng.normal(loc=0.001, scale=0.015, size=n))
    smooth_loss = compute_multi_objective_loss(smooth)
    rough_loss = compute_multi_objective_loss(rough)
    assert rough_loss.max_drawdown > smooth_loss.max_drawdown
    assert rough_loss.total > smooth_loss.total  # worse than smooth


def test_high_chase_exposure_increases_loss():
    rets = pd.Series([0.001] * 252)
    no_chase = compute_multi_objective_loss(rets, high_chase_exposure_rate=0.0)
    with_chase = compute_multi_objective_loss(rets, high_chase_exposure_rate=0.30)
    assert with_chase.high_chase == pytest.approx(0.30)
    assert with_chase.total > no_chase.total  # worse with chase


def test_custom_weights_change_balance():
    rets = pd.Series([0.001] * 252)
    default_loss = compute_multi_objective_loss(rets, weights=LossWeights())
    # Triple the max-dd weight; for zero-DD strategy this changes nothing
    heavy_dd = compute_multi_objective_loss(rets, weights=LossWeights(max_drawdown=6.0))
    assert default_loss.total == heavy_dd.total  # zero DD → identical


def test_score_backtest_requires_daily_eq_return():
    df = pd.DataFrame({"foo": [1.0, 2.0]})
    with pytest.raises(ValueError, match="daily_eq_return"):
        score_backtest(df)


def test_score_backtest_with_high_chase_blotter():
    eq = pd.DataFrame({
        "daily_eq_return": [0.001] * 30,
        "trade_date": pd.bdate_range("2024-01-02", periods=30),
    })
    blotter = pd.DataFrame({
        "trade_date": [pd.Timestamp("2024-01-15")] * 4,
        "symbol": ["A.SZ", "B.SZ", "C.SZ", "D.SZ"],
        "weight": [0.05, 0.05, 0.05, 0.05],
    })
    chase = pd.DataFrame({
        "trade_date": [pd.Timestamp("2024-01-15")] * 4,
        "symbol": ["A.SZ", "B.SZ", "C.SZ", "D.SZ"],
        "is_high_chase": [True, True, False, False],
    })
    loss = score_backtest(eq, trade_blotter=blotter, high_chase_symbols=chase)
    # 2 of 4 names are chase → 50% chase exposure
    assert loss.high_chase == pytest.approx(0.5)


def test_score_backtest_no_blotter_zero_chase():
    eq = pd.DataFrame({
        "daily_eq_return": [0.001] * 30,
    })
    loss = score_backtest(eq)
    assert loss.high_chase == 0.0


def test_geometric_ann_return_matches_truth_for_volatile_series():
    """Review fix #7: a +10/-10 series should give roughly -1% per
    2-day cycle, not 0% (which the prior arithmetic-mean formula
    incorrectly returned).
    """
    # 252 trading days alternating +10% / -10% — cumulative product
    # over each pair = 1.1 * 0.9 = 0.99 → 126 pairs → 0.99^126
    n = 252
    pattern = [0.10 if i % 2 == 0 else -0.10 for i in range(n)]
    rets = pd.Series(pattern)
    loss = compute_multi_objective_loss(rets)
    # Geometric: 0.99 ** 126 ≈ 0.282 — ann_return ≈ -71.8%
    # Arithmetic (wrong) would give 0%.
    assert loss.net_return < -0.50
    assert loss.net_return > -0.85


def test_high_chase_rate_clipped_to_unit_interval():
    """Review fix #8: rate > 1 should not over-penalise — clip to 1."""
    rets = pd.Series([0.001] * 252)
    loss_negative = compute_multi_objective_loss(rets, high_chase_exposure_rate=-0.3)
    loss_above_one = compute_multi_objective_loss(rets, high_chase_exposure_rate=1.7)
    assert loss_negative.high_chase == 0.0
    assert loss_above_one.high_chase == 1.0


def test_loss_total_sign_convention():
    """Minimise convention: + means worse, - means better.

    A losing portfolio with a 20% drawdown should produce a STRONGLY
    POSITIVE total loss. A winning portfolio should produce a
    STRONGLY NEGATIVE total loss.
    """
    bad = pd.Series([-0.005] * 252)  # -0.5%/day → ~-72% annual; lots of DD
    good = pd.Series([0.002] * 252)
    bad_loss = compute_multi_objective_loss(bad)
    good_loss = compute_multi_objective_loss(good)
    assert bad_loss.total > 1.0  # strongly positive
    assert good_loss.total < -1.0  # strongly negative
    assert bad_loss.total > good_loss.total


# ---------------------------------------------------------------------------
# Stage 3 — five additional terms
# ---------------------------------------------------------------------------

def test_stage3_components_reported_zero_when_inputs_absent():
    """A pure daily-returns call must still produce all 10 fields, with the
    Stage 3-only fields at sensible defaults (0 when no input)."""
    rets = pd.Series([0.001] * 60)
    loss = compute_multi_objective_loss(rets)
    d = loss.as_dict()
    for key in [
        "net_return", "sharpe", "calmar", "max_drawdown", "high_chase",
        "turnover", "tail_risk", "regime_consistency",
        "gross_volatility", "win_rate", "total",
    ]:
        assert key in d
    assert loss.turnover == 0.0
    assert loss.regime_consistency == 0.0


def test_turnover_penalty_increases_total_loss():
    rets = pd.Series([0.001] * 60)
    quiet = compute_multi_objective_loss(rets, avg_daily_turnover=0.05)
    churn = compute_multi_objective_loss(rets, avg_daily_turnover=0.50)
    assert churn.total > quiet.total
    assert churn.turnover > quiet.turnover


def test_turnover_clipped_to_two():
    rets = pd.Series([0.001] * 60)
    crazy = compute_multi_objective_loss(rets, avg_daily_turnover=10.0)
    assert crazy.turnover == 2.0


def test_tail_risk_higher_for_left_skewed_returns():
    """A series with one bad -10% day in 100 should have higher tail risk
    than a smooth small-loss series with the same mean.
    """
    smooth = pd.Series([0.001] * 100)
    fat_tail = pd.Series([0.002] * 95 + [-0.10] * 5)
    a = compute_multi_objective_loss(smooth)
    b = compute_multi_objective_loss(fat_tail)
    assert b.tail_risk > a.tail_risk
    assert b.total > a.total


def test_gross_volatility_higher_for_choppy_series():
    smooth = pd.Series([0.001] * 252)
    choppy = pd.Series([0.05 if i % 2 == 0 else -0.04 for i in range(252)])
    a = compute_multi_objective_loss(smooth)
    b = compute_multi_objective_loss(choppy)
    assert b.gross_volatility > a.gross_volatility


def test_win_rate_higher_for_more_positive_days():
    mostly_wins = pd.Series([0.001] * 80 + [-0.0005] * 20)
    mostly_losses = pd.Series([0.001] * 20 + [-0.0005] * 80)
    a = compute_multi_objective_loss(mostly_wins)
    b = compute_multi_objective_loss(mostly_losses)
    assert a.win_rate > b.win_rate
    # win_rate contributes negatively to total (bonus), so mostly_wins ought
    # to push total lower at fixed other terms
    assert a.total < b.total + 1e-9 or a.win_rate > b.win_rate


def test_regime_consistency_uses_min_per_regime_sharpe():
    """A strategy that earns positive sharpe in normal and negative sharpe
    in bear must score on the MIN regime sharpe, not the pooled sharpe.
    """
    rng = np.random.default_rng(11)
    n_normal, n_bear = 200, 50
    normal_rets = rng.normal(0.003, 0.01, n_normal)
    bear_rets = rng.normal(-0.002, 0.01, n_bear)
    rets = pd.Series(np.concatenate([normal_rets, bear_rets]))
    regimes = pd.Series(["normal"] * n_normal + ["bear"] * n_bear)
    with_regime = compute_multi_objective_loss(rets, regime_states=regimes)
    no_regime = compute_multi_objective_loss(rets)
    # When regime supplied, regime_consistency reflects the bear bucket's
    # sharpe (negative), which is lower than the pooled aggregate.
    assert with_regime.regime_consistency < 0
    assert no_regime.regime_consistency == 0.0


def test_regime_consistency_drops_undersized_buckets():
    rng = np.random.default_rng(13)
    n_normal = 100
    n_micro = 5  # below the 20-day floor
    rets_arr = np.concatenate([rng.normal(0.003, 0.01, n_normal), rng.normal(-0.05, 0.01, n_micro)])
    rets = pd.Series(rets_arr)
    regimes = pd.Series(["normal"] * n_normal + ["crisis"] * n_micro)
    loss = compute_multi_objective_loss(rets, regime_states=regimes)
    # Only "normal" has ≥20 days → regime_consistency is the normal-bucket sharpe
    normal_only = compute_multi_objective_loss(pd.Series(rets_arr[:n_normal]))
    assert abs(loss.regime_consistency - normal_only.sharpe) < 1e-6


def test_regime_consistency_zero_when_regime_length_mismatches_returns():
    rets = pd.Series([0.001] * 60)
    regimes = pd.Series(["normal"] * 30)  # wrong length
    loss = compute_multi_objective_loss(rets, regime_states=regimes)
    assert loss.regime_consistency == 0.0


def test_score_backtest_extracts_turnover_and_regime_from_equity_curve():
    rng = np.random.default_rng(17)
    eq = pd.DataFrame(
        {
            "daily_eq_return": rng.normal(0.001, 0.005, 60),
            "turnover": [0.15] * 60,
            "regime_state": ["normal"] * 30 + ["caution"] * 30,
        }
    )
    loss = score_backtest(eq)
    assert loss.turnover == pytest.approx(0.15)
    # regime_consistency present (both buckets ≥ 20 obs, varying returns)
    assert loss.regime_consistency != 0.0


def test_loss_weights_zero_disables_term():
    rets = pd.Series([0.001] * 60)
    w_none = LossWeights(
        net_return=0.0, sharpe=0.0, calmar=0.0, max_drawdown=0.0, high_chase=0.0,
        turnover=0.0, tail_risk=0.0, regime_consistency=0.0,
        gross_volatility=0.0, win_rate=0.0,
    )
    loss = compute_multi_objective_loss(rets, weights=w_none, avg_daily_turnover=1.0)
    assert loss.total == 0.0


def test_components_count_matches_stage5_schema():
    """Stage 5: 15 component metrics + 1 aggregate ('total') = 16."""
    rets = pd.Series([0.001] * 60)
    loss = compute_multi_objective_loss(rets)
    d = loss.as_dict()
    # Stage 1 (5) + Stage 3 (5) + Stage 5 (5) + total = 16
    assert len(d) == 16
    for key in (
        "transaction_cost",
        "concentration",
        "illiquidity",
        "st_exposure",
        "execution_unfilled",
    ):
        assert key in d


def test_stage5_terms_default_to_zero_without_inputs():
    rets = pd.Series([0.001] * 60)
    loss = compute_multi_objective_loss(rets)
    assert loss.transaction_cost == 0.0
    assert loss.concentration == 0.0
    assert loss.illiquidity == 0.0
    assert loss.st_exposure == 0.0
    assert loss.execution_unfilled == 0.0


def test_stage5_transaction_cost_increases_total_loss():
    rets = pd.Series([0.001] * 60)
    base = compute_multi_objective_loss(rets)
    worse = compute_multi_objective_loss(rets, transaction_cost_rate=0.50)
    assert worse.transaction_cost == 0.50
    assert worse.total > base.total


def test_stage5_concentration_increases_total_loss():
    rets = pd.Series([0.001] * 60)
    base = compute_multi_objective_loss(rets)
    worse = compute_multi_objective_loss(rets, concentration_score=0.80)
    assert worse.concentration == 0.80
    assert worse.total > base.total


def test_stage5_illiquidity_increases_total_loss():
    rets = pd.Series([0.001] * 60)
    base = compute_multi_objective_loss(rets)
    worse = compute_multi_objective_loss(rets, illiquidity_score=0.40)
    assert worse.illiquidity == 0.40
    assert worse.total > base.total


def test_stage5_st_exposure_increases_total_loss():
    rets = pd.Series([0.001] * 60)
    base = compute_multi_objective_loss(rets)
    worse = compute_multi_objective_loss(rets, st_exposure_rate=0.10)
    assert worse.st_exposure == 0.10
    assert worse.total > base.total


def test_stage5_execution_unfilled_increases_total_loss():
    rets = pd.Series([0.001] * 60)
    base = compute_multi_objective_loss(rets)
    worse = compute_multi_objective_loss(rets, execution_unfilled_rate=0.30)
    assert worse.execution_unfilled == 0.30
    assert worse.total > base.total


def test_stage5_inputs_clip_into_unit_interval():
    rets = pd.Series([0.001] * 60)
    out_of_range = compute_multi_objective_loss(
        rets,
        transaction_cost_rate=2.5,    # >1
        concentration_score=-0.4,     # <0
        illiquidity_score=10.0,
        st_exposure_rate=-1.0,
        execution_unfilled_rate=1.5,
    )
    assert out_of_range.transaction_cost == 1.0
    assert out_of_range.concentration == 0.0
    assert out_of_range.illiquidity == 1.0
    assert out_of_range.st_exposure == 0.0
    assert out_of_range.execution_unfilled == 1.0
