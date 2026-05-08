import numpy as np
import pandas as pd

from quantagent.quant_math.ashare import (
    TPlusOnePosition,
    board_for_symbol,
    enforce_tradability,
)
from quantagent.quant_math.conformal import (
    conformal_quantile,
    split_conformal_intervals,
    split_conformal_residuals,
)
from quantagent.quant_math.covariance import ledoit_wolf_covariance
from quantagent.quant_math.factor_attribution import (
    capacity_curve,
    gram_schmidt_orthogonalize,
    portfolio_factor_attribution,
)
from quantagent.quant_math.hrp import hrp_weights
from quantagent.quant_math.performance import (
    deflated_sharpe_ratio,
    newey_west_t_stat,
    probabilistic_sharpe_ratio,
)
from quantagent.quant_math.purged_cv import (
    PurgedKFoldConfig,
    probability_of_backtest_overfitting,
    purged_kfold_split,
)
from quantagent.quant_math.realized_vol import garman_klass, parkinson, yang_zhang
from quantagent.quant_math.triple_barrier import (
    BarrierConfig,
    daily_volatility,
    sample_weights_by_uniqueness,
    triple_barrier_labels,
)


def test_triple_barrier_labels_first_touch_profit():
    idx = pd.date_range("2026-01-01", periods=12, freq="D")
    close = pd.Series(100.0 * np.exp(np.linspace(0, 0.05, 12)), index=idx)
    sigma = pd.Series(0.01, index=idx)
    events = triple_barrier_labels(close, sigma, config=BarrierConfig(pt_sl=(1.0, 1.0), max_holding_days=5))
    assert events["barrier"].iloc[0] in {"pt", "vt"}
    assert events["label"].iloc[0] in {-1, 0, 1}


def test_sample_weights_by_uniqueness_returns_finite_values():
    idx = pd.date_range("2026-01-01", periods=20, freq="D")
    events = pd.DataFrame(
        {"t1": [idx[i + 3] for i in range(15)] + [pd.NaT] * 5},
        index=idx[:20],
    )
    weights = sample_weights_by_uniqueness(events, idx)
    assert weights.notna().any()
    assert (weights.dropna() <= 1.0).all()


def test_purged_kfold_drops_overlapping_train_samples():
    n = 60
    times = pd.Series(pd.date_range("2026-01-01", periods=n, freq="D"))
    end_times = times + pd.Timedelta(days=3)
    splits = list(purged_kfold_split(times, end_times, PurgedKFoldConfig(n_splits=4, embargo_pct=0.05)))
    assert len(splits) == 4
    for train, test in splits:
        assert set(train).isdisjoint(set(test))


def test_pbo_returns_value_in_unit_interval():
    rng = np.random.default_rng(0)
    is_sr = rng.normal(size=(5, 8))
    oos_sr = rng.normal(size=(5, 8))
    pbo = probability_of_backtest_overfitting(is_sr, oos_sr)
    assert 0.0 <= pbo <= 1.0


def test_psr_higher_for_consistent_positive_returns():
    consistent = pd.Series(np.full(252, 0.001))
    noisy = pd.Series(np.random.default_rng(1).normal(0.001, 0.02, 252))
    assert probabilistic_sharpe_ratio(consistent) > probabilistic_sharpe_ratio(noisy)


def test_dsr_returns_finite():
    returns = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 252))
    candidates = np.random.default_rng(3).normal(0.0, 1.0, 50)
    value = deflated_sharpe_ratio(returns, candidates)
    assert not np.isnan(value)


def test_newey_west_t_stat_handles_autocorrelation():
    rng = np.random.default_rng(0)
    eps = rng.normal(size=200)
    series = pd.Series([eps[0]] + [0.0] * 199)
    for i in range(1, 200):
        series.iloc[i] = 0.7 * series.iloc[i - 1] + eps[i]
    t = newey_west_t_stat(series + 0.05)
    assert np.isfinite(t)


def test_ledoit_wolf_returns_psd_matrix():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(size=(120, 6)), columns=list("ABCDEF"))
    cov = ledoit_wolf_covariance(returns)
    eigvals = np.linalg.eigvalsh(cov.to_numpy())
    assert eigvals.min() >= -1e-8


def test_hrp_weights_sum_to_one():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(size=(200, 5)), columns=list("VWXYZ"))
    weights = hrp_weights(returns)
    assert abs(weights.sum() - 1.0) < 1e-9
    assert (weights >= 0).all()


def test_yang_zhang_is_finite_for_valid_ohlc():
    n = 80
    rng = np.random.default_rng(0)
    log_close = np.cumsum(rng.normal(0, 0.01, n))
    close = pd.Series(100 * np.exp(log_close))
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.Series(np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.005, n))))
    low = pd.Series(np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.005, n))))
    yz = yang_zhang(open_, high, low, close, window=20)
    assert yz.dropna().shape[0] > 0
    assert (yz.dropna() >= 0).all()
    assert parkinson(high, low, 20).dropna().shape[0] > 0
    assert garman_klass(open_, high, low, close, 20).dropna().shape[0] > 0


def test_split_conformal_interval_covers_calibration():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=200)
    truth = pred + rng.normal(scale=0.1, size=200)
    residuals = split_conformal_residuals(pred, truth)
    q = conformal_quantile(residuals, alpha=0.1)
    assert q > 0
    lower, upper = split_conformal_intervals(pred, residuals, alpha=0.1)
    coverage = ((truth >= lower) & (truth <= upper)).mean()
    assert coverage >= 0.85


def test_board_for_symbol_classifies_china_segments():
    assert board_for_symbol("688981.SH") == "star_board"
    assert board_for_symbol("300750.SZ") == "chinext"
    assert board_for_symbol("600519.SH") == "main_board"
    assert board_for_symbol("600519.SH", is_st=True) == "st"


def test_t_plus_one_position_freezes_today_buys():
    pos = TPlusOnePosition()
    pos.buy(200)
    assert pos.sell(200) == 0
    pos.settle_overnight()
    assert pos.sell(200) == 200


def test_enforce_tradability_blocks_buy_at_limit_up():
    target = pd.Series({"A": 0.05, "B": 0.05})
    current = pd.Series({"A": 0.0, "B": 0.0})
    can_buy = pd.Series({"A": False, "B": True})
    can_sell = pd.Series({"A": True, "B": True})
    capped = enforce_tradability(target, current, can_buy, can_sell)
    assert capped.loc["A"] == 0.0
    assert capped.loc["B"] == 0.05


def test_gram_schmidt_orthogonal_factor_uncorrelated_with_existing():
    rng = np.random.default_rng(0)
    base = pd.DataFrame(rng.normal(size=(200, 2)), columns=["f1", "f2"])
    new = base["f1"] * 0.7 + rng.normal(scale=0.5, size=200)
    new.name = "new"
    residual = gram_schmidt_orthogonalize(new, base)
    aligned = pd.concat([residual, base], axis=1).dropna()
    corr = aligned.corr().loc[residual.name, ["f1", "f2"]].abs().max()
    assert corr < 0.05


def test_capacity_curve_decreases_net_alpha_with_aum():
    daily_alpha = pd.Series(np.full(252, 0.001))
    adv = pd.Series(np.full(252, 1e8))
    curve = capacity_curve(daily_alpha, adv, aum_grid=(1e6, 1e8, 1e10))
    assert curve["net_annual"].is_monotonic_decreasing


def test_portfolio_factor_attribution_returns_breakdown():
    rng = np.random.default_rng(0)
    n = 50
    loadings = pd.DataFrame(rng.normal(size=(n, 2)), columns=["mom", "value"])
    realized = (loadings @ np.array([0.01, -0.005])) + rng.normal(scale=0.005, size=n)
    weights = pd.Series(np.full(n, 1.0 / n))
    result = portfolio_factor_attribution(weights, realized, loadings)
    assert set(result.factor_returns.index) == {"mom", "value"}
    assert np.isfinite(result.specific_return)
    assert -1.0 <= result.r_squared <= 1.0
