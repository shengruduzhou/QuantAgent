from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from quantagent.factors.expression_safety import (
    expression_leakage_reasons,
    validate_feature_expression,
)
from quantagent.fusion.search_corrected import _within_fold_returns
from quantagent.quant_math.black_scholes import black_scholes, implied_volatility
from quantagent.quant_math.performance import probabilistic_sharpe_ratio, sharpe_ratio
from quantagent.research.foundation_gates import _staggered_daily_returns


def test_probabilistic_sharpe_uses_pearson_kurtosis_convention() -> None:
    rng = np.random.default_rng(7)
    values = pd.Series(rng.standard_t(df=6, size=900) * 0.01 + 0.0006)
    observed = probabilistic_sharpe_ratio(values)
    sr = sharpe_ratio(values) / np.sqrt(252.0)
    skew = float(values.skew())
    excess_kurtosis = float(values.kurt())
    denom = np.sqrt(
        max(
            1.0 - skew * sr + ((excess_kurtosis + 2.0) / 4.0) * sr**2,
            1e-12,
        )
    )
    expected = NormalDist().cdf(sr * np.sqrt(len(values) - 1) / denom)
    assert observed == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_fold_nav_reset_is_not_counted_as_a_return() -> None:
    dates = pd.bdate_range("2026-01-05", periods=8)
    nav = pd.Series(
        [1.0, 1.10, 1.21, np.nan, 1.0, 1.05, 1.1025, np.nan],
        index=dates,
    )
    folds = [
        {
            "foldIndex": "0",
            "trainStart": "2025-01-01",
            "trainEnd": "2025-12-31",
            "testStart": str(dates[0].date()),
            "testEnd": str(dates[2].date()),
        },
        {
            "foldIndex": "1",
            "trainStart": "2025-01-01",
            "trainEnd": "2025-12-31",
            "testStart": str(dates[4].date()),
            "testEnd": str(dates[6].date()),
        },
    ]
    returns = _within_fold_returns(nav, folds)
    np.testing.assert_allclose(returns.to_numpy(), [0.10, 0.10, 0.05, 0.05])
    assert (returns > -0.20).all()


def test_horizon_benchmark_is_converted_to_equal_capital_staggered_nav() -> None:
    index = pd.bdate_range("2026-01-05", periods=10)
    horizon_returns = pd.Series(0.10, index=index)
    daily = _staggered_daily_returns(horizon_returns, horizon_days=2)
    assert len(daily) == len(index) - 1
    assert np.isfinite(daily).all()
    # Raw 2-day forward returns are +10%. The equal-capital two-sleeve portfolio
    # must not book +10% on every business day.
    assert float(daily.max()) < 0.10
    assert float(daily.mean()) > 0.0


def test_formulaic_alpha_blocks_qlib_future_ref() -> None:
    assert "negative_ref_is_future" in expression_leakage_reasons(
        "Ref($close, -1) / $close - 1"
    )
    with pytest.raises(ValueError, match="not PIT-safe"):
        validate_feature_expression("Rank(Ref($close, -2))")
    assert validate_feature_expression("Rank(Ref($close, 5) / $close)")


def test_black_scholes_zero_vol_uses_discounted_forward_boundary() -> None:
    # spot < strike, yet high positive rates make the discounted strike smaller
    # than discounted spot. The deterministic call is therefore ITM.
    result = black_scholes(
        spot=100.0,
        strike=105.0,
        maturity=2.0,
        rate=0.10,
        dividend_yield=0.0,
        volatility=0.0,
        option_type="call",
    )
    expected = 100.0 - 105.0 * np.exp(-0.20)
    assert result.price == pytest.approx(expected, abs=1e-12)
    assert result.delta == pytest.approx(1.0, abs=1e-12)


def test_black_scholes_parity_and_implied_vol_recovery() -> None:
    kwargs = dict(
        spot=100.0,
        strike=105.0,
        maturity=0.75,
        rate=0.025,
        volatility=0.31,
        dividend_yield=0.01,
    )
    call = black_scholes(option_type="call", **kwargs)
    put = black_scholes(option_type="put", **kwargs)
    rhs = (
        kwargs["spot"] * np.exp(-kwargs["dividend_yield"] * kwargs["maturity"])
        - kwargs["strike"] * np.exp(-kwargs["rate"] * kwargs["maturity"])
    )
    assert call.price - put.price == pytest.approx(rhs, abs=1e-10)
    recovered = implied_volatility(
        call.price,
        spot=kwargs["spot"],
        strike=kwargs["strike"],
        maturity=kwargs["maturity"],
        rate=kwargs["rate"],
        dividend_yield=kwargs["dividend_yield"],
        option_type="call",
    )
    assert recovered == pytest.approx(kwargs["volatility"], abs=1e-7)


def test_implied_volatility_rejects_arbitrage_price() -> None:
    with pytest.raises(ValueError, match="no-arbitrage bounds"):
        implied_volatility(
            101.0,
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.0,
        )
