import numpy as np
import pandas as pd

from quantagent.quant_math.regime import MarketRegime
from quantagent.quant_math.regime_aware_optimizer import (
    RegimeAwareConfig,
    solve_v5_portfolio,
)


def _toy_inputs(n: int = 5):
    symbols = [f"S{i}" for i in range(n)]
    alpha = pd.Series(np.linspace(0.005, 0.025, n), index=symbols)
    covariance = pd.DataFrame(np.eye(n) * 0.04, index=symbols, columns=symbols)
    return alpha, covariance


def test_regime_aware_optimizer_returns_weights_and_diagnostics():
    alpha, covariance = _toy_inputs(5)
    config = RegimeAwareConfig(regime=MarketRegime.RANGE_BOUND.value)
    result = solve_v5_portfolio(alpha, covariance, config=config)
    assert result.target_weights.shape[0] == 5
    assert "regime" in result.constraint_diagnostics


def test_regime_aware_optimizer_scales_with_crisis_regime():
    alpha, covariance = _toy_inputs(5)
    base = solve_v5_portfolio(alpha, covariance, config=RegimeAwareConfig(regime=MarketRegime.RANGE_BOUND.value))
    crisis = solve_v5_portfolio(alpha, covariance, config=RegimeAwareConfig(regime=MarketRegime.LIQUIDITY_CRISIS.value))
    assert base.target_weights.sum() >= crisis.target_weights.sum() - 1e-6
