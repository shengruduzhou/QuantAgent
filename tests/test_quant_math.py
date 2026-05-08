import numpy as np
import pandas as pd

from quantagent.quant_math.constraints import weights_to_lot_shares
from quantagent.quant_math.ic_analysis import ic_summary, rank_ic_by_date
from quantagent.quant_math.optimizer import ContinuousMeanVarianceOptimizer, OptimizerConfig
from quantagent.quant_math.regime import MarketRegime, detect_regime
from quantagent.quant_math.risk_metrics import historical_cvar, historical_var
from quantagent.quant_math.signal_fusion import precision_weighted_alpha


def test_rank_ic_summary_detects_positive_signal():
    frame = pd.DataFrame(
        {
            "trade_date": ["2026-01-01"] * 4 + ["2026-01-02"] * 4,
            "signal": [1, 2, 3, 4, 1, 2, 3, 4],
            "future_return": [0.01, 0.02, 0.03, 0.04, 0.00, 0.01, 0.02, 0.03],
        }
    )

    rank_ic = rank_ic_by_date(frame, "signal", "future_return")
    summary = ic_summary(rank_ic)

    assert summary["mean"] > 0.99
    assert summary["positive_ratio"] == 1.0


def test_precision_weighted_alpha_prefers_lower_error_model():
    predictions = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "alpha": [0.01, 0.03],
            "error_variance": [0.0001, 0.01],
            "rank_ic": [0.05, 0.05],
        }
    )

    fused = precision_weighted_alpha(predictions)

    assert fused.loc["A"] < 0.015


def test_risk_metrics_return_positive_tail_loss():
    returns = pd.Series([0.01, -0.01, -0.03, 0.02, -0.05])

    assert historical_var(returns, 0.8) > 0
    assert historical_cvar(returns, 0.8) >= historical_var(returns, 0.8)


def test_regime_detects_liquidity_crisis_before_trend():
    row = pd.Series({"market_trend": 0.10, "market_vol": 0.01, "liquidity_change": -0.30})

    assert detect_regime(row) == MarketRegime.LIQUIDITY_CRISIS


def test_optimizer_respects_basic_weight_limits():
    alpha = pd.Series([0.03, 0.02, 0.01], index=["A", "B", "C"])
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=alpha.index, columns=alpha.index)
    optimizer = ContinuousMeanVarianceOptimizer(
        OptimizerConfig(max_position_weight=0.1, max_total_weight=0.2, max_turnover=0.2)
    )

    result = optimizer.solve(alpha, covariance)

    assert result.weights.sum() <= 0.2000001
    assert result.weights.max() <= 0.1000001


def test_weights_to_lot_shares_rounds_down_to_100_shares():
    weights = pd.Series([0.051], index=["300750.SZ"])
    prices = pd.Series([203.0], index=["300750.SZ"])

    shares = weights_to_lot_shares(weights, nav=100_000, prices=prices, lot_size=100)

    assert shares.loc["300750.SZ"] == 0
