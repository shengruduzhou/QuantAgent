import numpy as np
import pandas as pd

from quantagent.agents.arbitration import aggregate_agent_signals, agent_reliability_weights
from quantagent.domain.schemas import AgentSignal
from quantagent.fundamental.quality import fraud_risk_score, long_horizon_score, quality_score
from quantagent.fundamental.valuation import DCFInputs, dcf_intrinsic_value_per_share, margin_of_safety
from quantagent.quant_math.performance import hit_ratio, max_drawdown, profit_factor, sharpe_ratio
from quantagent.quant_math.technical_indicators import add_advanced_technical_indicators
from quantagent.strategy.rule_signals import add_short_horizon_rule_signals
from quantagent.strategy.weight_adapter import combine_short_long_weights, short_signal_to_weight


def test_agent_arbitration_weights_sum_to_one():
    stats = pd.DataFrame(
        {
            "agent_name": ["technical", "event"],
            "ir": [0.5, 0.1],
            "evidence_quality": [0.8, 0.7],
            "error": [0.1, 0.3],
        }
    )

    weights = agent_reliability_weights(stats)

    assert abs(weights.sum() - 1.0) < 1e-9
    assert weights.loc["technical"] > weights.loc["event"]


def test_agent_signals_aggregate_without_orders():
    signals = [
        AgentSignal("event", "NVDA", 10, 0.7, 0.8, 0.9, 0.1),
        AgentSignal("risk", "NVDA", 10, -0.3, 0.7, 0.8, 0.2),
    ]

    score = aggregate_agent_signals(signals)

    assert "NVDA" in score.index


def test_dcf_and_margin_of_safety_are_positive_for_cheap_stock():
    intrinsic = dcf_intrinsic_value_per_share(
        DCFInputs(
            fcff=1_000_000_000,
            growth_rate=0.08,
            terminal_growth_rate=0.03,
            wacc=0.10,
            years=5,
            net_debt=500_000_000,
            shares_outstanding=100_000_000,
        )
    )

    assert intrinsic > 0
    assert margin_of_safety(intrinsic, intrinsic * 0.8) > 0


def test_quality_and_fraud_scores_are_bounded():
    quality = quality_score(0.18, 0.09, 0.8, 0.9, 1.5, 8.0)
    fraud = fraud_risk_score(0.9, 0.8, 0.7, 0.6)
    long_score = long_horizon_score(80, quality, 70, 75, 60, 65, fraud)

    assert 0 <= quality <= 100
    assert 0 <= fraud <= 100
    assert 0 <= long_score <= 100


def test_technical_rule_pipeline_emits_short_signal():
    dates = pd.date_range("2026-01-01", periods=40, freq="D")
    close = pd.Series(np.linspace(10, 14, 40))
    prices = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": "TEST",
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.linspace(1000, 2000, 40),
            "amount": close * np.linspace(1000, 2000, 40),
        }
    )

    features = add_advanced_technical_indicators(prices)
    signals = add_short_horizon_rule_signals(features)

    assert "macd_hist" in signals.columns
    assert "short_rule_signal" in signals.columns


def test_weight_adapter_keeps_target_weight_bounded():
    short_weight = short_signal_to_weight(0.5, volatility=0.02, max_abs_weight=0.04)
    target = combine_short_long_weights("TEST", short_weight, 0.08, horizon_days=5, confidence=0.75)

    assert abs(target.target_weight) <= 0.10


def test_performance_metrics_are_defined_for_simple_returns():
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    nav = (1.0 + returns).cumprod()

    assert sharpe_ratio(returns) == sharpe_ratio(returns)
    assert max_drawdown(nav) <= 0
    assert hit_ratio(returns) > 0
    assert profit_factor(returns) > 0
