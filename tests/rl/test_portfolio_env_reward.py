from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from quantagent.rl.portfolio_env import PortfolioEnv, PortfolioEnvConfig


@pytest.mark.skipif(importlib.util.find_spec("gymnasium") is None, reason="gymnasium optional")
def test_portfolio_env_reward_reports_equal_weight_excess():
    dates = pd.bdate_range("2024-01-02", periods=4)
    predictions = pd.DataFrame([
        {"trade_date": d, "symbol": symbol, "prediction": 1.0 if symbol == "A" else 0.5}
        for d in dates for symbol in ["A", "B"]
    ])
    market = pd.DataFrame([
        {"trade_date": dates[0], "symbol": "A", "close": 10.0},
        {"trade_date": dates[0], "symbol": "B", "close": 10.0},
        {"trade_date": dates[1], "symbol": "A", "close": 11.0},
        {"trade_date": dates[1], "symbol": "B", "close": 10.0},
        {"trade_date": dates[2], "symbol": "A", "close": 12.0},
        {"trade_date": dates[2], "symbol": "B", "close": 10.0},
        {"trade_date": dates[3], "symbol": "A", "close": 13.0},
        {"trade_date": dates[3], "symbol": "B", "close": 10.0},
    ])
    env = PortfolioEnv(
        predictions,
        market,
        PortfolioEnvConfig(top_n=2, max_delta=1.0, max_weight_per_name=1.0, max_turnover=1.0, cost_bps=0.0),
    )
    env.reset()
    action = [0.0, 0.0]
    action[env.symbols.index("A")] = 1.0

    _, reward, _, _, info = env.step(action)

    assert info["benchmark_return"] == pytest.approx(0.05)
    assert info["excess_pnl"] == pytest.approx(0.05)
    assert reward == pytest.approx(0.05)
