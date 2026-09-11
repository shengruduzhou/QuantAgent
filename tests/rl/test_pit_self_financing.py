"""Cash conservation and frozen-share regression oracles, using synthetic bars."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")
from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig


def make_product(book, prices, flags=None, cost_bps=0):
    dates = pd.bdate_range('2024-01-08', periods=5)
    flags = flags or {}
    panel = pd.DataFrame([dict(trade_date=d, symbol=s, close=p[i], is_limit_up=False,
                              is_limit_down=False, is_suspended=flags.get((s, i), False))
                          for s, p in prices.items() for i, d in enumerate(dates)])
    preds = pd.DataFrame([dict(trade_date=d, symbol=s, alpha_score=float(j))
                          for d in dates for j, s in enumerate(prices)])
    return PITPortfolioEnv(pd.DataFrame(book, index=dates[:3]), preds, panel,
                           PITPortfolioEnvConfig(max_book=3, cost_bps=cost_bps))


def test_frozen_exit_cannot_fund_buy():
    env = make_product({'A':[1.,0.,0.], 'B':[0.,1.,1.]},
                       {'A':[100.]*5, 'B':[100.,100.,100.,110.,110.]}, {('A',2):True})
    env.reset()
    env.step(np.zeros(4))
    _, reward, _, _, info = env.step(np.zeros(4))
    assert info['weights']['A'] == pytest.approx(1.)
    assert info['weights']['B'] == pytest.approx(0.)
    assert info['nav'] == pytest.approx(1.)
    assert info['turnover_policy'] == pytest.approx(0.)
    assert reward == pytest.approx(0.)


def test_drift_keeps_frozen_asset_quantity():
    env = make_product({'A':[.5]*3, 'B':[.5]*3},
                       {'A':[100.,100.,110.,110.,110.], 'B':[100.,100.,100.,110.,110.]},
                       {('B',2):True})
    env.reset()
    _, _, _, _, first = env.step(np.zeros(4))
    assert first['nav'] == pytest.approx(1.05)
    _, _, _, _, second = env.step(np.zeros(4))
    assert second['weights']['B'] == pytest.approx(.5/1.05)
    assert second['nav'] == pytest.approx(1.10)
    assert second['turnover_policy'] == pytest.approx(.025/1.05)


def test_future_reward_price_cannot_change_next_signal_observation():
    observations, rewards = [], []
    for future in (110., 130.):
        env = make_product({'A':[.5]*3, 'B':[.5]*3},
                           {'A':[100.,100.,future,110.,110.], 'B':[100.]*5})
        env.reset()
        obs, reward, *_ = env.step(np.array([1.,0.,0.,0.]))
        observations.append(obs)
        rewards.append(reward)
    np.testing.assert_array_equal(*observations)
    assert rewards[0] != rewards[1]


def test_cash_pays_fees_before_investment():
    env = make_product({'A':[1.]*3}, {'A':[100.]*5}, cost_bps=100.)
    env.reset()
    _, reward, _, _, info = env.step(np.zeros(4))
    assert info['weights']['A'] == pytest.approx(1/1.01)
    assert info['nav'] == pytest.approx(1/1.01)
    assert info['cash_weight'] >= -1e-12
    assert reward == pytest.approx(0.)
