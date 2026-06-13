"""PIT portfolio env invariants: zero-action ⇒ zero reward, PIT universe, flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gym = pytest.importorskip("gymnasium")

from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig

DATES = pd.date_range("2026-01-05", periods=8, freq="B")
SYMS = ["A", "B", "C", "D"]


def _panel(limit_up: dict | None = None) -> pd.DataFrame:
    limit_up = limit_up or {}
    rng = np.random.default_rng(7)
    rows = []
    for si, s in enumerate(SYMS):
        px = 10.0 + si
        for d in DATES:
            px *= 1 + rng.normal(0.001, 0.01)
            rows.append({"symbol": s, "trade_date": d, "close": px,
                         "is_limit_up": bool(limit_up.get((s, d), False)),
                         "is_limit_down": False, "is_suspended": False})
    return pd.DataFrame(rows)


def _book() -> pd.DataFrame:
    # holds A,B first 4 days then A,C
    rows = {}
    for i, d in enumerate(DATES[:-1]):
        held = ["A", "B"] if i < 4 else ["A", "C"]
        rows[d] = pd.Series(0.5, index=held)
    return pd.DataFrame(rows).T.fillna(0.0)


def _preds() -> pd.DataFrame:
    rows = [{"symbol": s, "trade_date": d, "alpha_score": float(len(SYMS) - i)}
            for d in DATES for i, s in enumerate(SYMS)]
    return pd.DataFrame(rows)


def _env(limit_up=None, **cfg_kwargs) -> PITPortfolioEnv:
    cfg = PITPortfolioEnvConfig(max_book=4, **cfg_kwargs)
    return PITPortfolioEnv(_book(), _preds(), _panel(limit_up), cfg)


class TestZeroActionInvariant:
    def test_zero_action_zero_reward_every_step(self):
        env = _env()
        obs, _ = env.reset()
        done = False
        while not done:
            obs, reward, done, _, info = env.step(np.zeros(env.action_space.shape))
            assert reward == pytest.approx(0.0, abs=1e-12)
            assert info["value_add"] == pytest.approx(0.0, abs=1e-12)
        assert env._nav == pytest.approx(env._nav_passive, rel=1e-12)

    def test_zero_action_zero_reward_even_with_limit_up(self):
        # day-0 limit-up on A: both books face the same constraint
        env = _env(limit_up={("A", DATES[0]): True})
        env.reset()
        _, reward, *_ = env.step(np.zeros(env.action_space.shape))
        assert reward == pytest.approx(0.0, abs=1e-12)


class TestPITUniverse:
    def test_universe_is_each_days_book(self):
        env = _env()
        assert set(env.slot_symbols[0]) == {"A", "B"}
        assert set(env.slot_symbols[5]) == {"A", "C"}  # book switch respected

    def test_weights_only_on_book_names(self):
        env = _env()
        env.reset()
        _, _, _, _, info = env.step(np.ones(env.action_space.shape))
        nonzero = {s for s, w in info["weights"].items() if w > 1e-9}
        assert nonzero <= {"A", "B"}


class TestFlagConstraints:
    def test_limit_up_name_cannot_be_increased(self):
        env = _env(limit_up={("A", DATES[1]): True})
        env.reset()
        a = np.zeros(env.action_space.shape)
        _, _, _, _, info0 = env.step(a)          # day0: passive
        w_a_before = info0["weights"].get("A", 0.0)
        a_up = np.zeros(env.action_space.shape)
        idx = env.slot_symbols[1].index("A")
        a_up[idx] = 1.0                           # try to add to A on its limit-up day
        _, _, _, _, info1 = env.step(a_up)
        assert info1["weights"].get("A", 0.0) <= w_a_before + 1e-12

    def test_frozen_name_keeps_weight(self):
        panel = _panel()
        panel.loc[(panel["symbol"] == "B") & (panel["trade_date"] == DATES[1]),
                  "is_suspended"] = True
        env = PITPortfolioEnv(_book(), _preds(), panel, PITPortfolioEnvConfig(max_book=4))
        env.reset()
        _, _, _, _, info0 = env.step(np.zeros(env.action_space.shape))
        w_b = info0["weights"]["B"]
        a = np.full(env.action_space.shape, -1.0)  # try to dump everything
        _, _, _, _, info1 = env.step(a)
        assert info1["weights"]["B"] == pytest.approx(w_b, rel=1e-9)


class TestCostAccounting:
    def test_extra_turnover_costs_reduce_reward(self):
        env = _env(cost_bps=50.0)
        env.reset()
        env.step(np.zeros(env.action_space.shape))
        # alternate big tilts -> policy pays turnover the passive book doesn't
        rewards = []
        sign = 1.0
        done = False
        while not done:
            a = np.zeros(env.action_space.shape)
            a[0], a[1] = sign, -sign
            sign = -sign
            _, r, done, _, _ = env.step(a)
            rewards.append(r)
        assert np.mean(rewards) < 0  # random churn must lose after costs in expectation
