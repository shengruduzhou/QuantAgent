"""PIT RL invariants for executable reward timing, universe, flags and costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gym = pytest.importorskip("gymnasium")

from quantagent.rl.pit_portfolio_env import (
    PITPortfolioEnv,
    PITPortfolioEnvConfig,
    RL_REWARD_SEMANTICS,
)

DATES = pd.date_range("2026-01-05", periods=10, freq="B")
SYMS = ["A", "B", "C", "D"]


def _panel(
    limit_up: dict | None = None,
    limit_down: dict | None = None,
    suspended: dict | None = None,
    st: dict | None = None,
) -> pd.DataFrame:
    limit_up = limit_up or {}
    limit_down = limit_down or {}
    suspended = suspended or {}
    st = st or {}
    rng = np.random.default_rng(7)
    rows = []
    for si, symbol in enumerate(SYMS):
        price = 10.0 + si
        for date in DATES:
            price *= 1 + rng.normal(0.001, 0.01)
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": date,
                    "close": price,
                    "is_limit_up": limit_up.get((symbol, date), False),
                    "is_limit_down": limit_down.get((symbol, date), False),
                    "is_suspended": suspended.get((symbol, date), False),
                    "is_st": st.get((symbol, date), False),
                }
            )
    return pd.DataFrame(rows)


def _book() -> pd.DataFrame:
    rows = {}
    for i, date in enumerate(DATES[:-2]):
        held = ["A", "B"] if i < 4 else ["A", "C"]
        rows[date] = pd.Series(0.5, index=held)
    return pd.DataFrame(rows).T.fillna(0.0)


def _preds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": symbol, "trade_date": date, "alpha_score": float(len(SYMS) - i)}
            for date in DATES
            for i, symbol in enumerate(SYMS)
        ]
    )


def _env(panel: pd.DataFrame | None = None, **cfg_kwargs) -> PITPortfolioEnv:
    cfg = PITPortfolioEnvConfig(max_book=4, **cfg_kwargs)
    return PITPortfolioEnv(
        _book(),
        _preds(),
        _panel() if panel is None else panel,
        DATES,
        cfg,
    )


class TestExecutableRewardClock:
    def test_reward_uses_t1_to_t2_not_signal_to_t1(self):
        rows = []
        paths = {
            "A": [10.0, 11.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0],
            "B": [20.0] * len(DATES),
            "C": [30.0] * len(DATES),
            "D": [40.0] * len(DATES),
        }
        for symbol in SYMS:
            for date, close in zip(DATES, paths[symbol]):
                rows.append(
                    {
                        "symbol": symbol,
                        "trade_date": date,
                        "close": close,
                        "is_limit_up": False,
                        "is_limit_down": False,
                        "is_suspended": False,
                        "is_st": False,
                    }
                )
        env = _env(pd.DataFrame(rows), cost_bps=0.0)
        first_symbols = env.slot_symbols[0]
        a_slot = first_symbols.index("A")
        assert env.slot_ret[0, a_slot] == pytest.approx(1.0)  # 11 -> 22
        assert env.execution_dates[0] == DATES[1]
        assert env.reward_end_dates[0] == DATES[2]

        env.reset()
        action = np.zeros(env.action_space.shape)
        action[a_slot] = 1.0
        _, _, _, _, info = env.step(action)
        assert info["signal_date"] == str(DATES[0].date())
        assert info["execution_date"] == str(DATES[1].date())
        assert info["reward_end_date"] == str(DATES[2].date())
        assert info["reward_semantics"] == RL_REWARD_SEMANTICS

    def test_missing_exact_reward_bar_fails_closed_instead_of_row_shift(self):
        panel = _panel()
        # A is in the first passive book. Removing its exact T+2 row must not
        # silently use A's next later observation as the reward endpoint.
        panel = panel[
            ~((panel["symbol"] == "A") & (panel["trade_date"] == DATES[2]))
        ]
        with pytest.raises(ValueError, match="missing exact T\\+1->T\\+2 executable reward"):
            _env(panel)

    def test_book_signal_date_must_belong_to_explicit_market_calendar(self):
        bad_book = _book().copy()
        bad_book.index = bad_book.index.where(
            bad_book.index != DATES[0], pd.Timestamp("2026-01-04")
        )
        with pytest.raises(ValueError, match="book signal dates absent from market_sessions"):
            PITPortfolioEnv(
                bad_book,
                _preds(),
                _panel(),
                DATES,
                PITPortfolioEnvConfig(max_book=4),
            )


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

    def test_zero_action_zero_reward_even_with_entry_limit_up(self):
        # Signal is D0; execution constraint is evaluated at mapped T+1 = D1.
        env = _env(_panel(limit_up={("A", DATES[1]): True}))
        env.reset()
        _, reward, *_ = env.step(np.zeros(env.action_space.shape))
        assert reward == pytest.approx(0.0, abs=1e-12)


class TestPITUniverse:
    def test_universe_is_each_signal_days_book(self):
        env = _env()
        assert set(env.slot_symbols[0]) == {"A", "B"}
        assert set(env.slot_symbols[5]) == {"A", "C"}

    def test_weights_only_on_book_names(self):
        env = _env()
        env.reset()
        _, _, _, _, info = env.step(np.ones(env.action_space.shape))
        nonzero = {symbol for symbol, weight in info["weights"].items() if weight > 1e-9}
        assert nonzero <= {"A", "B"}


class TestExecutionFlagConstraints:
    def test_limit_up_at_t1_prevents_increase(self):
        env = _env(_panel(limit_up={("A", DATES[2]): True}))
        env.reset()
        _, _, _, _, info0 = env.step(np.zeros(env.action_space.shape))
        before = info0["weights"].get("A", 0.0)
        # Second action is formed at D1 and executes D2, where A is limit-up.
        idx = env.slot_symbols[1].index("A")
        action = np.zeros(env.action_space.shape)
        action[idx] = 1.0
        _, _, _, _, info1 = env.step(action)
        assert info1["weights"].get("A", 0.0) <= before + 1e-12

    def test_limit_down_at_t1_prevents_decrease_but_not_increase(self):
        env = _env(_panel(limit_down={("A", DATES[2]): True}))
        env.reset()
        _, _, _, _, info0 = env.step(np.zeros(env.action_space.shape))
        before = info0["weights"]["A"]
        idx = env.slot_symbols[1].index("A")
        action = np.zeros(env.action_space.shape)
        action[idx] = -1.0
        _, _, _, _, info1 = env.step(action)
        assert info1["weights"]["A"] >= before - 1e-12

    def test_suspended_at_t1_freezes_weight(self):
        env = _env(_panel(suspended={("B", DATES[2]): True}))
        env.reset()
        _, _, _, _, info0 = env.step(np.zeros(env.action_space.shape))
        before = info0["weights"]["B"]
        action = np.full(env.action_space.shape, -1.0)
        _, _, _, _, info1 = env.step(action)
        assert info1["weights"]["B"] == pytest.approx(before, rel=1e-9)

    def test_unknown_execution_flag_fails_closed(self):
        panel = _panel()
        panel.loc[
            (panel["symbol"] == "A") & (panel["trade_date"] == DATES[1]),
            "is_suspended",
        ] = pd.NA
        env = _env(panel)
        env.reset()
        idx = env.slot_symbols[0].index("A")
        action = np.zeros(env.action_space.shape)
        action[idx] = 1.0
        _, _, _, _, info = env.step(action)
        assert info["weights"].get("A", 0.0) == pytest.approx(0.0, abs=1e-12)


class TestCostAccounting:
    def test_extra_turnover_costs_reduce_reward(self):
        env = _env(cost_bps=50.0)
        env.reset()
        env.step(np.zeros(env.action_space.shape))
        rewards = []
        sign = 1.0
        done = False
        while not done:
            action = np.zeros(env.action_space.shape)
            action[0], action[1] = sign, -sign
            sign = -sign
            _, reward, done, _, _ = env.step(action)
            rewards.append(reward)
        assert np.mean(rewards) < 0
