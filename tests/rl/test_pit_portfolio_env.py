"""PIT portfolio env invariants for the executable T -> T+1 -> T+2 clock."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gym = pytest.importorskip("gymnasium")

from quantagent.rl.pit_portfolio_env import (
    RL_REWARD_CLOCK_SEMANTICS,
    PITPortfolioEnv,
    PITPortfolioEnvConfig,
)

DATES = pd.date_range("2026-01-05", periods=9, freq="B")
SYMS = ["A", "B", "C", "D"]


def _panel(
    limit_up: dict | None = None,
    limit_down: dict | None = None,
    suspended: dict | None = None,
) -> pd.DataFrame:
    limit_up = limit_up or {}
    limit_down = limit_down or {}
    suspended = suspended or {}
    rng = np.random.default_rng(7)
    rows = []
    for si, symbol in enumerate(SYMS):
        price = 10.0 + si
        for trade_date in DATES:
            price *= 1 + rng.normal(0.001, 0.01)
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "close": price,
                    "is_limit_up": bool(limit_up.get((symbol, trade_date), False)),
                    "is_limit_down": bool(limit_down.get((symbol, trade_date), False)),
                    "is_suspended": bool(suspended.get((symbol, trade_date), False)),
                }
            )
    return pd.DataFrame(rows)


def _book() -> pd.DataFrame:
    rows = {}
    for i, trade_date in enumerate(DATES[:-1]):
        held = ["A", "B"] if i < 4 else ["A", "C"]
        rows[trade_date] = pd.Series(0.5, index=held)
    return pd.DataFrame(rows).T.fillna(0.0)


def _preds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "alpha_score": float(len(SYMS) - i),
            }
            for trade_date in DATES
            for i, symbol in enumerate(SYMS)
        ]
    )


def _env(
    limit_up=None,
    limit_down=None,
    suspended=None,
    *,
    session_gaps: pd.DataFrame | None = None,
    **cfg_kwargs,
) -> PITPortfolioEnv:
    return PITPortfolioEnv(
        _book(),
        _preds(),
        _panel(limit_up, limit_down, suspended),
        PITPortfolioEnvConfig(max_book=4, **cfg_kwargs),
        session_gaps=session_gaps,
    )


class TestExecutableClock:
    def test_signal_execution_and_reward_dates_are_distinct(self):
        env = _env()
        assert env.dates[0] == DATES[0]
        assert env.execution_dates[0] == DATES[1]
        assert env.reward_end_dates[0] == DATES[2]
        assert env.reward_clock_semantics == RL_REWARD_CLOCK_SEMANTICS

        env.reset()
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape))
        assert info["signal_date"] == DATES[0].date().isoformat()
        assert info["execution_date"] == DATES[1].date().isoformat()
        assert info["reward_end_date"] == DATES[2].date().isoformat()
        assert info["reward_clock_semantics"] == RL_REWARD_CLOCK_SEMANTICS

    def test_reward_return_begins_after_execution_close(self):
        panel = _panel()
        mask_t0 = (panel["symbol"] == "A") & (panel["trade_date"] == DATES[0])
        mask_t1 = (panel["symbol"] == "A") & (panel["trade_date"] == DATES[1])
        mask_t2 = (panel["symbol"] == "A") & (panel["trade_date"] == DATES[2])
        panel.loc[mask_t0, "close"] = 10.0
        panel.loc[mask_t1, "close"] = 20.0
        panel.loc[mask_t2, "close"] = 18.0

        env = PITPortfolioEnv(
            _book(), _preds(), panel, PITPortfolioEnvConfig(max_book=4)
        )
        a_index = env.slot_symbols[0].index("A")
        assert env.slot_ret[0, a_index] == pytest.approx(-0.10)
        assert env.slot_ret[0, a_index] != pytest.approx(1.0)

    def test_training_boundary_censors_reward_end_not_only_signal_date(self):
        env = _env(reward_end_date_limit=DATES[5].date().isoformat())
        assert max(env.reward_end_dates) <= DATES[5]
        assert env.dates[-1] == DATES[3]
        assert env.execution_dates[-1] == DATES[4]
        assert env.reward_end_dates[-1] == DATES[5]

    def test_signal_day_limit_up_does_not_block_next_session_execution(self):
        env = _env(limit_up={("A", DATES[0]): True})
        env.reset()
        action = np.zeros(env.action_space.shape)
        action[env.slot_symbols[0].index("A")] = 1.0
        _, _, _, _, info = env.step(action)
        assert info["weights"]["A"] > 0.0

    def test_execution_day_limit_up_blocks_increase(self):
        env = _env(limit_up={("A", DATES[1]): True})
        env.reset()
        action = np.zeros(env.action_space.shape)
        action[env.slot_symbols[0].index("A")] = 1.0
        _, _, _, _, info = env.step(action)
        assert info["weights"]["A"] == pytest.approx(0.0)


class TestZeroActionInvariant:
    def test_zero_action_zero_reward_every_step(self):
        env = _env()
        env.reset()
        done = False
        while not done:
            _, reward, done, _, info = env.step(np.zeros(env.action_space.shape))
            assert reward == pytest.approx(0.0, abs=1e-12)
            assert info["value_add"] == pytest.approx(0.0, abs=1e-12)
        assert env._nav == pytest.approx(env._nav_passive, rel=1e-12)

    def test_zero_action_zero_reward_even_with_execution_limit_up(self):
        env = _env(limit_up={("A", DATES[1]): True})
        env.reset()
        _, reward, *_ = env.step(np.zeros(env.action_space.shape))
        assert reward == pytest.approx(0.0, abs=1e-12)


class TestPITUniverse:
    def test_universe_is_each_signal_books_target_plus_exit_names(self):
        env = _env()
        assert set(env.slot_symbols[0]) == {"A", "B"}
        switch_index = env.dates.index(DATES[4])
        assert set(env.slot_symbols[switch_index]) == {"A", "B", "C"}

    def test_weights_only_open_on_book_names(self):
        env = _env()
        env.reset()
        _, _, _, _, info = env.step(np.ones(env.action_space.shape))
        nonzero = {
            symbol for symbol, weight in info["weights"].items() if weight > 1e-9
        }
        assert nonzero <= {"A", "B"}

    def test_limit_down_exit_is_carried_until_later_execution(self):
        book = pd.DataFrame(
            {
                DATES[0]: pd.Series({"A": 0.5, "B": 0.5}),
                DATES[1]: pd.Series({"A": 1.0, "B": 0.0}),
                DATES[2]: pd.Series({"A": 1.0, "B": 0.0}),
                DATES[3]: pd.Series({"A": 1.0, "B": 0.0}),
                DATES[4]: pd.Series({"A": 1.0, "B": 0.0}),
            }
        ).T.fillna(0.0)
        env = PITPortfolioEnv(
            book,
            _preds(),
            _panel(limit_down={("B", DATES[2]): True}),
            PITPortfolioEnvConfig(max_book=4),
        )
        index_d1 = env.dates.index(DATES[1])
        index_d2 = env.dates.index(DATES[2])
        b_index = env.slot_symbols[index_d1].index("B")
        assert env.slot_no_decrease[index_d1, b_index]
        assert not env.slot_frozen[index_d1, b_index]
        assert "B" in env.slot_symbols[index_d2]


class TestExecutionConstraints:
    def test_limit_up_name_cannot_be_increased(self):
        env = _env(limit_up={("A", DATES[2]): True})
        env.reset()
        _, _, _, _, info0 = env.step(np.zeros(env.action_space.shape))
        weight_before = info0["weights"].get("A", 0.0)
        action = np.zeros(env.action_space.shape)
        action[env.slot_symbols[1].index("A")] = 1.0
        _, _, _, _, info1 = env.step(action)
        assert info1["weights"].get("A", 0.0) <= weight_before + 1e-12

    def test_limit_down_allows_increase_but_blocks_decrease(self):
        env = _env(limit_down={("A", DATES[2]): True})
        env.reset()
        _, _, _, _, info0 = env.step(np.zeros(env.action_space.shape))
        weight_before = info0["weights"]["A"]

        increase = np.zeros(env.action_space.shape)
        increase[env.slot_symbols[1].index("A")] = 1.0
        _, _, _, _, info1 = env.step(increase)
        assert info1["weights"]["A"] >= weight_before

        env = _env(limit_down={("A", DATES[2]): True})
        env.reset()
        _, _, _, _, info0 = env.step(np.zeros(env.action_space.shape))
        decrease = np.zeros(env.action_space.shape)
        decrease[env.slot_symbols[1].index("A")] = -1.0
        _, _, _, _, info1 = env.step(decrease)
        assert info1["weights"]["A"] >= info0["weights"]["A"] - 1e-12

    def test_suspended_bar_keeps_previous_weight(self):
        env = _env(suspended={("B", DATES[2]): True})
        env.reset()
        _, _, _, _, info0 = env.step(np.zeros(env.action_space.shape))
        weight_before = info0["weights"]["B"]
        action = np.full(env.action_space.shape, -1.0)
        _, _, _, _, info1 = env.step(action)
        assert info1["weights"]["B"] == pytest.approx(weight_before, rel=1e-9)

    def test_proven_suspension_without_bar_carries_last_close_and_freezes(self):
        panel = _panel()
        panel = panel[
            ~((panel["symbol"] == "B") & (panel["trade_date"] == DATES[2]))
        ].copy()
        gaps = pd.DataFrame(
            {
                "symbol": ["B"],
                "trade_date": [DATES[2]],
                "classification": ["SUSPENDED"],
            }
        )
        env = PITPortfolioEnv(
            _book(),
            _preds(),
            panel,
            PITPortfolioEnvConfig(max_book=4),
            session_gaps=gaps,
        )
        env.reset()
        _, _, _, _, info0 = env.step(np.zeros(env.action_space.shape))
        weight_before = info0["weights"]["B"]
        action = np.full(env.action_space.shape, -1.0)
        _, _, _, _, info1 = env.step(action)
        assert info1["weights"]["B"] == pytest.approx(weight_before, rel=1e-9)


class TestFailClosedEvidence:
    def test_missing_execution_flag_column_is_rejected(self):
        panel = _panel().drop(columns=["is_limit_down"])
        with pytest.raises(ValueError, match="missing required columns.*is_limit_down"):
            PITPortfolioEnv(
                _book(), _preds(), panel, PITPortfolioEnvConfig(max_book=4)
            )

    def test_missing_execution_flag_value_is_rejected(self):
        panel = _panel()
        panel.loc[
            (panel["symbol"] == "A") & (panel["trade_date"] == DATES[1]),
            "is_suspended",
        ] = None
        with pytest.raises(ValueError, match="missing is_suspended.*A"):
            PITPortfolioEnv(
                _book(), _preds(), panel, PITPortfolioEnvConfig(max_book=4)
            )

    def test_missing_reward_price_without_gap_evidence_is_rejected(self):
        panel = _panel()
        panel.loc[
            (panel["symbol"] == "A") & (panel["trade_date"] == DATES[2]),
            "close",
        ] = np.nan
        with pytest.raises(ValueError, match="without proven suspension.*A"):
            PITPortfolioEnv(
                _book(), _preds(), panel, PITPortfolioEnvConfig(max_book=4)
            )

    def test_unexplained_gap_is_not_treated_as_suspension(self):
        panel = _panel()
        panel = panel[
            ~((panel["symbol"] == "A") & (panel["trade_date"] == DATES[1]))
        ].copy()
        gaps = pd.DataFrame(
            {
                "symbol": ["A"],
                "trade_date": [DATES[1]],
                "classification": ["MISSING_UNEXPLAINED"],
            }
        )
        with pytest.raises(ValueError, match="MISSING_UNEXPLAINED"):
            PITPortfolioEnv(
                _book(),
                _preds(),
                panel,
                PITPortfolioEnvConfig(max_book=4),
                session_gaps=gaps,
            )

    def test_duplicate_market_row_is_rejected(self):
        panel = _panel()
        panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate execution rows"):
            PITPortfolioEnv(
                _book(), _preds(), panel, PITPortfolioEnvConfig(max_book=4)
            )


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
