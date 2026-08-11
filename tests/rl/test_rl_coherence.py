"""RL adaptability + coherence under the governed executable-session clock.

The alpha signal must flow into incremental reward, while a flat within-book
signal must be detected as non-selective.  Reward is never same-day: actions are
formed at close(T), execute on global session T+1, and are marked to T+2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gym = pytest.importorskip("gymnasium")

from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig

DATES = pd.date_range("2026-01-05", periods=10, freq="B")
SYMS = ["A", "B", "C", "D"]
_DAILY_RET = {"A": 0.020, "B": 0.015, "C": 0.010, "D": 0.005}


def _panel() -> pd.DataFrame:
    rows = []
    for symbol in SYMS:
        price = 10.0
        for date in DATES:
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": date,
                    "close": price,
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "is_suspended": False,
                    "is_st": False,
                }
            )
            price *= 1.0 + _DAILY_RET[symbol]
    return pd.DataFrame(rows)


def _book() -> pd.DataFrame:
    rows = {date: pd.Series(0.25, index=SYMS) for date in DATES[:-2]}
    return pd.DataFrame(rows).T


def _preds(dispersed: bool) -> pd.DataFrame:
    rows = []
    for date in DATES:
        for i, symbol in enumerate(SYMS):
            score = float(len(SYMS) - i) if dispersed else 1.0
            rows.append({"symbol": symbol, "trade_date": date, "alpha_score": score})
    return pd.DataFrame(rows)


def _env(dispersed: bool = True, **cfg) -> PITPortfolioEnv:
    return PITPortfolioEnv(
        _book(),
        _preds(dispersed),
        _panel(),
        DATES,
        PITPortfolioEnvConfig(max_book=4, cost_bps=2.0, **cfg),
    )


def test_alpha_signal_flows_through_env_to_reward():
    env = _env(dispersed=True)
    obs, _ = env.reset()
    assert env.slot_symbols[0][0] == "A"
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1.0
    value_add = 0.0
    done = False
    while not done:
        obs, reward, done, _, info = env.step(action)
        value_add += info["value_add"]
    assert value_add > 0, f"alpha tilt did not earn value-add: {value_add:.6f}"


def test_dispersion_guard_detects_dispersed_book():
    report = _env(dispersed=True).book_dispersion_report()
    assert report["n_dates"] >= 3
    assert report["env_can_select"] is True
    assert report["mean_within_book_alpha_std"] > 0.1
    assert report["flat_date_fraction"] < 0.5


def test_dispersion_guard_flags_flat_book():
    report = _env(dispersed=False).book_dispersion_report()
    assert report["env_can_select"] is False
    assert report["flat_date_fraction"] == pytest.approx(1.0)
    assert report["mean_within_book_alpha_std"] == pytest.approx(0.0, abs=1e-9)
