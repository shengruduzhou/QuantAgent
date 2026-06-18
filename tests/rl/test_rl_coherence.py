"""RL adaptability + coherence (Phase 2).

Two properties matter for the factors -> model -> predictions -> RL chain to be
trustworthy:

1. The alpha the model emits (and which the factor loop ultimately feeds) must
   actually flow through the env into reward: a policy that tilts toward
   high-alpha names should earn positive value-add when those names truly have
   higher forward returns. If it does not, the env is not consuming predictions
   coherently.
2. The env-flat guard (`book_dispersion_report`) must distinguish a book with
   real within-book alpha dispersion (the policy can select) from a flat book
   (the policy cannot, so any value-add is a gross/cash artifact).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gym = pytest.importorskip("gymnasium")

from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig

DATES = pd.date_range("2026-01-05", periods=9, freq="B")
SYMS = ["A", "B", "C", "D"]
# Deterministic daily return by rank: A best, D worst.
_DAILY_RET = {"A": 0.020, "B": 0.015, "C": 0.010, "D": 0.005}


def _panel() -> pd.DataFrame:
    rows = []
    for s in SYMS:
        px = 10.0
        for d in DATES:
            rows.append({"symbol": s, "trade_date": d, "close": px,
                         "is_limit_up": False, "is_limit_down": False, "is_suspended": False})
            px *= 1.0 + _DAILY_RET[s]
    return pd.DataFrame(rows)


def _book() -> pd.DataFrame:
    # Equal-weight hold of all four names every date (gross = 1.0).
    rows = {d: pd.Series(0.25, index=SYMS) for d in DATES[:-1]}
    return pd.DataFrame(rows).T


def _preds(dispersed: bool) -> pd.DataFrame:
    rows = []
    for d in DATES:
        for i, s in enumerate(SYMS):
            # Dispersed: alpha ranks A>B>C>D (matches forward returns).
            # Flat: identical alpha -> no within-book dispersion to tilt on.
            score = float(len(SYMS) - i) if dispersed else 1.0
            rows.append({"symbol": s, "trade_date": d, "alpha_score": score})
    return pd.DataFrame(rows)


def _env(dispersed: bool = True, **cfg) -> PITPortfolioEnv:
    return PITPortfolioEnv(_book(), _preds(dispersed), _panel(),
                           PITPortfolioEnvConfig(max_book=4, cost_bps=2.0, **cfg))


def test_alpha_signal_flows_through_env_to_reward():
    """Tilting toward the top-alpha slot earns positive cumulative value-add
    when alpha truly predicts forward returns — the chain is coherent."""
    env = _env(dispersed=True)
    obs, _ = env.reset()
    # Slot 0 is the highest-alpha name (env ranks slots by alpha desc).
    assert env.slot_symbols[0][0] == "A"
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1.0  # overweight the top-alpha name
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
    """Identical predictions -> no within-book dispersion -> env cannot select,
    so a positive value-add would be a gross/cash artifact (the +39pp risk)."""
    report = _env(dispersed=False).book_dispersion_report()
    assert report["env_can_select"] is False
    assert report["flat_date_fraction"] == pytest.approx(1.0)
    assert report["mean_within_book_alpha_std"] == pytest.approx(0.0, abs=1e-9)
