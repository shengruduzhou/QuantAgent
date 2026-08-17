"""A zero action must earn exactly zero value-add, at any passive gross.

Round 21 / R11 (RL) finding.  The reward is a difference against the passive
book, ``R = κ · [(⟨w,r⟩ − c_policy) − (⟨w_b,r⟩ − c_passive)]``, so an agent that
does nothing should score exactly 0.0.  That construction is what immunises the
environment against the env-flat trap, where a rising market pays an agent for
holding still.

It leaked: the gross target was clipped into a fixed ``[min_gross, max_gross]``
band, so whenever the passive book's own gross sat below ``min_gross`` a zero
action still produced ``w = passive · (min_gross / passive_gross) ≠ passive``.
The environment levered the book up on its own and credited the difference to
the agent.  A low-gross passive book is not exotic — it is exactly what a
de-risked or partially-cash book looks like, i.e. the regime where an honest
value-add measurement matters most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig

from tests.rl.test_pit_portfolio_env import DATES, SYMS, _panel, _preds


def _book_with_gross(gross: float) -> pd.DataFrame:
    """Passive book whose weights sum to ``gross`` on every signal date."""
    rows = {}
    for i, trade_date in enumerate(DATES[:-1]):
        held = ["A", "B"] if i < 4 else ["A", "C"]
        rows[trade_date] = pd.Series(gross / len(held), index=held)
    return pd.DataFrame(rows).T.fillna(0.0)


def _env_with_gross(gross: float, **cfg) -> PITPortfolioEnv:
    return PITPortfolioEnv(
        _book_with_gross(gross),
        _preds(),
        _panel(),
        PITPortfolioEnvConfig(max_book=4, **cfg),
    )


@pytest.mark.parametrize("passive_gross", [1.0, 0.8, 0.5, 0.30, 0.10])
def test_zero_action_is_zero_reward_at_every_passive_gross(passive_gross: float) -> None:
    env = _env_with_gross(passive_gross)
    env.reset()
    zero = np.zeros(env.action_space.shape, dtype=np.float64)

    rewards = []
    done = False
    while not done:
        _, reward, terminated, truncated, _ = env.step(zero)
        rewards.append(float(reward))
        done = terminated or truncated

    assert rewards, "environment produced no steps"
    for step, reward in enumerate(rewards):
        assert reward == pytest.approx(0.0, abs=1e-9), (
            f"zero action earned {reward} at step {step} with passive gross "
            f"{passive_gross}: the environment moved the book by itself"
        )


def test_low_gross_book_is_not_levered_up_to_the_floor() -> None:
    """The specific mechanism: gross floor must not bind above the passive book."""
    passive_gross = 0.20
    config = PITPortfolioEnvConfig(max_book=4, min_gross=0.5)
    env = PITPortfolioEnv(
        _book_with_gross(passive_gross), _preds(), _panel(), config
    )
    env.reset()

    env.step(np.zeros(env.action_space.shape, dtype=np.float64))
    held = sum(env._prev_w.values())

    assert held == pytest.approx(passive_gross, abs=1e-9), (
        f"a zero action left gross at {held}, not the passive {passive_gross}"
    )


def test_a_nonzero_action_can_still_move_gross() -> None:
    """The fix must not freeze the cash tilt — only stop it acting unasked."""
    env = _env_with_gross(0.8)
    env.reset()

    action = np.zeros(env.action_space.shape, dtype=np.float64)
    action[-1] = -1.0  # ask for less gross
    env.step(action)

    assert sum(env._prev_w.values()) < 0.8
