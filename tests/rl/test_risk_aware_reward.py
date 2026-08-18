"""Round 22 / R11: the reward must price drawdown, and must still pay zero for nothing.

Round 21 recorded the gap: ``PITPortfolioEnv``'s reward is linear in the return
difference and carries *no* variance, Sharpe or drawdown term, while the
deprecated ``PortfolioEnv`` it replaced carried ``drawdown_lambda=2.0``.  The
risk penalty was dropped, undocumented, when the reward clock was corrected, so
the training objective encoded only half of the stated goal ("highest excess
return, smallest drawdown") and was *not* isomorphic to the ``AGENTS.md``
max-drawdown acceptance gate: a policy could win on reward and be rejected by
the gate.

Two things have to hold simultaneously, and they pull against each other:

1. The penalty must have teeth -- it must actually charge a path that draws
   down harder than the passive book.
2. It must not break "a zero action earns exactly zero", the construction that
   immunises this environment against the env-flat trap (round 21 DEF-036).
   That is why both risk terms are *differences against the same constrained
   passive book*, exactly as the return term already is.

The telescoping test is the load-bearing one.  ``-λ·ΔMDD`` is algebraically a
potential-based shaping term (Ng, Harada & Russell 1999) with ``Φ(s) = -λ·MDD(s)``
and ``γ = 1``; had it satisfied the theorem's episodic precondition
``Φ(terminal) = 0`` it would be policy-*invariant*, i.e. a decorative penalty
that can never change what the agent does.  It deliberately does not: the
telescoped sum is ``-λ·MDD_T``, a policy-dependent quantity, and that is what
makes the episode return isomorphic to the acceptance gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

gym = pytest.importorskip("gymnasium")

from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig

RISK_DATES = pd.date_range("2026-01-05", periods=12, freq="B")
RISK_SYMS = ("SHOCK", "STEADY")

# SHOCK crashes ~18% over two sessions and then claws back; STEADY drifts.
# A book holding both equally therefore has a real, non-degenerate drawdown,
# and tilting toward or away from SHOCK moves the drawdown in a known
# direction without changing the accounting of either leg.
_CLOSES = {
    "SHOCK": [10.0, 10.0, 10.0, 9.4, 8.2, 8.6, 9.1, 9.6, 10.0, 10.2, 10.3, 10.4],
    "STEADY": [20.0, 20.02, 20.04, 20.06, 20.08, 20.10, 20.12, 20.14, 20.16, 20.18, 20.20, 20.22],
}


def _risk_panel() -> pd.DataFrame:
    rows = []
    for symbol in RISK_SYMS:
        for trade_date, close in zip(RISK_DATES, _CLOSES[symbol]):
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "close": float(close),
                    "is_limit_up": False,
                    "is_limit_down": False,
                    "is_suspended": False,
                }
            )
    return pd.DataFrame(rows)


def _risk_preds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "alpha_score": 2.0 - index,
            }
            for trade_date in RISK_DATES
            for index, symbol in enumerate(RISK_SYMS)
        ]
    )


def _risk_book(gross: float = 1.0) -> pd.DataFrame:
    rows = {
        trade_date: pd.Series(gross / 2.0, index=list(RISK_SYMS))
        for trade_date in RISK_DATES[:-2]
    }
    return pd.DataFrame(rows).T.fillna(0.0)


def _risk_env(*, gross: float = 1.0, **cfg) -> PITPortfolioEnv:
    return PITPortfolioEnv(
        _risk_book(gross),
        _risk_preds(),
        _risk_panel(),
        PITPortfolioEnvConfig(max_book=2, **cfg),
    )


def _rollout(env: PITPortfolioEnv, *, tilt_symbol: str, tilt: float) -> dict:
    """Roll a fixed tilt on one name and collect the episode's accounting."""
    env.reset()
    n = env.config.max_book
    rewards: list[float] = []
    infos: list[dict] = []
    done = False
    while not done:
        action = np.zeros(n + 1, dtype=np.float64)
        symbols = env.slot_symbols[env._t]
        if tilt_symbol in symbols:
            action[symbols.index(tilt_symbol)] = tilt
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        infos.append(info)
        done = terminated or truncated
    return {
        "rewards": rewards,
        "infos": infos,
        "episode_return": float(np.sum(rewards)),
        "value_add": float(sum(info["value_add"] for info in infos)),
        "max_drawdown": float(infos[-1]["max_drawdown_policy"]),
        "max_drawdown_passive": float(infos[-1]["max_drawdown_passive"]),
        "drawdown_penalty": float(sum(i["drawdown_penalty"] for i in infos)),
        "volatility_penalty": float(sum(i["volatility_penalty"] for i in infos)),
    }


# --------------------------------------------------------------------------
# 1. The zero-action guarantee must survive the risk terms.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("passive_gross", [1.0, 0.8, 0.5, 0.30, 0.10])
@pytest.mark.parametrize(
    ("drawdown_lambda", "volatility_lambda"),
    [(1.0, 0.0), (0.0, 25.0), (2.0, 25.0), (50.0, 500.0)],
)
def test_zero_action_still_earns_exactly_zero_with_risk_terms_on(
    passive_gross: float, drawdown_lambda: float, volatility_lambda: float
) -> None:
    env = _risk_env(
        gross=passive_gross,
        drawdown_lambda=drawdown_lambda,
        volatility_lambda=volatility_lambda,
    )
    result = _rollout(env, tilt_symbol="SHOCK", tilt=0.0)

    assert result["rewards"], "environment produced no steps"
    # The passive book itself draws down here; that is the point. An absolute
    # risk penalty would charge the idle agent for it.
    # The drawdown scales with how much of the book is actually invested, so
    # the guard is relative to gross rather than absolute.
    assert result["max_drawdown_passive"] > 0.05 * passive_gross, (
        "fixture no longer exercises a drawdown; the test would pass vacuously"
    )
    for step, reward in enumerate(result["rewards"]):
        assert reward == pytest.approx(0.0, abs=1e-12), (
            f"zero action earned {reward} at step {step} with "
            f"drawdown_lambda={drawdown_lambda}, "
            f"volatility_lambda={volatility_lambda}: the risk term is charging "
            "the agent for the passive book's own risk"
        )
    assert result["drawdown_penalty"] == pytest.approx(0.0, abs=1e-12)
    assert result["volatility_penalty"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# 2. The penalty must have teeth: it must charge a deeper path.
# --------------------------------------------------------------------------


def test_drawdown_penalty_charges_the_deeper_path() -> None:
    """Tilting into the crashing name must cost more once the term is on."""
    unpenalised = _rollout(_risk_env(), tilt_symbol="SHOCK", tilt=0.9)
    penalised = _rollout(
        _risk_env(drawdown_lambda=2.0), tilt_symbol="SHOCK", tilt=0.9
    )

    assert penalised["max_drawdown"] > penalised["max_drawdown_passive"], (
        "fixture does not produce an excess drawdown to charge for"
    )
    assert penalised["drawdown_penalty"] > 0.0
    assert penalised["episode_return"] < unpenalised["episode_return"], (
        "turning drawdown_lambda on did not reduce the reward of a path that "
        "draws down harder than the passive book"
    )
    # The return leg is untouched: only the penalty moved.
    assert penalised["value_add"] == pytest.approx(unpenalised["value_add"], rel=1e-12)


def test_drawdown_penalty_credits_the_shallower_path() -> None:
    """The term is symmetric: drawing down *less* than the book is rewarded."""
    unpenalised = _rollout(_risk_env(), tilt_symbol="SHOCK", tilt=-0.9)
    penalised = _rollout(
        _risk_env(drawdown_lambda=2.0), tilt_symbol="SHOCK", tilt=-0.9
    )

    assert penalised["max_drawdown"] < penalised["max_drawdown_passive"], (
        "fixture does not produce a shallower-than-passive path"
    )
    assert penalised["episode_return"] > unpenalised["episode_return"]


def test_penalty_ranks_the_deep_path_below_the_shallow_one() -> None:
    """With the term on, the ordering the acceptance gate cares about appears."""
    config = dict(drawdown_lambda=2.0)
    deep = _rollout(_risk_env(**config), tilt_symbol="SHOCK", tilt=0.9)
    shallow = _rollout(_risk_env(**config), tilt_symbol="SHOCK", tilt=-0.9)

    assert deep["max_drawdown"] > shallow["max_drawdown"]
    assert deep["drawdown_penalty"] > shallow["drawdown_penalty"]


# --------------------------------------------------------------------------
# 3. Isomorphism with the acceptance gate, and non-invariance.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tilt", [0.9, 0.4, -0.4, -0.9])
def test_drawdown_penalty_telescopes_to_excess_max_drawdown(tilt: float) -> None:
    """Σ penalty == λ·(MDD_policy − MDD_passive): the episode return *is* the gate."""
    lam = 3.0
    result = _rollout(_risk_env(drawdown_lambda=lam), tilt_symbol="SHOCK", tilt=tilt)

    expected = lam * (result["max_drawdown"] - result["max_drawdown_passive"])
    assert result["drawdown_penalty"] == pytest.approx(expected, abs=1e-12), (
        "the per-step drawdown increments do not telescope to the episode's "
        "excess max drawdown, so the episode return is not isomorphic to the "
        "AGENTS.md max-drawdown gate"
    )


def test_drawdown_shaping_is_not_policy_invariant() -> None:
    """Ng-Harada-Russell would make this inert if Φ(terminal) were forced to 0.

    The cumulative shaping over an episode is ``Φ(s_T) − Φ(s_0) = −λ·MDD_T``.
    If that were zero the penalty could not change any policy's ranking. It is
    not zero, and it differs *between* policies -- which is exactly the
    precondition violation that gives the term teeth.
    """
    lam = 3.0
    deep = _rollout(_risk_env(drawdown_lambda=lam), tilt_symbol="SHOCK", tilt=0.9)
    shallow = _rollout(_risk_env(drawdown_lambda=lam), tilt_symbol="SHOCK", tilt=-0.9)

    assert deep["drawdown_penalty"] != pytest.approx(0.0, abs=1e-9)
    assert deep["drawdown_penalty"] != pytest.approx(
        shallow["drawdown_penalty"], abs=1e-9
    ), "the shaping term is policy-invariant and therefore decorative"


# --------------------------------------------------------------------------
# 4. The downside term charges downside only.
# --------------------------------------------------------------------------


def test_volatility_penalty_is_downside_only() -> None:
    """Every charged step must be one where the policy's own net return was negative."""
    result = _rollout(
        _risk_env(volatility_lambda=50.0), tilt_symbol="SHOCK", tilt=0.9
    )

    charged = [
        info for info in result["infos"] if info["volatility_penalty"] > 1e-15
    ]
    assert charged, "fixture produced no downside step to charge"
    for info in charged:
        assert info["net_policy"] < 0.0, (
            "a positive-return step was charged a downside-risk penalty"
        )
    # An upside-dominated step where the policy beat the passive book upward
    # must never be charged.
    for info in result["infos"]:
        if info["net_policy"] >= 0.0:
            assert info["volatility_penalty"] <= 0.0


# --------------------------------------------------------------------------
# 5. Defaults, and the refusal to accept a sign-flipped lambda.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tilt", [0.9, 0.0, -0.9])
def test_default_config_reproduces_the_previous_reward_exactly(tilt: float) -> None:
    """Turning risk on is a caller's decision; the default must change nothing."""
    result = _rollout(_risk_env(), tilt_symbol="SHOCK", tilt=tilt)

    assert PITPortfolioEnvConfig().drawdown_lambda == 0.0
    assert PITPortfolioEnvConfig().volatility_lambda == 0.0
    for info, reward in zip(result["infos"], result["rewards"]):
        assert info["risk_penalty"] == 0.0
        assert reward == pytest.approx(info["value_add"] * 100.0, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize(
    "cfg",
    [
        {"drawdown_lambda": -1.0},
        {"volatility_lambda": -1e-9},
        {"drawdown_lambda": float("nan")},
        {"volatility_lambda": float("inf")},
    ],
)
def test_negative_or_non_finite_lambda_is_rejected(cfg: dict) -> None:
    """A negative lambda would pay the agent for drawing down harder."""
    with pytest.raises(ValueError, match="must be finite and >= 0"):
        _risk_env(**cfg)


def test_risk_accounting_is_reset_between_episodes() -> None:
    """A stale peak would leak one episode's drawdown into the next one's reward."""
    env = _risk_env(drawdown_lambda=2.0)
    first = _rollout(env, tilt_symbol="SHOCK", tilt=0.9)
    second = _rollout(env, tilt_symbol="SHOCK", tilt=0.9)

    assert first["max_drawdown"] == pytest.approx(second["max_drawdown"], abs=1e-12)
    assert first["drawdown_penalty"] == pytest.approx(
        second["drawdown_penalty"], abs=1e-12
    )
    assert first["episode_return"] == pytest.approx(
        second["episode_return"], abs=1e-12
    )
