"""Controlled comparison of RL reward variants on one fixed environment.

This exists because "we added a risk term" is not a result. The round-22 brief
is explicit: a reward change may only be recommended if it is *measured* to
help, and "no incremental value, do not enable" is a publishable outcome.

What makes the comparison honest:

* Every arm is scored by the **same** evaluation environment, always built with
  ``drawdown_lambda = volatility_lambda = 0``. Training reward is a property of
  the arm; the yardstick is not. Scoring a drawdown-penalised policy with a
  drawdown-penalised metric would be circular.
* The controls include ``zero`` (the passive book itself, which by construction
  scores exactly zero value-add) and ``random`` (an untrained policy network).
  A trained arm that cannot beat both has not demonstrated anything.
* Every arm runs over the same seed list, and the report carries mean **and**
  dispersion. A single seed is not evidence.
* Metrics are read off the environment's own info stream, so all arms share one
  accounting path.

The drawdown reported here is the environment's internal NAV drawdown --
``prod(1 + <w,r> - cost)`` over consecutive ``close(T+1) -> close(T+2)``
intervals. It excludes slippage, lot rounding and cash constraints, so it is a
*proxy* for the strict-simulator drawdown the acceptance gate actually reads.
The proxy is identical across arms, which is what makes the ranking meaningful,
but an absolute number from here is not a gate result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig

#: Trading sessions per year, used only to annualise for the Calmar ratio.
SESSIONS_PER_YEAR = 244.0


@dataclass(frozen=True)
class RewardArm:
    """One training configuration under test."""

    name: str
    drawdown_lambda: float = 0.0
    volatility_lambda: float = 0.0
    #: ``zero`` and ``random`` are controls and are never trained.
    kind: str = "trained"

    def env_config(self, base: PITPortfolioEnvConfig) -> PITPortfolioEnvConfig:
        from dataclasses import replace

        return replace(
            base,
            drawdown_lambda=self.drawdown_lambda,
            volatility_lambda=self.volatility_lambda,
        )


@dataclass
class EpisodeMetrics:
    """Everything the round-22 verdict needs, from one deterministic rollout."""

    steps: int
    cumulative_value_add: float
    nav: float
    nav_passive: float
    max_drawdown: float
    max_drawdown_passive: float
    excess_max_drawdown: float
    calmar: float | None
    calmar_passive: float | None
    mean_turnover: float
    mean_turnover_passive: float

    def as_row(self) -> dict[str, float | int | None]:
        return dict(self.__dict__)


def evaluate_episode(env: PITPortfolioEnv, policy) -> EpisodeMetrics:
    """Roll ``policy`` once through ``env`` and reduce the info stream.

    ``policy`` maps an observation to an action. It is called once per step and
    must be deterministic for the result to be reproducible.
    """
    observation, _ = env.reset(seed=0)
    value_add: list[float] = []
    turnover: list[float] = []
    turnover_passive: list[float] = []
    info: dict = {}
    done = False
    while not done:
        action = policy(observation)
        observation, _, terminated, truncated, info = env.step(action)
        value_add.append(float(info["value_add"]))
        turnover.append(float(info["turnover_policy"]))
        turnover_passive.append(float(info["turnover_passive"]))
        done = terminated or truncated
    if not info:
        raise RuntimeError("evaluation environment produced no steps")

    steps = len(value_add)
    max_drawdown = float(info["max_drawdown_policy"])
    max_drawdown_passive = float(info["max_drawdown_passive"])
    return EpisodeMetrics(
        steps=steps,
        cumulative_value_add=float(np.sum(value_add)),
        nav=float(info["nav"]),
        nav_passive=float(info["nav_passive"]),
        max_drawdown=max_drawdown,
        max_drawdown_passive=max_drawdown_passive,
        excess_max_drawdown=max_drawdown - max_drawdown_passive,
        calmar=_calmar(float(info["nav"]), steps, max_drawdown),
        calmar_passive=_calmar(
            float(info["nav_passive"]), steps, max_drawdown_passive
        ),
        mean_turnover=float(np.mean(turnover)),
        mean_turnover_passive=float(np.mean(turnover_passive)),
    )


def _calmar(nav: float, steps: int, max_drawdown: float) -> float | None:
    """Annualised return over max drawdown.

    Returns ``None`` -- never ``0.0`` and never a sentinel -- when the drawdown
    is zero, because "never drew down" is an undefined Calmar, not an infinite
    one, and a downstream mean over a fabricated number would be silently wrong.
    """
    if steps <= 0 or nav <= 0.0 or max_drawdown <= 1e-12:
        return None
    annualised = nav ** (SESSIONS_PER_YEAR / steps) - 1.0
    if not math.isfinite(annualised):
        return None
    return annualised / max_drawdown


def zero_policy(action_dim: int):
    """The control that must score exactly zero value-add by construction."""

    def _policy(_observation):
        return np.zeros(action_dim, dtype=np.float64)

    return _policy


def random_policy(action_dim: int, seed: int):
    """Untrained uniform actions: the null a trained arm has to clear."""
    rng = np.random.default_rng(seed)

    def _policy(_observation):
        return rng.uniform(-1.0, 1.0, size=action_dim)

    return _policy


@dataclass(frozen=True)
class AblationConfig:
    timesteps: int = 200_000
    n_envs: int = 4
    device: str = "cpu"
    seeds: tuple[int, ...] = (1729, 20260819, 7, 13, 42)
    base_env: PITPortfolioEnvConfig = field(
        default_factory=PITPortfolioEnvConfig
    )
    train_reward_end_limit: str | None = None
    eval_reward_end_limit: str | None = None


def _make_env(
    book: pd.DataFrame,
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    config: PITPortfolioEnvConfig,
    session_gaps: pd.DataFrame | None,
) -> PITPortfolioEnv:
    return PITPortfolioEnv(
        book, predictions, panel, config, session_gaps=session_gaps
    )


def run_ablation(
    *,
    train_book: pd.DataFrame,
    test_book: pd.DataFrame,
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    arms: list[RewardArm],
    config: AblationConfig | None = None,
    session_gaps: pd.DataFrame | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Train and score every ``arm`` on every seed; return one row per run.

    The returned frame is deliberately long/tidy rather than pre-aggregated:
    the dispersion across seeds is part of the evidence, and a caller that only
    ever sees a mean cannot tell a real effect from seed noise.
    """
    from dataclasses import replace
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg = config or AblationConfig()
    train_base = replace(
        cfg.base_env, reward_end_date_limit=cfg.train_reward_end_limit
    )
    # The yardstick never carries a risk term: see module docstring.
    eval_config = replace(
        cfg.base_env,
        reward_end_date_limit=cfg.eval_reward_end_limit,
        drawdown_lambda=0.0,
        volatility_lambda=0.0,
    )

    def build_eval() -> PITPortfolioEnv:
        return _make_env(
            test_book, predictions, panel, eval_config, session_gaps
        )

    probe = build_eval()
    action_dim = int(probe.action_space.shape[0])

    rows: list[dict] = []
    for arm in arms:
        for seed in cfg.seeds:
            if arm.kind == "zero":
                policy = zero_policy(action_dim)
            elif arm.kind == "random":
                policy = random_policy(action_dim, seed)
            elif arm.kind == "trained":
                train_config = arm.env_config(train_base)

                def factory(rank: int):
                    def _make():
                        env = _make_env(
                            train_book,
                            predictions,
                            panel,
                            train_config,
                            session_gaps,
                        )
                        env.reset(seed=seed + rank)
                        return env

                    return _make

                # In-process vectorisation on purpose. SubprocVecEnv would
                # have to pickle the full market panel into every worker; on a
                # real A-share panel that is a multi-million-row frame per
                # process and the transfer fails outright. The environment's
                # own step is a 40-wide dot product, so the policy forward pass
                # dominates and batching in-process still gives PPO its speedup.
                vec = DummyVecEnv([factory(i) for i in range(cfg.n_envs)])
                model = PPO("MlpPolicy", vec, device=cfg.device, seed=seed)
                model.learn(total_timesteps=int(cfg.timesteps), progress_bar=False)
                vec.close()

                def policy(observation, _model=model):
                    action, _ = _model.predict(observation, deterministic=True)
                    return action

            else:
                raise ValueError(f"unknown arm kind {arm.kind!r}")

            metrics = evaluate_episode(build_eval(), policy)
            row = {"arm": arm.name, "kind": arm.kind, "seed": int(seed)}
            row.update(metrics.as_row())
            rows.append(row)
            if progress:
                print(
                    f"  {arm.name:<28} seed={seed:<9} "
                    f"vadd={row['cumulative_value_add']:+.4f} "
                    f"mdd={row['max_drawdown']:.4f} "
                    f"exMdd={row['excess_max_drawdown']:+.4f} "
                    f"turn={row['mean_turnover']:.3f}",
                    flush=True,
                )

    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and dispersion per arm.

    ``None`` Calmar values are excluded from the mean rather than coerced to
    zero, and the count of contributing runs is carried so a mean over one
    surviving seed cannot be mistaken for a mean over five.
    """
    metrics = [
        "cumulative_value_add",
        "nav",
        "max_drawdown",
        "excess_max_drawdown",
        "calmar",
        "mean_turnover",
    ]
    out: list[dict] = []
    for arm, group in runs.groupby("arm", sort=False):
        row: dict[str, object] = {"arm": arm, "n_seeds": int(len(group))}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = (
                float(values.mean()) if len(values) else None
            )
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None
            )
            row[f"{metric}_n"] = int(len(values))
        out.append(row)
    return pd.DataFrame(out)


__all__ = [
    "SESSIONS_PER_YEAR",
    "AblationConfig",
    "EpisodeMetrics",
    "RewardArm",
    "evaluate_episode",
    "random_policy",
    "run_ablation",
    "summarise",
    "zero_policy",
]
