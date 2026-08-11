"""PPO training entry point for the governed PIT portfolio overlay.

The historical generic trainer instantiated ``PortfolioEnv``, whose fixed
whole-window top-80 universe was later proven to contain look-ahead.  That path
is intentionally unavailable.  Training now requires a signal-dated PIT book
and an explicit global market-session calendar and instantiates only
:class:`PITPortfolioEnv`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.cuda_runtime import (
    configure_cuda_environment,
    cuda_runtime_probe,
    format_cuda_diagnostic,
)
from quantagent.rl.pit_portfolio_env import (
    PITPortfolioEnv,
    PITPortfolioEnvConfig,
    RL_REWARD_SEMANTICS,
)

configure_cuda_environment()


@dataclass(frozen=True)
class PPOTrainingConfig:
    timesteps: int = 5_000_000
    device: str = "cuda"
    n_envs: int = 8
    output_dir: str = field(
        default_factory=lambda: str(quant_paths().models / "v7_rl_policy")
    )
    tensorboard_log: str = field(
        default_factory=lambda: str(quant_paths().logs / "tb" / "rl")
    )
    env: PITPortfolioEnvConfig = field(default_factory=PITPortfolioEnvConfig)
    seed: int = 1729
    require_gpu: bool = True


def train_ppo_policy(
    predictions: pd.DataFrame,
    market_panel: pd.DataFrame,
    config: PPOTrainingConfig | None = None,
    *,
    book_weights: pd.DataFrame | None = None,
    market_sessions: Iterable[object] | None = None,
) -> dict[str, object]:
    """Train PPO only inside the governed PIT hold-band environment.

    ``book_weights`` must be indexed by *signal date*.  It must not already be
    delayed to the execution date; the environment reward and strict simulator
    are the only owners of the T-close -> T+1 execution mapping.

    The explicit ``market_sessions`` input prevents a per-symbol next-row clock
    and makes the reward schedule auditable.  Missing either input is a hard
    failure rather than a fallback to the rejected legacy environment.
    """

    if book_weights is None:
        raise RuntimeError(
            "legacy RL training is quarantined: train_ppo_policy requires explicit "
            "signal-dated book_weights; the whole-window top-80 PortfolioEnv must not be used"
        )
    if market_sessions is None:
        raise RuntimeError(
            "governed RL training requires explicit global market_sessions"
        )

    cfg = config or PPOTrainingConfig()
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("PIT PPO training requires torch and stable_baselines3") from exc

    if cfg.require_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "RL GPU training was required, but torch.cuda.is_available() is false. "
            + format_cuda_diagnostic(cuda_runtime_probe(torch))
        )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.tensorboard_log).mkdir(parents=True, exist_ok=True)
    sessions = tuple(market_sessions)

    def make_env(rank: int):
        def _factory():
            env = PITPortfolioEnv(
                book_weights,
                predictions,
                market_panel,
                sessions,
                cfg.env,
            )
            env.reset(seed=cfg.seed + rank)
            return env

        return _factory

    if cfg.n_envs > 1:
        vec_env = SubprocVecEnv([make_env(i) for i in range(cfg.n_envs)])
    else:
        vec_env = DummyVecEnv([make_env(0)])

    probe_env = PITPortfolioEnv(
        book_weights,
        predictions,
        market_panel,
        sessions,
        cfg.env,
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        device=cfg.device,
        tensorboard_log=cfg.tensorboard_log,
        seed=cfg.seed,
    )
    model.learn(total_timesteps=int(cfg.timesteps), progress_bar=False)
    policy_path = output_dir / "policy.zip"
    model.save(policy_path)
    summary = {
        "status": "passed_research_training",
        "researchOnly": True,
        "productionEligible": False,
        "policy_path": str(policy_path),
        "timesteps": int(cfg.timesteps),
        "device": cfg.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "reward_semantics": RL_REWARD_SEMANTICS,
        "market_session_schedule_sha256": probe_env.market_session_schedule_sha256,
        "environment_dispersion": probe_env.book_dispersion_report(),
        "config": asdict(cfg),
        "productionBlockers": [
            "training success is not OOS economic certification",
            "policy must be re-simulated through strict A-share execution",
            "independent governed holdout and Stage-4 promotion evidence are required",
        ],
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    vec_env.close()
    return summary


__all__ = ["PPOTrainingConfig", "train_ppo_policy"]
