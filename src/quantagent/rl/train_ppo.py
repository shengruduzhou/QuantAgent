"""PPO training entry point for the V7 portfolio environment.

**Currently fails closed.** This trainer builds :class:`PortfolioEnv`, whose
reward is the ``close(T) -> close(T+1)`` return on the *signal* date -- a return
no policy could have captured, because it cannot trade at ``close(T)`` on
information contained in ``close(T)``. Verified numerically: on a panel where
the two candidate intervals differ, this environment reports ``[0.0, 0.10]``
where the executable interval gives ``[0.20, 0.0]``.

Constructing that environment now requires
``PortfolioEnvConfig(acknowledge_untradable_reward=True)``, which this module
deliberately does NOT set. Training here therefore raises until the trainer is
rewired to :class:`quantagent.rl.pit_portfolio_env.PITPortfolioEnv`, which
signals at ``close(T)``, executes at ``close(T+1)``, rewards over
``close(T+1) -> close(T+2)`` and fails closed on missing prices.

That rewiring is not a rename: ``PITPortfolioEnv`` also takes a passive/benchmark
weight book, so the call sites in ``cli/v7_train.py`` and
``scripts/rl_train_eval_2026.py`` must decide what that book is. Until someone
makes that decision explicitly, a loud failure is the correct behaviour -- any
policy trained here optimised an objective that does not exist, so its results
must not be cited.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.rl.portfolio_env import PortfolioEnv, PortfolioEnvConfig
from quantagent.cuda_runtime import configure_cuda_environment, cuda_runtime_probe, format_cuda_diagnostic

configure_cuda_environment()


@dataclass(frozen=True)
class PPOTrainingConfig:
    timesteps: int = 5_000_000
    device: str = "cuda"
    n_envs: int = 8
    output_dir: str = field(default_factory=lambda: str(quant_paths().models / "v7_rl_policy"))
    tensorboard_log: str = field(default_factory=lambda: str(quant_paths().logs / "tb" / "rl"))
    env: PortfolioEnvConfig = field(default_factory=PortfolioEnvConfig)
    seed: int = 1729
    require_gpu: bool = True


def train_ppo_policy(
    predictions: pd.DataFrame,
    market_panel: pd.DataFrame,
    config: PPOTrainingConfig | None = None,
) -> dict[str, object]:
    cfg = config or PPOTrainingConfig()
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("train-rl-agent requires torch and stable_baselines3") from exc
    if cfg.require_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "RL GPU training was required, but torch.cuda.is_available() is false. "
            + format_cuda_diagnostic(cuda_runtime_probe(torch))
        )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.tensorboard_log).mkdir(parents=True, exist_ok=True)

    def make_env(rank: int):
        def _factory():
            env = PortfolioEnv(predictions, market_panel, cfg.env)
            env.reset(seed=cfg.seed + rank)
            return env

        return _factory

    if cfg.n_envs > 1:
        vec_env = SubprocVecEnv([make_env(i) for i in range(cfg.n_envs)])
    else:
        vec_env = DummyVecEnv([make_env(0)])
    model = PPO("MlpPolicy", vec_env, device=cfg.device, tensorboard_log=cfg.tensorboard_log, seed=cfg.seed)
    model.learn(total_timesteps=int(cfg.timesteps), progress_bar=False)
    policy_path = output_dir / "policy.zip"
    model.save(policy_path)
    summary = {
        "status": "passed",
        "policy_path": str(policy_path),
        "timesteps": int(cfg.timesteps),
        "device": cfg.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": asdict(cfg),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    vec_env.close()
    return summary


__all__ = ["PPOTrainingConfig", "train_ppo_policy"]
