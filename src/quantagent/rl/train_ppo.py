"""PPO training entry point for the V7 portfolio environment.

Trains against :class:`quantagent.rl.pit_portfolio_env.PITPortfolioEnv`, which
signals at ``close(T)``, executes at ``close(T+1)``, rewards over
``close(T+1) -> close(T+2)`` and fails closed on missing prices.

It previously used :class:`~quantagent.rl.portfolio_env.PortfolioEnv`, whose
reward is the ``close(T) -> close(T+1)`` return on the *signal* date -- a return
no policy could have captured, because it cannot trade at ``close(T)`` on
information contained in ``close(T)``. Measured on a panel where the two
candidate intervals differ, that environment reports ``[0.0, 0.10]`` where the
executable interval gives ``[0.20, 0.0]``. **Any RL result produced before this
change optimised an objective that does not exist and must not be cited.**

``book_weights`` is a required keyword argument rather than a defaulted one: the
reward is value-add over that passive book, so it fixes the benchmark the result
is quoted against, and choosing it silently would be choosing the caller's
benchmark for them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import pandas as pd

from quantagent.config.paths import quant_paths
from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig
from quantagent.cuda_runtime import configure_cuda_environment, cuda_runtime_probe, format_cuda_diagnostic

configure_cuda_environment()


@dataclass(frozen=True)
class PPOTrainingConfig:
    timesteps: int = 5_000_000
    device: str = "cuda"
    n_envs: int = 8
    output_dir: str = field(default_factory=lambda: str(quant_paths().models / "v7_rl_policy"))
    tensorboard_log: str = field(default_factory=lambda: str(quant_paths().logs / "tb" / "rl"))
    env: PITPortfolioEnvConfig = field(default_factory=PITPortfolioEnvConfig)
    seed: int = 1729
    require_gpu: bool = True


def equal_weight_book_from_predictions(
    predictions: pd.DataFrame,
    *,
    top_k: int,
    score_column: str = "alpha_score",
    date_column: str = "trade_date",
    symbol_column: str = "symbol",
) -> pd.DataFrame:
    """Build a naive equal-weight top-k passive book from alpha scores.

    This is a *baseline you are choosing*, not a neutral default, which is why
    ``train_ppo_policy`` will not construct one for you. The reward is the
    policy's value-add over this book, so the book defines what "no skill" means:
    a policy that merely reproduces it scores zero. Pick it deliberately -- an
    equal-weight top-k book is a much weaker opponent than the live target
    weights, and beating it is correspondingly less meaningful.

    **It also has no holding period.** The top-k list is rebuilt from scratch on
    every session, so the book follows the ranking wherever it goes. Measured on
    the round-23 dataset (690 sessions, ``top_k=30``) that is a median of 27 of
    30 names replaced per session: two-sided turnover median 1.80, p90 2.00,
    median holding spell one session, ~21 bps of cost per session at the
    environment's 12 bps. Any result quoted against this book is dominated by
    transaction cost. Use
    :func:`quantagent.rl.books.hold_band_book_from_predictions` when the
    benchmark is supposed to represent something anyone would actually hold; see
    ``docs/audits/round23/11_rl_ablation.md`` for the before/after measurement.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    frame = predictions[[date_column, symbol_column, score_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=[date_column, symbol_column, score_column])
    if frame.empty:
        raise ValueError("cannot build a passive book from empty predictions")

    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for date, group in frame.groupby(date_column, sort=True):
        chosen = group.nlargest(top_k, score_column)[symbol_column].astype(str)
        if chosen.empty:
            continue
        weight = 1.0 / float(len(chosen))
        rows[date] = {symbol: weight for symbol in chosen}
    if not rows:
        raise ValueError("passive book construction selected no names on any date")
    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0).sort_index()


def train_ppo_policy(
    predictions: pd.DataFrame,
    market_panel: pd.DataFrame,
    config: PPOTrainingConfig | None = None,
    *,
    book_weights: pd.DataFrame,
) -> dict[str, object]:
    """Train a PPO policy on the PIT-safe environment.

    ``book_weights`` is required and keyword-only on purpose. The reward is the
    policy's value-add over this passive book, so it defines the benchmark the
    result will be quoted against; defaulting it would silently pick the
    benchmark on the caller's behalf. Use
    :func:`equal_weight_book_from_predictions` for a naive baseline, or supply
    the live target weights to measure improvement over what is actually held.
    """
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
            env = PITPortfolioEnv(book_weights, predictions, market_panel, cfg.env)
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


__all__ = [
    "PPOTrainingConfig",
    "equal_weight_book_from_predictions",
    "train_ppo_policy",
]
