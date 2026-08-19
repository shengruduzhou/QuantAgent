"""Reinforcement-learning portfolio research components.

:class:`PITPortfolioEnv` is the environment to use: it signals at close(T),
executes at close(T+1) and rewards over close(T+1) -> close(T+2), failing closed
on missing prices rather than scoring them as flat days.

:class:`PortfolioEnv` is retained only for comparison against historical runs.
Its reward is the close(T) -> close(T+1) return on the signal date, which is not
executable, so constructing it requires an explicit acknowledgement flag.
"""

from quantagent.rl.books import hold_band_book_from_predictions
from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig
from quantagent.rl.portfolio_env import PortfolioEnv, PortfolioEnvConfig
from quantagent.rl.train_ppo import PPOTrainingConfig, train_ppo_policy

__all__ = [
    "hold_band_book_from_predictions",
    "PITPortfolioEnv",
    "PITPortfolioEnvConfig",
    "PortfolioEnv",
    "PortfolioEnvConfig",
    "PPOTrainingConfig",
    "train_ppo_policy",
]
