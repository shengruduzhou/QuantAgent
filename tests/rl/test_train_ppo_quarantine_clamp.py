"""PPO training must not read quarantined holdout rewards (round-29 R11-F01)."""

import pandas as pd

from quantagent.rl.pit_portfolio_env import PITPortfolioEnvConfig
from quantagent.rl.train_ppo import _quarantine_clamped_env_config


def _panel(start, end):
    return pd.DataFrame({"trade_date": pd.bdate_range(start, end)})


def test_reward_end_is_clamped_before_the_first_quarantined_window():
    cfg = _quarantine_clamped_env_config(PITPortfolioEnvConfig(), _panel("2022-01-03", "2026-08-05"))
    assert cfg.reward_end_date_limit == "2025-08-31"


def test_an_earlier_explicit_limit_is_kept():
    requested = PITPortfolioEnvConfig(reward_end_date_limit="2024-12-31")
    cfg = _quarantine_clamped_env_config(requested, _panel("2022-01-03", "2026-08-05"))
    assert cfg.reward_end_date_limit == "2024-12-31"


def test_a_later_explicit_limit_cannot_reach_into_the_holdout():
    requested = PITPortfolioEnvConfig(reward_end_date_limit="2026-06-30")
    cfg = _quarantine_clamped_env_config(requested, _panel("2022-01-03", "2026-08-05"))
    assert cfg.reward_end_date_limit == "2025-08-31"
