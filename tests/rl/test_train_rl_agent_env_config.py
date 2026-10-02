"""`train-rl-agent --env-config` must apply the YAML it is given, or refuse it.

Round 29 R9-F02: the allow-list fell back to legacy PortfolioEnv keys because
`PITPortfolioEnvConfig` was never in module globals, so `max_book`,
`volatility_lambda` and `reward_end_date_limit` were dropped without a word and
`max_turnover` passed the filter only to crash the constructor.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from quantagent.cli.v7_train import _load_env_config
from quantagent.rl.pit_portfolio_env import PITPortfolioEnvConfig


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "env.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_every_real_field_survives_the_round_trip(tmp_path):
    path = _write(
        tmp_path,
        "rl_env:\n"
        "  max_book: 40\n"
        "  volatility_lambda: 0.5\n"
        "  drawdown_lambda: 0.2\n"
        "  reward_end_date_limit: '2025-08-29'\n",
    )
    cfg = PITPortfolioEnvConfig(**_load_env_config(path))
    assert cfg.max_book == 40
    assert cfg.volatility_lambda == 0.5
    assert cfg.drawdown_lambda == 0.2
    assert cfg.reward_end_date_limit == "2025-08-29"


def test_a_legacy_key_is_refused_by_name(tmp_path):
    path = _write(tmp_path, "rl_env:\n  max_turnover: 0.3\n")
    with pytest.raises(typer.BadParameter, match="max_turnover"):
        _load_env_config(path)


def test_no_config_means_no_overrides():
    assert _load_env_config(None) == {}
