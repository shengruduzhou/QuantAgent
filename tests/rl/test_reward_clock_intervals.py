"""Pin the reward interval of BOTH RL environments, numerically.

The repo carries two portfolio environments. Until now the *unsafe* one was the
only export of ``quantagent.rl`` and the one ``train_ppo`` imported, so the
default training path optimised an objective no policy could have traded, while
the PIT-safe environment sat beside it unused.

The panel below is built so the two candidate intervals give DIFFERENT answers,
which is what makes this a decisive test rather than a restatement of the code:

    A: 100 -> 100 -> 120     0% over D0->D1, +20% over D1->D2
    B: 100 -> 110 -> 110   +10% over D0->D1,   0% over D1->D2

so for signal date D0:
    close(T)   -> close(T+1)  gives [0.00, 0.10]   (untradable)
    close(T+1) -> close(T+2)  gives [0.20, 0.00]   (executable)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig
from quantagent.rl.portfolio_env import PortfolioEnv, PortfolioEnvConfig

DATES = pd.to_datetime(
    ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
)
PRICES = {
    "A": [100.0, 100.0, 120.0, 120.0, 120.0],
    "B": [100.0, 110.0, 110.0, 110.0, 110.0],
}


def _panel(prices=None, flags=None, drop=None) -> pd.DataFrame:
    prices = prices or PRICES
    flags = flags or {}
    drop = drop or set()
    rows = []
    for i, date in enumerate(DATES):
        for symbol, series in prices.items():
            if (symbol, i) in drop:
                continue
            flag = flags.get((symbol, i), {})
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "close": series[i],
                    "is_limit_up": flag.get("lu", False),
                    "is_limit_down": flag.get("ld", False),
                    "is_suspended": flag.get("susp", False),
                }
            )
    return pd.DataFrame(rows)


def _preds() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_date": d, "symbol": s, "alpha_score": 1.0 if s == "A" else -1.0}
            for d in DATES
            for s in PRICES
        ]
    )


def _book() -> pd.DataFrame:
    return pd.DataFrame({"A": [0.5] * 3, "B": [0.5] * 3}, index=DATES[:3])


class TestPITEnvIsExecutable:
    def test_reward_uses_t_plus_1_to_t_plus_2(self):
        env = PITPortfolioEnv(_book(), _preds(), _panel(), PITPortfolioEnvConfig(max_book=4))
        np.testing.assert_allclose(env.slot_ret[0][:2], [0.20, 0.00], atol=1e-9)

    def test_reward_value_matches_the_hand_computed_pit_answer(self):
        """Passive 50/50, policy 90/10, identical turnover => costs cancel.

        reward = 100 * 0.4 * (r_A - r_B). Under close(T+1)->close(T+2) that is
        100*0.4*(0.20-0.00) = +8.0. The untradable interval would give -4.0.
        """
        env = PITPortfolioEnv(_book(), _preds(), _panel(), PITPortfolioEnvConfig(max_book=4))
        env.reset()
        _, reward, _, _, _ = env.step(np.array([1.0, -1.0, 0.0, 0.0, 0.0]))
        assert reward == pytest.approx(8.0, abs=1e-6)

    def test_semantics_tag_matches_measured_behaviour(self):
        env = PITPortfolioEnv(_book(), _preds(), _panel(), PITPortfolioEnvConfig(max_book=4))
        assert "reward_t_plus_1_to_t_plus_2" in env.reward_clock_semantics


class TestPITEnvFailsClosed:
    @pytest.mark.parametrize("missing_index", [1, 2])
    def test_missing_close_raises_rather_than_scoring_a_flat_day(self, missing_index):
        """A dropped bar must not become a 0.0 return the agent learns from."""
        with pytest.raises(ValueError, match="missing/non-positive close"):
            PITPortfolioEnv(
                _book(), _preds(), _panel(drop={("A", missing_index)}),
                PITPortfolioEnvConfig(max_book=4),
            )

    def test_nan_close_raises(self):
        prices = dict(PRICES, A=[100.0, 100.0, float("nan"), 120.0, 120.0])
        with pytest.raises(ValueError, match="missing/non-positive close"):
            PITPortfolioEnv(_book(), _preds(), _panel(prices), PITPortfolioEnvConfig(max_book=4))

    def test_zero_close_raises(self):
        prices = dict(PRICES, A=[100.0, 0.0, 120.0, 120.0, 120.0])
        with pytest.raises(ValueError, match="missing/non-positive close"):
            PITPortfolioEnv(_book(), _preds(), _panel(prices), PITPortfolioEnvConfig(max_book=4))

    @pytest.mark.parametrize("flag", ["susp", "lu"])
    def test_untradable_name_gets_zero_weight_on_the_execution_date(self, flag):
        """Suspended or limit-up on T+1: the policy must not hold it."""
        env = PITPortfolioEnv(
            _book(), _preds(), _panel(flags={("A", 1): {flag: True}}),
            PITPortfolioEnvConfig(max_book=4),
        )
        env.reset()
        _, _, _, _, info = env.step(np.array([1.0, -1.0, 0.0, 0.0, 0.0]))
        assert info["weights"]["A"] == pytest.approx(0.0)


class TestLegacyEnvIsUntradableAndGuarded:
    def test_construction_is_refused_without_acknowledgement(self):
        with pytest.raises(ValueError, match="not executable"):
            PortfolioEnv(_preds(), _panel(), PortfolioEnvConfig(top_n=2))

    def test_reward_interval_is_the_untradable_one(self):
        env = PortfolioEnv(
            _preds(), _panel(),
            PortfolioEnvConfig(top_n=2, acknowledge_untradable_reward=True),
        )
        matrix = np.asarray(env._forward_return_matrix, dtype=float)
        # close(T)->close(T+1): A flat, B +10%. NOT [0.20, 0.00].
        np.testing.assert_allclose(matrix[0][:2], [0.00, 0.10], atol=1e-9)

    def test_the_two_environments_genuinely_disagree(self):
        """Guard against a future edit quietly making these identical."""
        legacy = PortfolioEnv(
            _preds(), _panel(),
            PortfolioEnvConfig(top_n=2, acknowledge_untradable_reward=True),
        )
        pit = PITPortfolioEnv(_book(), _preds(), _panel(), PITPortfolioEnvConfig(max_book=4))
        legacy_row = np.asarray(legacy._forward_return_matrix, dtype=float)[0][:2]
        pit_row = np.asarray(pit.slot_ret[0][:2], dtype=float)
        assert not np.allclose(legacy_row, pit_row), (
            "the legacy and PIT environments returned the same interval; if the "
            "legacy env was fixed, delete it rather than leaving two copies"
        )


class TestPackageExports:
    def test_pit_env_is_exported(self):
        import quantagent.rl as rl

        assert "PITPortfolioEnv" in rl.__all__
        assert hasattr(rl, "PITPortfolioEnv")
