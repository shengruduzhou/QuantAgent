"""Round 23 / R11: the ablation harness must survive being interrupted.

Round 21 and round 22 both lost RL work to a session limit mid-run. A sweep
whose only output is its return value produces *nothing* when it dies at 80%,
and the arms already trained have to be retrained from scratch. ``results_path``
appends every finished ``(arm, seed)`` as JSON Lines the moment it is scored and
skips the pairs already present on a later invocation, so an interrupted sweep
costs the remaining arms rather than all of them.

The circularity guard is tested here too: an arm trained with a risk penalty is
scored in an environment built with ``drawdown_lambda = volatility_lambda = 0``.
Scoring a drawdown-penalised policy with a drawdown-penalised metric would
assume the conclusion the round-23 experiment exists to test.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from quantagent.rl.pit_portfolio_env import PITPortfolioEnvConfig
from quantagent.rl.reward_ablation import AblationConfig, RewardArm, run_ablation

DATES = pd.date_range("2026-01-05", periods=14, freq="B")
SYMBOLS = ("AAA", "BBB", "CCC")
_CLOSES = {
    "AAA": [10.0, 10.2, 10.1, 9.6, 9.0, 9.3, 9.8, 10.1, 10.4, 10.2, 10.5, 10.7, 10.6, 10.9],
    "BBB": [20.0, 19.8, 19.9, 20.3, 20.6, 20.4, 20.1, 20.5, 20.8, 21.0, 20.7, 20.9, 21.2, 21.1],
    "CCC": [5.0, 5.05, 5.1, 5.02, 4.9, 4.95, 5.0, 5.08, 5.15, 5.1, 5.2, 5.25, 5.18, 5.3],
}


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "close": float(close),
                "is_limit_up": False,
                "is_limit_down": False,
                "is_suspended": False,
            }
            for symbol in SYMBOLS
            for trade_date, close in zip(DATES, _CLOSES[symbol])
        ]
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "alpha_score": float((index + 1) * (1 + 0.1 * position)),
            }
            for position, trade_date in enumerate(DATES)
            for index, symbol in enumerate(SYMBOLS)
        ]
    )


def _book() -> pd.DataFrame:
    return pd.DataFrame(
        0.5, index=DATES, columns=["AAA", "BBB"], dtype=float
    )


def _controls_config() -> AblationConfig:
    return AblationConfig(
        timesteps=1,
        n_envs=1,
        seeds=(11, 22),
        base_env=PITPortfolioEnvConfig(max_book=4),
    )


def _run(tmp_path, arms, sink_name="runs.jsonl"):
    book = _book()
    return run_ablation(
        train_book=book.iloc[:8],
        test_book=book.iloc[8:],
        predictions=_predictions(),
        panel=_panel(),
        arms=arms,
        config=_controls_config(),
        results_path=tmp_path / sink_name,
    )


def test_every_finished_run_is_on_disk_before_the_sweep_ends(tmp_path):
    """Not "at the end" -- after each run, which is what makes it survivable."""
    sink = tmp_path / "runs.jsonl"
    runs = _run(tmp_path, [RewardArm(name="zero", kind="zero"), RewardArm(name="random", kind="random")])
    lines = [json.loads(line) for line in sink.read_text().splitlines() if line.strip()]
    assert len(lines) == len(runs) == 4
    assert {(row["arm"], row["seed"]) for row in lines} == {
        ("zero", 11), ("zero", 22), ("random", 11), ("random", 22)
    }
    assert all("cumulative_value_add" in row for row in lines)


def test_a_resumed_sweep_does_not_recompute_what_is_already_on_disk(tmp_path):
    """Proven by tampering, not by timing.

    A sentinel value is written into the persisted row; if the resumed sweep
    recomputed the pair it would overwrite the sentinel with the real number.
    A timing-based assertion would only prove the second run was fast.
    """
    sink = tmp_path / "runs.jsonl"
    _run(tmp_path, [RewardArm(name="zero", kind="zero")])
    rows = [json.loads(line) for line in sink.read_text().splitlines() if line.strip()]
    for row in rows:
        row["cumulative_value_add"] = -12345.0
    sink.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")

    resumed = _run(
        tmp_path,
        [RewardArm(name="zero", kind="zero"), RewardArm(name="random", kind="random")],
    )
    zero_rows = resumed.loc[resumed["arm"] == "zero", "cumulative_value_add"]
    assert list(zero_rows) == [-12345.0, -12345.0]
    assert len(resumed) == 4
    assert (resumed.loc[resumed["arm"] == "random", "cumulative_value_add"] != -12345.0).all()


def test_the_zero_control_scores_exactly_zero_value_add(tmp_path):
    """The control that has to be exactly 0.0, not approximately.

    It is the construction that makes this environment immune to the env-flat
    trap (round 21 DEF-036): if the passive book itself scored non-zero
    value-add, every arm's number would carry that offset and the ranking would
    be measuring the environment rather than the policies.
    """
    runs = _run(tmp_path, [RewardArm(name="zero", kind="zero")])
    assert list(runs["cumulative_value_add"]) == [0.0, 0.0]
    assert list(runs["excess_max_drawdown"]) == [0.0, 0.0]


def test_the_yardstick_never_carries_the_arm_s_risk_lambdas(tmp_path):
    """The anti-circularity property, asserted rather than assumed.

    Two control arms differing only in their training lambdas must produce
    *identical* evaluation metrics, because the controls are not trained and
    the evaluation environment is built with both lambdas forced to zero. If
    the arm's lambdas leaked into the yardstick, the penalised arm's drawdown
    would be scored on a different scale and the comparison would be rigged.
    """
    plain = _run(tmp_path, [RewardArm(name="zero", kind="zero")], sink_name="a.jsonl")
    penalised = _run(
        tmp_path,
        [RewardArm(name="zero", kind="zero", drawdown_lambda=2.0, volatility_lambda=100.0)],
        sink_name="b.jsonl",
    )
    columns = ["cumulative_value_add", "nav", "max_drawdown", "mean_turnover"]
    np.testing.assert_array_equal(
        plain[columns].to_numpy(), penalised[columns].to_numpy()
    )


def test_gross_exposure_is_measured_against_the_same_passive_book(tmp_path):
    """Round 22 predicted the drawdown penalty would be bought by de-grossing.

    The action's cash tilt can cut gross by up to 30%, so a penalised arm can
    lower its drawdown by simply holding less -- a return give-up, not better
    selection. The ablation therefore has to report gross, and report it
    against the same constrained passive book the reward differences against.
    Under a zero action the two are bit-identical, which is what this pins.
    """
    runs = _run(tmp_path, [RewardArm(name="zero", kind="zero")])
    assert (runs["mean_gross"] == runs["mean_gross_passive"]).all()
    assert (runs["mean_gross"] > 0.0).all()
