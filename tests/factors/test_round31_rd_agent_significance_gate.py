"""Round 31 Q-F01: the RD-Agent acceptance gate admitted pure noise.

With the old defaults (validation rank-IC >= 0, ICIR >= 0) the loop accepted
48% of candidates on panels whose label is independent of everything. The gate
now requires a non-overlapping validation IC t-stat >= 3 while still accepting
a real (planted) reversal signal.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quantagent.factors.factor_synthesis import (
    RDAgentFactorLoopConfig,
    _nonoverlapping_ic_tstat,
    synthesize_factors_rd_agent,
)


def _panel(seed: int, *, planted: float = 0.0, T: int = 420, N: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=T)
    r = 0.02 * rng.standard_normal((T, N))
    close = 10 * np.exp(np.cumsum(r, axis=0))
    open_ = close * np.exp(0.005 * rng.standard_normal((T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(0.01 * rng.standard_normal((T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(0.01 * rng.standard_normal((T, N))))
    vol = np.exp(12 + 0.5 * rng.standard_normal((T, N)))
    noise = 0.04 * rng.standard_normal((T, N))
    past_5d = np.vstack([np.full((5, N), np.nan), close[5:] / close[:-5] - 1.0])
    label = noise - planted * np.nan_to_num(past_5d)
    return pd.DataFrame({
        "trade_date": np.repeat(dates, N), "symbol": np.tile([f"S{i:04d}" for i in range(N)], T),
        "open": open_.ravel(), "high": high.ravel(), "low": low.ravel(), "close": close.ravel(),
        "volume": vol.ravel(), "amount": (vol * close).ravel(), "forward_return_5d": label.ravel(),
    })


def test_pure_noise_is_not_accepted():
    logging.disable(logging.WARNING)
    try:
        accepted = sum(
            len(synthesize_factors_rd_agent(_panel(seed), config=RDAgentFactorLoopConfig(rounds=4, seed=1729 + seed)).definitions)
            for seed in range(3)
        )
    finally:
        logging.disable(logging.NOTSET)
    assert accepted == 0


def test_planted_reversal_signal_is_still_accepted():
    logging.disable(logging.WARNING)
    try:
        result = synthesize_factors_rd_agent(
            _panel(0, planted=0.4), config=RDAgentFactorLoopConfig(rounds=4, seed=1729)
        )
    finally:
        logging.disable(logging.NOTSET)
    assert len(result.definitions) >= 1


def test_nonoverlapping_tstat_ignores_overlap_inflation():
    rng = np.random.default_rng(3)
    # Persistent (AR(1)) noise IC: overlapping t-stats inflate, cohorts do not.
    e = rng.standard_normal(2000)
    ic = np.empty_like(e)
    ic[0] = e[0]
    for i in range(1, len(e)):
        ic[i] = 0.8 * ic[i - 1] + e[i]
    assert abs(_nonoverlapping_ic_tstat(pd.Series(0.01 * ic), 5)) < 3.0
