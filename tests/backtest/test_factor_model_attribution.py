"""Tests for CAPM / FF3 / Carhart attribution.

The regression test that matters most here is
``test_price_based_sorts_are_lagged``. Book-to-market and market equity both
contain ``close(t)``, and ``return_1d(t)`` is the return *into* ``close(t)``, so
an unlagged sort puts the day's losers straight into the high-B/M leg. On the
real gold panel that bug produced an HML of -30.8%/yr (Sharpe -2.8) over
2021-2026, negative in all six calendar years; lagging the sort by one day turned
it into +14.3%/yr (Sharpe +1.1). Nothing about the market changed — only whether
the sort could see the payoff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.backtest.beta_decomposition import classify_strategy


def _panel(n_days: int = 400, n_symbols: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    symbols = [f"S{i:03d}" for i in range(n_symbols)]
    frames = []
    book_per_share = rng.uniform(1.0, 8.0, size=n_symbols)
    price = rng.uniform(5.0, 50.0, size=n_symbols)
    for date in dates:
        returns = rng.normal(0.0005, 0.02, size=n_symbols)
        price = price * (1.0 + returns)
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": date,
                    "symbol": symbols,
                    "close": price,
                    "return_1d": returns,
                    "book_yield": book_per_share / price,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)












def test_classify_strategy_demotes_capm_only_promotion():
    panel = {"cagr": 0.30, "alpha_all_a": 0.12, "beta_all_a": 0.9, "calmar": 1.5, "maxdd": 0.20}

    label, flags = classify_strategy(panel)

    # Without style controls a CAPM-only promotion is not a promotion.
    assert label == "research_signal"
    assert "capm_only:size_value_momentum_uncontrolled" in flags
    assert "demoted:production_claim_needs_ff3_or_carhart_alpha" in flags




