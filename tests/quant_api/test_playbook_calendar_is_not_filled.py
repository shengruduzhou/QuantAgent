"""Multi-asset playbooks must not forward-fill a session an asset never printed.

Round 21 / R9 finding A-06.  `market_playbooks_v3.time_series_momentum` built
its panel as the UNION of every asset's dates and then `.ffill()`ed the holes.
Three consequences, all in the optimistic direction:

* the strategy was recorded as trading at an open price that was never printed,
  while `assumptions` declared open-to-open marking;
* every gap session became a 0% return rather than a missing one;
* the fill suppressed realised volatility, and since weights are inverse-vol,
  the book tilted TOWARDS whichever asset had the worst data.

Measured on one price process: +33.5102% fully observed against +41.1854%
forward-filled -- 7.67pp of pure fill.

The benchmark leg had the same defect in `cross_sectional_reversal`, which is
DEF-022 exactly: a filled benchmark gap is a 0% benchmark day, so excess return
inflates by the whole missing move.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from services.quant_api.services import market_playbooks_v3


def test_momentum_panel_is_not_forward_filled() -> None:
    source = inspect.getsource(market_playbooks_v3.MarketPlaybookService.time_series_momentum)
    assert ".ffill()" not in source, "the momentum panel is being filled again"
    assert "observed" in source, "the panel must be restricted to observed sessions"


def test_benchmark_gap_is_not_filled() -> None:
    source = inspect.getsource(
        market_playbooks_v3.MarketPlaybookService.short_term_reversal
    )
    assert "benchmark_close" in source
    assert ".reindex(close.index).ffill()" not in source, (
        "a filled benchmark gap is a 0% benchmark day (DEF-022)"
    )


def test_forward_filling_injects_sessions_that_did_not_happen() -> None:
    """The mechanism itself, independent of the service.

    A session an asset did not print enters the filled return series as an
    exact 0%. That is a session the strategy is recorded as having held
    through, at a price never printed, contributing to every path statistic --
    drawdown, volatility, the inverse-vol weights -- computed downstream.

    The direction of the volatility distortion depends on the price path (a
    fill inserts a flat day AND a doubled move), so this pins the invariant
    rather than a direction: the filled series contains return observations
    the market never produced, and the observed series contains none.
    """
    dates = pd.bdate_range("2026-01-01", periods=12)
    prices = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0,
         106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
        index=dates,
    )
    holed = prices.copy()
    holed.iloc[[3, 6, 9]] = np.nan   # three sessions this asset did not print

    filled_returns = holed.ffill().pct_change(fill_method=None).dropna()
    observed_returns = holed.dropna().pct_change(fill_method=None).dropna()

    manufactured = int((filled_returns == 0.0).sum())
    assert manufactured == 3, (
        "each unprinted session became a 0% return the market never produced"
    )
    assert int((observed_returns == 0.0).sum()) == 0
    # And the filled path is strictly longer: three extra compounding steps.
    assert len(filled_returns) == len(observed_returns) + 3


def test_calendar_facts_are_published_not_absorbed() -> None:
    """"How many sessions were dropped" must be readable from the response."""
    source = inspect.getsource(market_playbooks_v3.MarketPlaybookService.time_series_momentum)
    assert "unobservedSessionsDropped" in source
    assert "sessionsEvaluated" in source

    reversal = inspect.getsource(
        market_playbooks_v3.MarketPlaybookService.short_term_reversal
    )
    assert "benchmarkMissingSessions" in reversal
