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
from quantagent.backtest.factor_model_attribution import (
    MARKET,
    MOMENTUM,
    SIZE,
    VALUE,
    attribute_strategy_returns,
    build_ashare_style_factors,
)


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


def test_style_factors_build_and_declare_status():
    factors = build_ashare_style_factors(_panel(), return_column="return_1d")

    assert factors.status[MARKET] == "constructed"
    assert factors.status[VALUE] == "constructed"
    assert factors.status[MOMENTUM] == "constructed"
    # No share count was supplied, so size must be declared unavailable rather
    # than proxied by turnover or by anything else derivable from price.
    assert factors.status[SIZE] == "unavailable"
    assert SIZE not in factors.returns.columns
    assert "NOT proxied" in factors.notes[SIZE]


def test_price_based_sorts_are_lagged():
    """An unlagged B/M sort manufactures a huge negative value premium.

    ``book_yield`` here is exactly ``book_per_share / close(t)``, so a stock that
    fell today has a mechanically higher B/M today. If the sort used the same
    row as the payoff, HML would be strongly negative on data whose returns are
    pure noise. It must instead be indistinguishable from zero.
    """
    factors = build_ashare_style_factors(_panel(seed=3), return_column="return_1d")
    hml = factors.returns[VALUE].dropna()

    assert len(hml) > 100
    t_stat = float(hml.mean() / hml.std() * np.sqrt(len(hml)))
    # Returns are iid noise, so any surviving premium is an artifact.
    assert abs(t_stat) < 3.0, f"HML t={t_stat:.2f} on noise returns implies a look-ahead sort"


def test_shares_snapshot_marks_size_approximate():
    panel = _panel()
    shares = {symbol: 1e8 for symbol in panel["symbol"].unique()}

    factors = build_ashare_style_factors(
        panel,
        return_column="return_1d",
        shares_outstanding=shares,
        share_count_status="current_snapshot",
    )

    assert factors.status[SIZE] == "approximate"
    assert factors.share_count_status == "current_snapshot"
    assert "CURRENT-SNAPSHOT" in factors.notes[SIZE]


def test_attribution_reports_unavailable_rather_than_degrading():
    panel = _panel()
    factors = build_ashare_style_factors(panel, return_column="return_1d")
    strategy = panel.groupby("trade_date")["return_1d"].mean()

    report = attribute_strategy_returns(strategy, factors)

    assert report.levels["capm"].status == "measured"
    # SMB is unavailable, so FF3 and Carhart must say so rather than quietly
    # running a two-factor regression under a four-factor name.
    assert report.levels["ff3"].status == "unavailable"
    assert SIZE in report.levels["ff3"].missing_factors
    assert report.levels["carhart4"].status == "unavailable"
    assert report.strictest_measured == "capm"


def test_pure_style_exposure_has_no_carhart_alpha():
    """A book that is literally the value factor must not show value alpha."""
    panel = _panel(seed=5)
    shares = {symbol: 1e8 for symbol in panel["symbol"].unique()}
    factors = build_ashare_style_factors(
        panel,
        return_column="return_1d",
        shares_outstanding=shares,
        share_count_status="point_in_time",
    )
    # Strategy = 1.0x HML + 0.5x MKT and nothing else.
    strategy = (factors.returns[VALUE] * 1.0 + factors.returns[MARKET] * 0.5).dropna()

    report = attribute_strategy_returns(strategy, factors)

    assert report.strictest_measured == "carhart4"
    carhart = report.levels["carhart4"]
    assert carhart.status == "measured"
    assert carhart.loadings[VALUE] == pytest.approx(1.0, abs=0.05)
    assert carhart.loadings[MARKET] == pytest.approx(0.5, abs=0.05)
    assert abs(carhart.alpha_annual or 0.0) < 0.01
    assert report.survives_style_controls is False


def test_classify_strategy_demotes_capm_only_promotion():
    panel = {"cagr": 0.30, "alpha_all_a": 0.12, "beta_all_a": 0.9, "calmar": 1.5, "maxdd": 0.20}

    label, flags = classify_strategy(panel)

    # Without style controls a CAPM-only promotion is not a promotion.
    assert label == "research_signal"
    assert "capm_only:size_value_momentum_uncontrolled" in flags
    assert "demoted:production_claim_needs_ff3_or_carhart_alpha" in flags


def test_classify_strategy_demotes_when_alpha_absorbed():
    panel = {"cagr": 0.30, "alpha_all_a": 0.12, "beta_all_a": 0.9, "calmar": 1.5, "maxdd": 0.20}
    attribution = {
        "strictestMeasured": "carhart4",
        "survivesStyleControls": False,
        "levels": {"carhart4": {"status": "measured", "alphaAnnual": 0.04}},
        "factorSet": {"status": {SIZE: "constructed"}},
    }

    label, flags = classify_strategy(panel, style_attribution=attribution)

    assert label == "style_exposure"
    assert "alpha_does_not_survive_carhart4" in flags
    assert "alpha_basis:carhart4" in flags


def test_classify_strategy_promotes_on_surviving_carhart_alpha():
    panel = {"cagr": 0.30, "alpha_all_a": 0.12, "beta_all_a": 0.9, "calmar": 1.5, "maxdd": 0.20}
    attribution = {
        "strictestMeasured": "carhart4",
        "survivesStyleControls": True,
        "levels": {"carhart4": {"status": "measured", "alphaAnnual": 0.09}},
        "factorSet": {"status": {SIZE: "constructed"}},
    }

    label, flags = classify_strategy(panel, style_attribution=attribution)

    assert label == "production_candidate"
    assert "alpha_basis:carhart4" in flags
