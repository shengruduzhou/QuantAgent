"""VWAP must not fabricate a price when turnover is merely unpublished.

``vwap = amount / volume`` is undefined in two different situations that a bare
``.fillna(close)`` collapses into one:

* volume == 0  -> nothing traded; VWAP does not exist; close is a defensible
  reference price.
* amount is NaN -> the name traded, but the source published no turnover; the
  average price is UNKNOWN.

The second case is not harmless. ``close / vwap - 1`` is exactly 0 when vwap ==
close, so every affected row reports a real, measured, zero-valued factor
instead of missing data. That became live once the AKShare Tencent adapter
started marking ``amount`` unavailable -- and Tencent is the failover, so it
engages whenever EastMoney is unreachable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.factors.cicc_high_freq import compute_cicc_high_freq_factors
from quantagent.factors.vwap import vwap_or_unknown


def _bars(amount, volume, close=10.0) -> pd.DataFrame:
    n = len(amount)
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2026-07-01", periods=n, freq="B"),
            "symbol": ["600000.SH"] * n,
            "open": [close] * n,
            "high": [close * 1.02] * n,
            "low": [close * 0.98] * n,
            "close": [close] * n,
            "volume": volume,
            "amount": amount,
        }
    )


class TestVwapOrUnknown:
    def test_normal_bars_give_the_true_average_price(self):
        frame = _bars(amount=[1_000_000.0], volume=[100_000.0], close=9.5)
        assert vwap_or_unknown(frame).iloc[0] == pytest.approx(10.0)

    def test_no_trading_falls_back_to_close(self):
        """volume == 0: VWAP genuinely does not exist."""
        frame = _bars(amount=[0.0], volume=[0.0], close=9.5)
        assert vwap_or_unknown(frame).iloc[0] == pytest.approx(9.5)

    def test_missing_turnover_stays_unknown_and_is_not_the_close(self):
        """The regression: real trading, unpublished amount -> NaN, not close."""
        frame = _bars(amount=[float("nan")], volume=[100_000.0], close=9.5)
        result = vwap_or_unknown(frame).iloc[0]
        assert np.isnan(result), f"expected NaN for unpublished turnover, got {result}"

    def test_missing_turnover_and_no_trading_are_distinguishable(self):
        frame = _bars(
            amount=[float("nan"), 0.0],
            volume=[100_000.0, 0.0],
            close=9.5,
        )
        vwap = vwap_or_unknown(frame)
        assert np.isnan(vwap.iloc[0])  # unknown
        assert vwap.iloc[1] == pytest.approx(9.5)  # no trading


class TestDownstreamFactorDoesNotReportZero:
    def test_unpublished_turnover_yields_nan_not_a_zero_factor(self):
        """close/vwap - 1 must be NaN, never a confident 0.0.

        A zero here is indistinguishable from a genuine "close equals VWAP"
        observation, so the factor would be silently fabricated across every row
        served by a source that publishes no turnover.
        """
        frame = _bars(
            amount=[float("nan")] * 30,
            volume=[100_000.0] * 30,
            close=9.5,
        )
        result = compute_cicc_high_freq_factors(frame)
        values = result.factors
        last30 = values[values["factor_name"] == "last_30min_return"]["factor_value"]
        assert last30.notna().sum() == 0, (
            "last_30min_return must be NaN when turnover is unavailable; "
            f"got {last30.dropna().unique()[:5]}"
        )
        assert not (last30.fillna(-999) == 0.0).any()

    def test_real_turnover_still_produces_a_finite_factor(self):
        """Guard against over-correcting into all-NaN."""
        rng = np.random.default_rng(0)
        n = 30
        volume = rng.uniform(80_000, 120_000, n)
        close = 9.5
        amount = volume * rng.uniform(9.4, 9.6, n)
        frame = _bars(amount=list(amount), volume=list(volume), close=close)
        result = compute_cicc_high_freq_factors(frame)
        values = result.factors
        last30 = values[values["factor_name"] == "last_30min_return"]["factor_value"]
        assert last30.notna().sum() > 0
        assert np.isfinite(last30.dropna()).all()
