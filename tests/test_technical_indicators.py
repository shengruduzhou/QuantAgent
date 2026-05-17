"""Tests for the classic technical indicators (Bollinger / RSI / MACD)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.factors import default_registry  # ensures registration
from quantagent.factors.technical_indicators import (
    bollinger_bandwidth,
    bollinger_percent_b,
    macd_hist,
    rsi_14,
)


def _panel(n: int = 120, n_symbols: int = 2, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    rows = []
    for k in range(n_symbols):
        prices = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5 + 0.02)
        for d, p in zip(dates, prices, strict=True):
            rows.append({"trade_date": d, "symbol": f"S{k:02d}", "close": float(p)})
    return pd.DataFrame(rows)


def test_factors_registered() -> None:
    names = set(default_registry.names(category="technical_indicators"))
    assert {
        "boll_percent_b_20",
        "boll_bandwidth_20",
        "rsi_14",
        "macd_hist_12_26_9",
        "macd_hist_norm_12_26_9",
    }.issubset(names)


def test_bollinger_percent_b_bounded() -> None:
    out = bollinger_percent_b(_panel())
    # After the lookup window the values are defined and mostly in [-0.5, 1.5].
    defined = out["factor_value"].dropna()
    assert len(defined) > 0
    assert defined.between(-1.5, 2.5).mean() > 0.95


def test_bollinger_bandwidth_non_negative() -> None:
    out = bollinger_bandwidth(_panel())
    defined = out["factor_value"].dropna()
    assert (defined >= 0).all()


def test_rsi_14_bounded_zero_hundred() -> None:
    out = rsi_14(_panel())
    defined = out["factor_value"].dropna()
    assert len(defined) > 0
    assert defined.between(0.0, 100.0).all()


def test_rsi_14_monotone_rising_series() -> None:
    n = 60
    dates = pd.bdate_range("2024-01-02", periods=n)
    rising = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": "RISER",
            "close": np.linspace(10.0, 60.0, n),
        }
    )
    out = rsi_14(rising).dropna()
    # All-up series should give RSI very close to 100 once warmed.
    assert (out["factor_value"].iloc[-1] > 95.0)


def test_macd_hist_shapes() -> None:
    panel = _panel(n=200)
    out = macd_hist(panel)
    assert list(out.columns) == ["trade_date", "symbol", "factor_name", "factor_value"]
    assert out["factor_name"].iloc[0] == "macd_hist_12_26_9"
    assert len(out) == len(panel)


def test_registry_compute_path() -> None:
    panel = _panel(n=80)
    out = default_registry.compute("rsi_14", panel)
    assert out.frame.shape[0] == len(panel)
    assert "factor_value" in out.frame.columns
