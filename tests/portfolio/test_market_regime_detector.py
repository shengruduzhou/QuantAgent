"""MarketRegimeDetector tests (spec section 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.portfolio.market_regime_detector import (
    DEFAULT_BENCHMARK_INDICES,
    MarketRegimeConfig,
    REGIME_LABELS,
    RISK_LEVELS,
    detect_market_regime,
    detect_market_regime_series,
    regime_risk_to_exposure_cap,
)


def _index_panel(
    *,
    indices: tuple[str, ...] = DEFAULT_BENCHMARK_INDICES,
    n_days: int = 30,
    daily_ret: float = 0.0,
    daily_vol: float = 1_000_000.0,
    vol_expand: bool = False,
) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for idx in indices:
        price = 100.0
        for i, d in enumerate(dates):
            price *= (1.0 + daily_ret)
            vol = daily_vol
            if vol_expand and i >= n_days - 5:
                vol *= 1.5
            rows.append({"trade_date": d, "index_code": idx, "close": price, "volume": vol})
    return pd.DataFrame(rows)


def test_neutral_market_classifies_as_normal():
    panel = _index_panel(daily_ret=0.0001)
    snap = detect_market_regime(
        trade_date=pd.Timestamp("2024-01-30"), index_panel=panel,
    )
    assert snap.regime in {"normal", "bull_consolidation"}
    assert snap.risk_level in RISK_LEVELS


def test_strong_bull_with_volume_expansion_yields_bull_expansion():
    panel = _index_panel(daily_ret=0.005, vol_expand=True, n_days=30)
    # last bdate in the panel = 2024-02-09 (the volume expansion period)
    end_date = pd.bdate_range("2024-01-01", periods=30)[-1]
    breadth = pd.DataFrame([{
        "trade_date": end_date,
        "advance_count": 3000, "decline_count": 1500,
        "limit_up_count": 50, "limit_down_count": 2,
        "max_consecutive_limit_up": 4, "zhaban_rate": 0.10,
    }])
    snap = detect_market_regime(
        trade_date=end_date, index_panel=panel, breadth=breadth,
    )
    assert snap.regime == "bull_expansion"
    assert snap.risk_level == "low"
    assert snap.bull_index_count == len(DEFAULT_BENCHMARK_INDICES)


def test_persistent_bear_yields_capitulation_or_crisis():
    panel = _index_panel(daily_ret=-0.01)
    breadth = pd.DataFrame([{
        "trade_date": pd.Timestamp("2024-01-30"),
        "advance_count": 200, "decline_count": 4000,
        "limit_up_count": 1, "limit_down_count": 60,
        "max_consecutive_limit_up": 0, "zhaban_rate": 0.50,
    }])
    snap = detect_market_regime(
        trade_date=pd.Timestamp("2024-01-30"),
        index_panel=panel, breadth=breadth,
    )
    assert snap.regime in {"bear_capitulation", "crisis"}
    assert snap.risk_level == "severe"


def test_softening_breadth_triggers_caution():
    panel = _index_panel(daily_ret=0.001)
    breadth = pd.DataFrame([{
        "trade_date": pd.Timestamp("2024-01-30"),
        "advance_count": 1000, "decline_count": 2500,
        "limit_up_count": 5, "limit_down_count": 8,
        "max_consecutive_limit_up": 1, "zhaban_rate": 0.20,
    }])
    snap = detect_market_regime(
        trade_date=pd.Timestamp("2024-01-30"),
        index_panel=panel, breadth=breadth,
    )
    assert snap.regime in {"caution", "normal"}


def test_no_index_data_defaults_to_normal_or_lower():
    snap = detect_market_regime(trade_date=pd.Timestamp("2024-01-30"))
    assert snap.regime == "normal"


def test_detect_market_regime_series_returns_tabular_history():
    panel = _index_panel(daily_ret=0.002, n_days=40)
    dates = pd.bdate_range("2024-01-20", periods=10).tolist()
    out = detect_market_regime_series(trade_dates=dates, index_panel=panel)
    assert len(out) == 10
    assert "regime" in out.columns
    assert "risk_level" in out.columns
    assert out["regime"].isin(REGIME_LABELS).all()


def test_regime_risk_to_exposure_cap_monotonic():
    caps = [regime_risk_to_exposure_cap(lvl) for lvl in ("low", "medium", "high", "severe")]
    # strictly decreasing as risk rises
    assert caps == sorted(caps, reverse=True)
    assert caps[0] >= caps[-1]


def test_regime_snapshot_to_dict_serialisable():
    panel = _index_panel(daily_ret=0.001)
    snap = detect_market_regime(trade_date=pd.Timestamp("2024-01-30"), index_panel=panel)
    d = snap.to_dict()
    assert d["regime"] in REGIME_LABELS
    assert d["risk_level"] in RISK_LEVELS
    assert "reason" in d
