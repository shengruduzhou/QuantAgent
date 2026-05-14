import numpy as np
import pandas as pd

from quantagent.portfolio.sleeve import SleeveType
from quantagent.portfolio.walk_forward_sleeve_allocator import (
    WalkForwardSleeveConfig,
    allocate_sleeves_walk_forward,
    synthesise_sleeve_returns,
)


def _sleeve_panel(seed: int = 7, periods: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=periods, freq="B")
    # long_fundamental has a small positive drift; hedge is negatively correlated
    # with the market; cash_buffer is deterministic 0 return.
    market = rng.normal(0.0005, 0.012, periods)
    long_fund = market + rng.normal(0.0003, 0.004, periods)
    short_event = market + rng.normal(-0.0001, 0.018, periods)
    sector_rotation = market + rng.normal(0.0001, 0.010, periods)
    hedge = -0.5 * market + rng.normal(0.0, 0.006, periods)
    cash = np.zeros(periods)
    frame = pd.DataFrame(
        {
            SleeveType.LONG_FUNDAMENTAL.value: long_fund,
            SleeveType.SHORT_EVENT.value: short_event,
            SleeveType.SECTOR_ROTATION.value: sector_rotation,
            SleeveType.HEDGE.value: hedge,
            SleeveType.CASH_BUFFER.value: cash,
        },
        index=dates,
    )
    return frame


def test_walk_forward_sleeve_allocator_returns_full_dispatch():
    panel = _sleeve_panel()
    result = allocate_sleeves_walk_forward(
        panel,
        config=WalkForwardSleeveConfig(walk_forward_splits=3, embargo_days=3, min_window_days=30, grid_step=0.1),
    )

    sleeve_weights = {target.sleeve_type: target.target_weight for target in result.targets}
    assert SleeveType.LONG_FUNDAMENTAL in sleeve_weights
    assert SleeveType.CASH_BUFFER in sleeve_weights
    total = sum(sleeve_weights.values())
    assert total == pytest_approx_one(total)
    assert result.cash_weight == sleeve_weights[SleeveType.CASH_BUFFER]
    assert result.diagnostics["walk_forward_windows"] >= 1.0


def pytest_approx_one(value: float) -> float:
    return round(value, 6)


def test_walk_forward_sleeve_allocator_falls_back_on_insufficient_history():
    short_panel = _sleeve_panel(periods=20)
    result = allocate_sleeves_walk_forward(short_panel)
    assert result.diagnostics.get("fallback", 0.0) == 1.0
    reasons = {target.reason for target in result.targets}
    assert reasons == {"insufficient_sleeve_history"} or reasons == {"walk_forward_window_unavailable"}


def test_synthesise_sleeve_returns_aggregates_by_membership():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-01-02", periods=50, freq="B")
    symbols = ["A.SH", "B.SH", "C.SH"]
    rows = []
    for symbol in symbols:
        prices = 10.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, len(dates))))
        for date, close in zip(dates, prices):
            rows.append({"trade_date": date, "symbol": symbol, "close": close})
    market_panel = pd.DataFrame(rows)
    membership = {
        "A.SH": SleeveType.LONG_FUNDAMENTAL.value,
        "B.SH": SleeveType.LONG_FUNDAMENTAL.value,
        "C.SH": SleeveType.SHORT_EVENT.value,
    }
    panel = synthesise_sleeve_returns(market_panel, membership)
    assert SleeveType.LONG_FUNDAMENTAL.value in panel.columns
    assert SleeveType.SHORT_EVENT.value in panel.columns
    assert panel.dropna(how="all").shape[0] > 0
