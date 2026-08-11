"""Hold-band target-weight builder semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.portfolio.hold_band import (
    EXECUTION_DATE_SEMANTICS,
    SIGNAL_DATE_SEMANTICS,
    HoldBandConfig,
    build_hold_band_weights,
    build_regime_hold_band_weights,
    turnover_stats,
)


def _preds(rows):
    return pd.DataFrame(rows, columns=["symbol", "trade_date", "alpha_score"])


DATES = pd.date_range("2024-01-02", periods=6, freq="B")


def test_enter_top_then_hold_until_exit_band():
    rows = []
    scores = {
        0: {"A": 10, "B": 9, "C": 1, "D": 0.5},
        1: {"C": 10, "B": 9, "A": 5, "D": 0.5},
        2: {"C": 10, "B": 9, "D": 8, "A": 0.1},
    }
    for i, day_scores in scores.items():
        for sym, sc in day_scores.items():
            rows.append({"symbol": sym, "trade_date": DATES[i], "alpha_score": sc})
    cfg = HoldBandConfig(n_hold=2, entry_rank=2, exit_rank=3, delay_days=0)
    tw = build_hold_band_weights(_preds(rows), config=cfg, trade_dates=list(DATES))
    assert tw.attrs["target_index_semantics"] == SIGNAL_DATE_SEMANTICS
    assert tw.attrs["hold_band_delay_days"] == (0,)
    assert set(tw.loc[DATES[0]][tw.loc[DATES[0]] > 0].index) == {"A", "B"}
    assert set(tw.loc[DATES[1]][tw.loc[DATES[1]] > 0].index) == {"A", "B"}
    assert set(tw.loc[DATES[2]][tw.loc[DATES[2]] > 0].index) == {"B", "C"}


def test_delay_shifts_execution_date_and_declares_execution_semantics():
    rows = [{"symbol": "A", "trade_date": DATES[0], "alpha_score": 1.0}]
    cfg = HoldBandConfig(n_hold=1, entry_rank=1, exit_rank=2, delay_days=1)
    tw = build_hold_band_weights(_preds(rows), config=cfg, trade_dates=list(DATES))
    assert list(tw.index) == [DATES[1]]
    assert tw.loc[DATES[1], "A"] == 1.0
    assert tw.attrs["target_index_semantics"] == EXECUTION_DATE_SEMANTICS
    assert tw.attrs["hold_band_delay_days"] == (1,)


def test_negative_delay_rejected():
    with pytest.raises(ValueError, match="delay_days"):
        HoldBandConfig(delay_days=-1)


def test_ineligible_names_cannot_enter():
    rows = pd.DataFrame({
        "symbol": ["A", "B"],
        "trade_date": [DATES[0]] * 2,
        "alpha_score": [10.0, 5.0],
        "is_st": [True, False],
    })
    cfg = HoldBandConfig(n_hold=1, entry_rank=1, exit_rank=2, delay_days=0)
    tw = build_hold_band_weights(rows, config=cfg, trade_dates=list(DATES))
    assert set(tw.loc[DATES[0]][tw.loc[DATES[0]] > 0].index) == {"B"}
    assert tw.attrs["target_index_semantics"] == SIGNAL_DATE_SEMANTICS


def test_entry_wider_than_exit_rejected():
    with pytest.raises(ValueError):
        build_hold_band_weights(
            _preds([{"symbol": "A", "trade_date": DATES[0], "alpha_score": 1.0}]),
            config=HoldBandConfig(entry_rank=100, exit_rank=50),
        )


def test_regime_switch_changes_exit_band():
    scores = {
        0: {"A": 10, "B": 9, "C": 1},
        1: {"C": 10, "B": 9, "A": 5},
        2: {"C": 10, "B": 9, "A": 5},
    }
    rows = [
        {"symbol": s, "trade_date": DATES[i], "alpha_score": sc}
        for i, day in scores.items()
        for s, sc in day.items()
    ]
    config_map = {
        "sideways": HoldBandConfig(n_hold=2, entry_rank=2, exit_rank=3, delay_days=0),
        "bull": HoldBandConfig(n_hold=2, entry_rank=2, exit_rank=2, delay_days=0),
    }
    regime = pd.Series(["sideways", "sideways", "bull"], index=DATES[:3])
    tw = build_regime_hold_band_weights(
        _preds(rows),
        config_map=config_map,
        regime_by_date=regime,
        trade_dates=list(DATES),
    )
    assert tw.attrs["target_index_semantics"] == SIGNAL_DATE_SEMANTICS
    assert set(tw.loc[DATES[1]][tw.loc[DATES[1]] > 0].index) == {"A", "B"}
    assert set(tw.loc[DATES[2]][tw.loc[DATES[2]] > 0].index) == {"B", "C"}


def test_mixed_regime_delay_declares_mixed_semantics():
    rows = [
        {"symbol": "A", "trade_date": DATES[0], "alpha_score": 1.0},
        {"symbol": "A", "trade_date": DATES[1], "alpha_score": 1.0},
    ]
    config_map = {
        "sideways": HoldBandConfig(n_hold=1, entry_rank=1, exit_rank=2, delay_days=0),
        "bull": HoldBandConfig(n_hold=1, entry_rank=1, exit_rank=2, delay_days=1),
    }
    regime = pd.Series(["sideways", "bull"], index=DATES[:2])
    tw = build_regime_hold_band_weights(
        _preds(rows),
        config_map=config_map,
        regime_by_date=regime,
        trade_dates=list(DATES),
    )
    assert tw.attrs["target_index_semantics"] == "mixed"
    assert tw.attrs["hold_band_delay_days"] == (0, 1)


def test_regime_map_requires_default():
    with pytest.raises(ValueError):
        build_regime_hold_band_weights(
            _preds([{"symbol": "A", "trade_date": DATES[0], "alpha_score": 1.0}]),
            config_map={"bull": HoldBandConfig()},
            regime_by_date=pd.Series(["bull"], index=[DATES[0]]),
        )


def test_hold_band_turnover_below_daily_roll():
    rng = np.random.default_rng(0)
    symbols = [f"S{i:03d}" for i in range(80)]
    rows = []
    for d in pd.date_range("2024-01-02", periods=40, freq="B"):
        for j, s in enumerate(symbols):
            rows.append({
                "symbol": s,
                "trade_date": d,
                "alpha_score": -j + rng.normal(0, 3.0),
            })
    preds = _preds(rows)
    band = build_hold_band_weights(
        preds,
        config=HoldBandConfig(n_hold=10, entry_rank=5, exit_rank=30, delay_days=0),
    )
    roll = build_hold_band_weights(
        preds,
        config=HoldBandConfig(n_hold=10, entry_rank=10, exit_rank=10, delay_days=0),
    )
    assert turnover_stats(band)["mean_daily_turnover"] < turnover_stats(roll)["mean_daily_turnover"]
