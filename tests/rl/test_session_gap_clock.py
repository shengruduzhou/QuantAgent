"""Synthetic missing-session regressions; no real market data is consumed."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gymnasium")

from quantagent.rl.pit_portfolio_env import PITPortfolioEnv, PITPortfolioEnvConfig


def _inputs(missing_index=3, classification="MISSING_UNEXPLAINED"):
    dates = pd.bdate_range("2024-01-08", periods=8)
    prices = {"A": [80, 100, 120, 120, 126, 126, 130, 130],
              "B": [200, 100, 100, 110, 110, 110, 110, 110]}
    panel = pd.DataFrame([
        dict(trade_date=date, symbol=symbol, close=values[i],
             is_limit_up=False, is_limit_down=False, is_suspended=False)
        for symbol, values in prices.items() for i, date in enumerate(dates)
        if i != missing_index
    ])
    book = pd.DataFrame({"A": [0.6, 0.2, 0.2], "B": [0.4, 0.8, 0.8]}, index=dates[:3])
    predictions = pd.DataFrame([
        dict(trade_date=date, symbol=symbol, alpha_score=1.0 if symbol == "A" else -1.0)
        for date in dates for symbol in prices
    ])
    gaps = pd.DataFrame([
        dict(trade_date=dates[missing_index], symbol=symbol, classification=classification)
        for symbol in prices
    ])
    return dates, book, predictions, panel, gaps


@pytest.mark.parametrize("missing_index", [1, 2, 3])
def test_whole_session_unexplained_gap_is_not_skipped(missing_index):
    dates, book, predictions, panel, gaps = _inputs(missing_index)
    with pytest.raises(ValueError, match=f"{dates[missing_index].date()}: MISSING_UNEXPLAINED"):
        PITPortfolioEnv(book, predictions, panel, PITPortfolioEnvConfig(max_book=2), session_gaps=gaps)


def test_whole_session_proven_suspension_keeps_clock_and_frozen_holdings():
    dates, book, predictions, panel, gaps = _inputs(classification="SUSPENDED")
    env = PITPortfolioEnv(book, predictions, panel, PITPortfolioEnvConfig(max_book=2), session_gaps=gaps)
    assert env.execution_dates == list(dates[1:4])
    assert env.reward_end_dates == list(dates[2:5])
    np.testing.assert_allclose(env.slot_ret[1, :2], [0.0, 0.0])
    np.testing.assert_allclose(env.slot_ret[2, :2], [0.05, 0.1])
    assert env.slot_frozen[2, :2].all()
    env.reset()
    env.step(np.zeros(env.action_space.shape))
    _, _, _, _, before = env.step(np.zeros(env.action_space.shape))
    _, _, _, _, after = env.step(np.ones(env.action_space.shape))
    for symbol in ("A", "B"):
        assert after["weights"][symbol] == pytest.approx(before["drift_weights"][symbol])
    assert after["execution_date"] == str(dates[3].date())
    assert after["reward_end_date"] == str(dates[4].date())


def test_suspension_evidence_for_one_name_does_not_cover_another_missing_name():
    _, book, predictions, panel, gaps = _inputs(classification="SUSPENDED")
    with pytest.raises(ValueError, match="B.*NO_GAP_EVIDENCE"):
        PITPortfolioEnv(book, predictions, panel, PITPortfolioEnvConfig(max_book=2), session_gaps=gaps.iloc[:1])


def test_reward_cutoff_uses_the_gap_session_without_skipping_or_reading_beyond_it():
    dates, book, predictions, panel, gaps = _inputs(classification="SUSPENDED")
    book = pd.DataFrame({"A": [0.5] * 5, "B": [0.5] * 5}, index=dates[:5])
    future_gap = pd.DataFrame([
        dict(trade_date=dates[6], symbol=symbol, classification="MISSING_UNEXPLAINED")
        for symbol in ("A", "B")
    ])
    panel = panel[panel.trade_date != dates[6]]
    env = PITPortfolioEnv(
        book, predictions, panel,
        PITPortfolioEnvConfig(max_book=2, reward_end_date_limit=str(dates[4].date())),
        session_gaps=pd.concat([gaps, future_gap], ignore_index=True),
    )
    assert env.dates == list(dates[:3])
    assert env.reward_end_dates == list(dates[2:5])


def test_sparse_book_cannot_bridge_a_known_gap_session():
    dates, _, predictions, panel, gaps = _inputs(classification="SUSPENDED")
    book = pd.DataFrame({"A": [0.5] * 4, "B": [0.5] * 4}, index=dates[[0, 1, 2, 4]])
    with pytest.raises(ValueError, match="consecutive execution sessions"):
        PITPortfolioEnv(book, predictions, panel, PITPortfolioEnvConfig(max_book=2), session_gaps=gaps)


def test_gap_register_beyond_the_panel_span_does_not_extend_the_clock():
    dates, book, predictions, panel, gaps = _inputs(classification="SUSPENDED")
    outside = pd.DataFrame([
        dict(trade_date=pd.Timestamp("2030-01-02"), symbol=symbol, classification="MISSING_UNEXPLAINED")
        for symbol in ("A", "B")
    ])
    env = PITPortfolioEnv(
        book, predictions, panel, PITPortfolioEnvConfig(max_book=2),
        session_gaps=pd.concat([gaps, outside], ignore_index=True),
    )
    assert env.execution_dates == list(dates[1:4])
