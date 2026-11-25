"""FRESH blind runner must use the canonical execution clock (Cycle 32).

#119 (2026-08-11) deleted ``baseline_protocol._apply_delay`` and made the
strict simulator the sole owner of the T-close -> next-session mapping. The
FRESH daily runner kept calling the deleted function; the crash stayed hidden
while the panel was stale (the weights step was BLOCKED_UPSTREAM) and surfaced
on 2026-11-25 as ``orders=FAILED`` once the catch-up made the data current.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

import scripts.fresh_blind_daily as runner

SOURCE = Path(runner.__file__).read_text(encoding="utf-8")


def _frozen_pre119_apply_delay(tw: pd.DataFrame, trade_dates, delay_days: int) -> pd.DataFrame:
    """Verbatim copy of ``baseline_protocol._apply_delay`` as of 44d3fca^ (pre-#119)."""
    if delay_days <= 0:
        return tw
    date_index = pd.DatetimeIndex(sorted(trade_dates))
    positions = date_index.searchsorted(tw.index) + delay_days
    keep = positions < len(date_index)
    tw = tw.iloc[keep]
    tw.index = date_index[positions[keep]]
    return tw[~tw.index.duplicated(keep="last")].sort_index()


def _book(dates, symbols=("600000.SH", "000001.SZ", "300750.SZ")) -> pd.DataFrame:
    rows = []
    for i, _ in enumerate(dates):
        w = [0.0] * len(symbols)
        w[i % len(symbols)] = 1.0
        rows.append(w)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=list(symbols))


def test_runner_never_calls_removed_pre_shift() -> None:
    calls = [
        node for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.Attribute) and node.attr == "_apply_delay"
    ]
    assert calls == [], "the strict simulator owns the clock; do not pre-shift targets"


def test_holdings_in_effect_matches_frozen_record_semantics() -> None:
    sessions = list(pd.bdate_range("2026-11-02", periods=8))
    book = _book(sessions)
    expected = _frozen_pre119_apply_delay(book, sessions, 1).iloc[-1]
    got = runner.holdings_in_effect(book, sessions)
    pd.testing.assert_series_equal(got, expected, check_names=False)
    # In effect on the last session = the previous session's signal, not today's.
    pd.testing.assert_series_equal(got, book.iloc[-2], check_names=False)


def test_holdings_in_effect_with_sparse_signal_sessions() -> None:
    sessions = list(pd.bdate_range("2026-11-02", periods=10))
    signal = [sessions[0], sessions[3], sessions[4], sessions[9]]
    book = _book(signal)
    expected = _frozen_pre119_apply_delay(book, sessions, 1).iloc[-1]
    got = runner.holdings_in_effect(book, sessions)
    pd.testing.assert_series_equal(got, expected, check_names=False)
    pd.testing.assert_series_equal(got, book.loc[sessions[4]], check_names=False)


def test_holdings_in_effect_fails_closed() -> None:
    sessions = list(pd.bdate_range("2026-11-02", periods=3))
    with pytest.raises(ValueError):
        runner.holdings_in_effect(_book([]), sessions)
    with pytest.raises(ValueError):
        runner.holdings_in_effect(_book([sessions[-1]]), sessions)

# The simulator side (signal T -> one execution at T+1) is pinned by
# tests/backtest/test_baseline_protocol_execution_clock.py.


def test_unshifted_book_first_executes_next_session(tmp_path) -> None:
    """The runner's book (signal-dated, not pre-shifted) fills once at T+1."""
    from quantagent.backtest.ashare_execution_simulator import AShareExecutionSimulationConfig
    from quantagent.backtest.strict_v8 import run_strict_backtest_v8

    sessions = pd.bdate_range("2026-11-02", periods=4)
    market = pd.DataFrame([
        {"trade_date": d, "symbol": "600000.SH", "close": 10.0 + i,
         "volume": 1_000_000.0, "amount": 10_000_000.0, "is_suspended": False,
         "is_st": False, "is_limit_up": False, "is_limit_down": False}
        for i, d in enumerate(sessions)
    ])
    book = pd.DataFrame({"600000.SH": [1.0]}, index=pd.DatetimeIndex([sessions[0]]))
    result = run_strict_backtest_v8(
        book.sort_index(), market,
        config=AShareExecutionSimulationConfig(
            initial_cash=100_000.0, slippage_bps=0.0, audit_log_dir=str(tmp_path / "audit")),
    )
    assert not result.trades.empty
    assert set(pd.to_datetime(result.trades["execution_date"])) == {sessions[1]}
    assert pd.Timestamp(result.trades.iloc[0]["signal_date"]) == sessions[0]
