"""The clean-room engine must not book P&L on a day the position was not held.

Round 21 / R1 (backtest) reproduced the legacy defect: a flat tape with a
single overnight gap turns a −0.046% strategy into +11.05% and flips the
reported Sharpe from −7.10 to +7.10, because the legacy engine fills at
``open(T+1)`` and marks the resulting book onto ``close(T)``.

These tests pin the clean-room contract instead: a book decided at ``close(T)``
executes at ``close(T+1)`` and first appears in ``NAV(T+1)``, so the fill price
and the valuation price are the same observation and no gap can be booked.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.clean_room.engine import (
    CleanRoomResult,
    CostConfig,
    compute_metrics,
    run_backtest,
)

SYMBOL = "600000.SH"


def _panel(closes: list[float], *, amount: float = 1e12) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame(
        [
            {"symbol": SYMBOL, "trade_date": d, "close": c, "amount": amount}
            for d, c in zip(dates, closes)
        ]
    )


def _book(dates: pd.DatetimeIndex, weight: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({SYMBOL: [weight] * len(dates)}, index=dates)


def _free() -> CostConfig:
    """Costs off, so a test about the clock is only about the clock."""
    return CostConfig(
        commission_rate=0.0, min_commission=0.0, stamp_duty_rate=0.0,
        transfer_fee_rate=0.0, slippage_bps=0.0, impact_alpha_bps=0.0,
    )


def test_flat_tape_earns_nothing_regardless_of_intraday_gaps() -> None:
    """The scenario that broke the legacy engine.

    Every close is 10.00. Whatever happened between closes, a book marked and
    filled on closes cannot make money here.
    """
    panel = _panel([10.0] * 6)
    dates = pd.DatetimeIndex(panel["trade_date"].unique())
    result = run_backtest(_book(dates), panel, initial_cash=1_000_000.0, config=_free())

    assert result.nav.notna().all()
    # Held from the second session onward, always at 10.00.
    assert result.nav.iloc[-1] == pytest.approx(1_000_000.0, abs=1e-6)
    assert result.metrics["total_return"] == pytest.approx(0.0, abs=1e-9)
    assert result.metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-9)


def test_gain_is_booked_on_the_session_the_price_moved() -> None:
    panel = _panel([10.0, 10.0, 11.0, 11.0])
    dates = pd.DatetimeIndex(panel["trade_date"].unique())
    result = run_backtest(_book(dates), panel, initial_cash=1_000_000.0, config=_free())

    navs = result.nav.tolist()
    # Session 0: no prior decision, still all cash.
    assert navs[0] == pytest.approx(1_000_000.0)
    # Session 1: the session-0 book executes at close(1)=10.00. Buying at the
    # same price it is marked at cannot change NAV.
    assert navs[1] == pytest.approx(1_000_000.0)
    # Session 2: price moves 10 -> 11 while fully invested.
    assert navs[2] == pytest.approx(1_100_000.0, rel=1e-9)


def test_position_is_never_marked_before_it_is_bought() -> None:
    """A book decided on T must not appear in NAV(T)."""
    panel = _panel([10.0, 20.0, 20.0])
    dates = pd.DatetimeIndex(panel["trade_date"].unique())
    # Only decide on the first session.
    book = pd.DataFrame({SYMBOL: [1.0]}, index=[dates[0]])
    result = run_backtest(book, panel, initial_cash=1_000_000.0, config=_free())

    # If the engine marked the session-0 book at close(0)=10 and filled at
    # close(1)=20 it would fabricate a 100% gain. It must not.
    assert result.nav.iloc[0] == pytest.approx(1_000_000.0)
    assert result.nav.iloc[1] == pytest.approx(1_000_000.0)


def test_costs_make_a_flat_tape_lose_exactly_the_fees() -> None:
    panel = _panel([10.0] * 4)
    dates = pd.DatetimeIndex(panel["trade_date"].unique())
    config = CostConfig(
        commission_rate=0.00025, min_commission=0.0, stamp_duty_rate=0.0005,
        transfer_fee_rate=0.00001, slippage_bps=5.0, impact_alpha_bps=0.0,
    )
    result = run_backtest(_book(dates), panel, initial_cash=1_000_000.0, config=config)

    assert result.nav.iloc[-1] < 1_000_000.0, "a flat tape with costs must lose"
    assert result.costs.sum() > 0
    # The loss is exactly the fees paid: no gap can be booked on a flat tape.
    assert result.nav.iloc[-1] == pytest.approx(
        1_000_000.0 - result.costs.sum(), rel=1e-9
    )


def test_stamp_duty_is_charged_on_sells_only() -> None:
    panel = _panel([10.0] * 4)
    dates = pd.DatetimeIndex(panel["trade_date"].unique())
    config = CostConfig(
        commission_rate=0.0, min_commission=0.0, stamp_duty_rate=0.001,
        transfer_fee_rate=0.0, slippage_bps=0.0, impact_alpha_bps=0.0,
    )
    buy_only = pd.DataFrame({SYMBOL: [1.0, 1.0, 1.0, 1.0]}, index=dates)
    round_trip = pd.DataFrame({SYMBOL: [1.0, 0.0, 0.0, 0.0]}, index=dates)

    buy_cost = run_backtest(buy_only, panel, config=config).costs.sum()
    trip_cost = run_backtest(round_trip, panel, config=config).costs.sum()

    assert buy_cost == pytest.approx(0.0, abs=1e-9), "buys pay no stamp duty"
    assert trip_cost > 0.0, "the sell leg must pay stamp duty"


def test_participation_cap_refuses_the_excess_instead_of_assuming_a_fill() -> None:
    panel = _panel([10.0] * 2, amount=1_000_000.0)  # one thin trading session
    dates = pd.DatetimeIndex(panel["trade_date"].unique())
    config = CostConfig(
        commission_rate=0.0, min_commission=0.0, stamp_duty_rate=0.0,
        transfer_fee_rate=0.0, slippage_bps=0.0, impact_alpha_bps=0.0,
        max_participation=0.10,
    )
    result = run_backtest(
        _book(dates), panel, initial_cash=10_000_000.0, config=config
    )

    # 10% of a 1,000,000 CNY bar is 100,000 CNY, not the 10m CNY the book
    # asked for. The excess is refused, not assumed filled.
    invested = result.weights.iloc[-1].get(SYMBOL, 0.0) * result.nav.iloc[-1]
    assert invested == pytest.approx(100_000.0, rel=1e-6)


def test_held_but_unpriced_name_makes_nav_unknown_not_zero() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    rows = [
        {"symbol": SYMBOL, "trade_date": dates[0], "close": 10.0, "amount": 1e12},
        {"symbol": SYMBOL, "trade_date": dates[1], "close": 10.0, "amount": 1e12},
        # Session 2 has a bar for a different name only: the held name is unpriced.
        {"symbol": "000001.SZ", "trade_date": dates[2], "close": 5.0, "amount": 1e12},
    ]
    panel = pd.DataFrame(rows)
    book = pd.DataFrame({SYMBOL: [1.0]}, index=[dates[0]])
    result = run_backtest(book, panel, initial_cash=1_000_000.0, config=_free())

    assert pd.isna(result.nav.iloc[2]), "unpriced holding must not be valued at 0"
    assert result.unpriced_days, "the unpriced day must be recorded, not swallowed"
    assert any("unpriced_nav_days" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Metrics: a statistic that cannot be computed is None, never 0.0
# ---------------------------------------------------------------------------

def test_metrics_report_none_rather_than_zero_when_unmeasurable() -> None:
    empty = pd.Series(dtype=float)
    metrics = compute_metrics(empty, empty)

    assert metrics["max_drawdown"] is None
    assert metrics["sharpe"] is None
    assert metrics["annualised_return"] is None
    assert metrics["n_days"] == 0.0


def test_empty_book_is_reported_as_unmeasured() -> None:
    result = run_backtest(pd.DataFrame(), _panel([10.0, 10.0]))

    assert isinstance(result, CleanRoomResult)
    assert result.measured is False
    assert "empty_target_weights" in result.notes


def test_measured_flag_is_true_only_with_a_complete_metric_set() -> None:
    panel = _panel([10.0, 10.5, 10.2, 10.8, 11.0])
    dates = pd.DatetimeIndex(panel["trade_date"].unique())
    result = run_backtest(_book(dates), panel, config=_free())

    assert result.measured is True
    assert result.metrics["max_drawdown"] is not None
    assert result.metrics["sharpe"] is not None
