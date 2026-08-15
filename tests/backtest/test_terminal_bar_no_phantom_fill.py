"""A next-day-fill policy must not execute the final bar on its own bar.

`fill_date = dates[i+1] if (next_day_fill and i+1 < len(dates)) else date` fell
back to same-bar execution on the last bar. That both violates the policy and
leaves a position no later bar can unwind, so the phantom mark never washes out
and the *total* return is wrong -- not just the path.

It is not a corner case. `v7_pipeline_service.py:1550` sets
`weights.iloc[-1] = target_weights` with every other row zero, so the terminal
row is the ONLY signal and 100% of fills took the fallback. Measured on a tape
whose closes never move, the engine reported +23.80% annualised with Sharpe 2.07
and zero drawdown; the honest answer is that nothing happened at all.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.backtest.engine import BacktestConfig, EventDrivenBacktester

SYMBOL = "600000.SH"


def _flat_tape(n: int = 40, price: float = 10.0) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.bdate_range("2026-01-01", periods=n)
    closes = [price] * n
    frame = pd.DataFrame({
        "trade_date": dates,
        "symbol": SYMBOL,
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": [1_000_000.0] * n,
        "amount": [price * 1_000_000.0] * n,
        "flag_up": False,
        "flag_down": False,
        "flag_susp": False,
    })
    return frame, dates


def _terminal_only_weights(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """The production shape: only the last row carries a target."""
    weights = pd.DataFrame(0.0, index=dates, columns=[SYMBOL])
    weights.iloc[-1] = 1.0
    return weights


class TestTerminalBarProducesNoReturn:
    def test_a_motionless_tape_yields_exactly_zero_return(self):
        prices, dates = _flat_tape()
        result = EventDrivenBacktester().run(
            target_weights=_terminal_only_weights(dates), prices=prices
        )
        nav = result.nav_curve.dropna()
        assert nav.iloc[-1] == pytest.approx(nav.iloc[0]), (
            "closes never moved, so NAV must not move either"
        )

    def test_a_motionless_tape_has_no_drawdown(self):
        prices, dates = _flat_tape()
        result = EventDrivenBacktester().run(
            target_weights=_terminal_only_weights(dates), prices=prices
        )
        nav = result.nav_curve.dropna()
        drawdown = float((nav / nav.cummax() - 1.0).min())
        assert drawdown == pytest.approx(0.0, abs=1e-12)

    def test_the_unfillable_terminal_order_is_recorded_not_silently_dropped(self):
        prices, dates = _flat_tape()
        result = EventDrivenBacktester().run(
            target_weights=_terminal_only_weights(dates), prices=prices
        )
        rejects = result.rejects
        assert not rejects.empty, "the order could not fill; that must leave a record"
        assert (rejects["reason"] == "no_next_session_to_fill").any()

    def test_nav_still_covers_every_bar(self):
        """The fix must not shorten the NAV series."""
        prices, dates = _flat_tape(n=40)
        result = EventDrivenBacktester().run(
            target_weights=_terminal_only_weights(dates), prices=prices
        )
        assert len(result.nav_curve.dropna()) == 40


class TestSameDayFillPolicyIsUnaffected:
    def test_next_day_fill_disabled_still_fills_on_the_final_bar(self):
        """Guard against over-reach: same-day policy legitimately fills there."""
        prices, dates = _flat_tape()
        result = EventDrivenBacktester(BacktestConfig(next_day_fill=False)).run(
            target_weights=_terminal_only_weights(dates), prices=prices
        )
        reasons = set(result.rejects["reason"]) if not result.rejects.empty else set()
        assert "no_next_session_to_fill" not in reasons
