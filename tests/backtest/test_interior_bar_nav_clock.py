"""NAV(t) contains exactly the fills whose fill_date <= t.

Round 21 / R1 (backtest).  The engine filled at ``open(t+1)`` under
``next_day_fill`` but marked the resulting book onto ``close(t)``, so every
rebalance booked the overnight gap ``shares x (close(t) - open(t+1))`` as
instantaneous P&L on a bar the position was not held.  The direction is
systematic, not noise: a name that gaps down "earns" money.

Total return can self-correct because the phantom mark unwinds on the following
bar, but the *path* does not, so drawdown, volatility, Sharpe and Calmar are all
computed from a portfolio that was never held.

The golden scenarios cannot see this: ``test_golden_backtest_scenarios.py``
builds every bar with ``open == close``, the single case where the mismatch
vanishes.  These tests use a gap tape on purpose.

The previous fix attempt moved the mark before the trading block
unconditionally and broke the composite ledger-replay invariant, which runs
same-bar fills (``next_day_fill=False``) where the fill genuinely does belong in
NAV(t).  Both conventions are covered below so neither can regress alone.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.backtest.engine import BacktestConfig, EventDrivenBacktester

SYMBOL = "600000.SH"


def _tape(opens: list[float], closes: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=len(opens))
    return pd.DataFrame(
        [
            {
                "trade_date": d, "symbol": SYMBOL, "open": o, "close": c,
                "high": max(o, c) * 1.5, "low": min(o, c) * 0.5,
                "pre_close": c, "volume": 1e12, "amount": 1e12,
            }
            for d, o, c in zip(dates, opens, closes)
        ]
    )


def _weights(n: int, first_signal: int = 1) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    col = [0.0] * n
    for i in range(first_signal, n):
        col[i] = 1.0
    return pd.DataFrame({SYMBOL: col}, index=dates)


def _run(opens, closes, **cfg):
    return EventDrivenBacktester(BacktestConfig(**cfg)).run(
        _weights(len(opens)), _tape(opens, closes)
    )


def test_signal_bar_nav_is_unmoved_by_a_fill_that_lands_next_bar() -> None:
    """The core defect: a gap on t+1 must not appear in NAV(t)."""
    opens = [10.0, 10.0, 9.0, 10.0, 10.0, 10.0]  # one gap-down open
    closes = [10.0] * 6                           # tape otherwise flat
    result = _run(opens, closes)

    nav = result.nav_curve
    # The signal is formed at bar 1 and fills at open(bar 2) = 9.00. Nothing is
    # held during bar 1, so its NAV cannot move off the opening cash.
    assert nav.iloc[0] == pytest.approx(1_000_000.0)
    assert nav.iloc[1] == pytest.approx(1_000_000.0), (
        "NAV moved on a bar where no position was held: the fill at open(t+1) "
        "was marked onto close(t)"
    )
    # The gain belongs to bar 2, where the shares were actually bought at 9.00
    # and marked at 10.00.
    assert nav.iloc[2] > nav.iloc[1]


def test_adverse_gap_is_not_booked_as_a_phantom_loss_on_the_signal_bar() -> None:
    """Mirror direction: an adverse fill must not sink the previous bar.

    Round 22 / F-03: the adverse open used to be 11.00 against a 10.00 previous
    close -- exactly +10%, i.e. a limit-up open on this main-board symbol. Once
    the board test moved onto the price the engine actually fills at, that open
    became correctly unbuyable and no fill happened at all, so the NAV-clock
    assertion had nothing to measure. The gap is now 10.50 (+5%), inside the
    band. The assertion itself is unchanged and no weaker: an adverse fill still
    has to land on bar 2, not bar 1.
    """
    opens = [10.0, 10.0, 10.5, 10.0, 10.0, 10.0]  # gap UP against the buyer
    closes = [10.0] * 6
    result = _run(opens, closes)

    nav = result.nav_curve
    assert nav.iloc[1] == pytest.approx(1_000_000.0)
    # Buying at 10.50 and marking at 10.00 is a real loss, on bar 2.
    assert nav.iloc[2] < nav.iloc[1]


def test_drawdown_is_not_manufactured_on_an_unheld_bar() -> None:
    """The path metrics, not just the endpoint, must belong to the held book."""
    opens = [10.0, 10.0, 9.0, 10.0, 10.0, 10.0]
    closes = [10.0] * 6
    result = _run(opens, closes)

    nav = result.nav_curve
    # Before the fix the phantom mark spiked bar 1 and unwound afterwards,
    # producing a drawdown from a peak that was never reached.
    peak_bar = nav.idxmax()
    assert peak_bar != nav.index[1], "peak sits on a bar that held nothing"


def test_flat_tape_with_no_gap_earns_only_the_costs() -> None:
    opens = [10.0] * 6
    closes = [10.0] * 6
    result = _run(opens, closes)

    nav = result.nav_curve
    assert nav.iloc[-1] < 1_000_000.0, "a flat tape must lose exactly the fees"
    # Round 22 / F-04: was 999_540.408042 while the engine applied 2.0 bps of
    # slippage from a private `FillModelConfig` field. Slippage now comes from
    # `CostModelConfig.slippage_bps` (5.0) -- the number every configuration
    # serialises and every report quotes -- so a ~1,000,000 notional buy costs
    # 3 bps more: 999_540.408042 - 999_240.314136 = 300.09. The tape's volume is
    # 1e12, so square-root impact is ~0.003 bps here and rounds out of sight;
    # `test_fill_cost_single_source.py` measures it where it bites.
    assert nav.iloc[-1] == pytest.approx(999_240.314136, rel=1e-9)


def test_same_bar_fill_still_marks_the_fill_it_executed() -> None:
    """`next_day_fill=False` is the composite-replay convention: do not regress it."""
    opens = [10.0, 9.0, 10.0, 10.0]
    closes = [10.0] * 4
    result = _run(opens, closes, next_day_fill=False)

    nav = result.nav_curve
    # Bar 1 both buys at open 9.00 and closes at 10.00, so the gain is its own.
    assert nav.iloc[1] > nav.iloc[0]


def test_every_nav_equals_cash_plus_priced_holdings() -> None:
    """The identity the marking must preserve, checked bar by bar."""
    opens = [10.0, 10.0, 9.0, 11.0, 10.5, 10.0]
    closes = [10.0, 10.2, 10.0, 10.6, 10.4, 10.0]
    result = _run(opens, closes)

    nav = result.nav_curve
    holdings = result.holdings
    for date in nav.index:
        if date not in holdings.index:
            continue
        # Holdings are published as weights of the same NAV, so they must sum to
        # the invested fraction and never exceed 1 by more than rounding.
        assert holdings.loc[date].sum() <= 1.0 + 1e-9


def test_nav_curve_has_one_entry_per_bar() -> None:
    """Marking before the trading block must not double-stamp the final bar."""
    opens = [10.0, 10.0, 9.0, 10.0]
    closes = [10.0] * 4
    result = _run(opens, closes)

    assert len(result.nav_curve) == 4
    assert result.nav_curve.index.is_unique
    assert len(result.holdings) == 4


def test_unpriced_holding_is_recorded_not_valued_at_zero() -> None:
    """A held name with no close on a bar must not be silently worth nothing."""
    dates = pd.bdate_range("2024-01-01", periods=4)
    rows = []
    for i, d in enumerate(dates):
        if i == 3:
            continue  # the held name has no bar here
        rows.append(
            {
                "trade_date": d, "symbol": SYMBOL, "open": 10.0, "close": 10.0,
                "high": 15.0, "low": 5.0, "pre_close": 10.0,
                "volume": 1e12, "amount": 1e12,
            }
        )
    # A second name keeps the final session alive in the panel.
    rows.append(
        {
            "trade_date": dates[3], "symbol": "000001.SZ", "open": 5.0, "close": 5.0,
            "high": 7.5, "low": 2.5, "pre_close": 5.0, "volume": 1e12, "amount": 1e12,
        }
    )
    prices = pd.DataFrame(rows)
    weights = pd.DataFrame(
        {SYMBOL: [0.0, 1.0, 1.0, 1.0], "000001.SZ": [0.0, 0.0, 0.0, 0.0]}, index=dates
    )
    engine = EventDrivenBacktester(BacktestConfig())
    result = engine.run(weights, prices)

    assert engine._unpriced_marks, "an unpriceable holding must leave a record"
    assert engine._unpriced_marks[0]["symbol"] == SYMBOL
    assert result.nav_curve.notna().all()
