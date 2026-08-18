"""The board test and the execution price must be the same measurement.

Round 22 / R1 (backtest), closing F-03 from `docs/audits/round21/01_backtest.md`.

`EventDrivenBacktester` fills at `BacktestConfig.fill_price_column` (default the
next bar's `open`) but used to decide tradability from `limit_up_mask` /
`limit_down_mask`, which compare the bar's *close* against the previous close.
Criterion and execution price were therefore two measurements of two different
moments, and the mismatch is wrong in both directions:

* **false block** — the bar opens inside the band and only seals at the close.
  The order would have filled at the open; the engine refused it. Visible in the
  reject log, so at worst it understates the strategy.
* **false fill** — the bar opens sealed and comes off the board intraday. The
  close-based flag reads False, so the engine buys at an open price nobody could
  reach. This one is a *favourable*-direction bias and it is invisible: there is
  no reject to find, only a trade that looks entirely normal.

Measured on 30 real A-share symbols over 2022-01..2026-05 (31,650 bars): 210
close-based limit-ups against 31 open-based ones — 189 false blocks and 10 bars
that were sealed at the open and got filled anyway.

A one-word board (`high == low` at the band) is tested separately, because it is
the case neither limit mask can see when the previous close is unknown.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.backtest.engine import BacktestConfig, EventDrivenBacktester
from quantagent.quant_math.ashare import (
    limit_down_mask,
    limit_up_mask,
    one_word_board_mask,
)

SYMBOL = "600000.SH"  # main board: +/-10% band
NAV = 1_000_000.0


def _prices(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["symbol"] = SYMBOL
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["high"] = frame["high"].fillna(frame[["open", "close"]].max(axis=1))
    frame["low"] = frame["low"].fillna(frame[["open", "close"]].min(axis=1))
    frame["volume"] = frame.get("volume", pd.Series(1e12, index=frame.index)).fillna(1e12)
    frame["amount"] = frame["close"] * frame["volume"]
    frame["pre_close"] = frame["close"]
    return frame


def _weights(dates: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({SYMBOL: values}, index=pd.to_datetime(dates))


def _run(prices: pd.DataFrame, weights: pd.DataFrame, **cfg) -> object:
    return EventDrivenBacktester(BacktestConfig(initial_nav=NAV, **cfg)).run(weights, prices)


# ---------------------------------------------------------------------------
# 1. The two directions of the mismatch
# ---------------------------------------------------------------------------
def test_a_bar_that_opens_tradable_fills_even_though_it_closes_sealed() -> None:
    """False block: sealing at 15:00 does not retract the 09:30 print."""
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    prices = _prices([
        {"trade_date": dates[0], "open": 10.0, "close": 10.0, "high": None, "low": None},
        # opens at +2%, runs to a sealed +10% close. The open was buyable.
        {"trade_date": dates[1], "open": 10.2, "close": 11.0, "high": 11.0, "low": 10.1},
        {"trade_date": dates[2], "open": 11.0, "close": 11.0, "high": None, "low": None},
    ])
    result = _run(prices, _weights(dates, [0.10, 0.10, 0.10]))

    fills = result.trades[result.trades["trade_date"] == pd.Timestamp(dates[1])]
    assert len(fills) == 1, "an order filled at a tradable open was refused by the close"
    assert float(fills.iloc[0]["price"]) == pytest.approx(10.2, rel=2e-3)


def test_a_bar_that_opens_sealed_is_not_filled_when_it_closes_off_the_board() -> None:
    """False fill: the dangerous half, because it leaves no reject to find."""
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    prices = _prices([
        {"trade_date": dates[0], "open": 10.0, "close": 10.0, "high": None, "low": None},
        # sealed +10% at the open, opens the board and closes at -5%.
        {"trade_date": dates[1], "open": 11.0, "close": 9.5, "high": 11.0, "low": 9.4},
        {"trade_date": dates[2], "open": 9.5, "close": 9.5, "high": None, "low": None},
    ])
    result = _run(prices, _weights(dates, [0.10, 0.10, 0.10]))

    on_bar = result.trades[result.trades["trade_date"] == pd.Timestamp(dates[1])]
    assert on_bar.empty, "filled at a limit-up open that no buyer could have reached"
    blocked = result.rejects[result.rejects["trade_date"] == pd.Timestamp(dates[1])]
    assert list(blocked["reason"]) == ["limit_up_no_buy"]


def test_a_sell_into_a_bar_that_opens_at_limit_down_is_refused() -> None:
    """Same rule, sell side: the exit prices at the open, so the open decides."""
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
    prices = _prices([
        {"trade_date": dates[0], "open": 10.0, "close": 10.0, "high": None, "low": None},
        {"trade_date": dates[1], "open": 10.0, "close": 10.0, "high": None, "low": None},
        {"trade_date": dates[2], "open": 10.0, "close": 10.0, "high": None, "low": None},
        # opens sealed at -10%, recovers to flat by the close.
        {"trade_date": dates[3], "open": 9.0, "close": 10.0, "high": 10.1, "low": 9.0},
        {"trade_date": dates[4], "open": 10.0, "close": 10.0, "high": None, "low": None},
    ])
    result = _run(prices, _weights(dates, [0.10, 0.10, 0.0, 0.0, 0.0]))

    limit_day = pd.Timestamp(dates[3])
    sells = result.trades[
        (result.trades["side"] == "sell") & (result.trades["trade_date"] == limit_day)
    ]
    assert sells.empty, "sold at a limit-down open the venue would not have crossed"
    assert "limit_down_no_sell" in set(result.rejects[result.rejects["trade_date"] == limit_day]["reason"])


# ---------------------------------------------------------------------------
# 2. The one-word board
# ---------------------------------------------------------------------------
def test_one_word_board_is_flagged_on_the_reject_row() -> None:
    """A sealed-all-day bar must be distinguishable from a merely-at-the-band one."""
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    prices = _prices([
        {"trade_date": dates[0], "open": 10.0, "close": 10.0, "high": 10.5, "low": 9.5},
        # open == high == low == close == +10%: one price, all session.
        {"trade_date": dates[1], "open": 11.0, "close": 11.0, "high": 11.0, "low": 11.0},
        {"trade_date": dates[2], "open": 11.0, "close": 11.0, "high": 11.4, "low": 10.9},
    ])
    result = _run(prices, _weights(dates, [0.10, 0.10, 0.10]))

    blocked = result.rejects[result.rejects["trade_date"] == pd.Timestamp(dates[1])]
    assert list(blocked["reason"]) == ["limit_up_no_buy"]
    assert "one_word_board" in result.rejects.columns
    assert bool(blocked.iloc[0]["one_word_board"]) is True


def test_a_one_word_up_board_blocks_the_buy_but_not_the_exit() -> None:
    """Only the queuing side is refused. Selling into a sealed bid crosses."""
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
    prices = _prices([
        {"trade_date": dates[0], "open": 10.0, "close": 10.0, "high": 10.5, "low": 9.5},
        {"trade_date": dates[1], "open": 10.0, "close": 10.0, "high": 10.5, "low": 9.5},
        {"trade_date": dates[2], "open": 10.0, "close": 10.0, "high": 10.5, "low": 9.5},
        # sealed +10% for the whole session; the exit lands here.
        {"trade_date": dates[3], "open": 11.0, "close": 11.0, "high": 11.0, "low": 11.0},
        {"trade_date": dates[4], "open": 11.0, "close": 11.0, "high": 11.4, "low": 10.9},
    ])
    result = _run(prices, _weights(dates, [0.10, 0.10, 0.0, 0.0, 0.0]))

    board_day = pd.Timestamp(dates[3])
    sells = result.trades[
        (result.trades["side"] == "sell") & (result.trades["trade_date"] == board_day)
    ]
    assert len(sells) == 1, "a sell into a sealed up board was refused; it would have crossed"
    assert float(sells.iloc[0]["price"]) == pytest.approx(11.0, rel=1e-3)


def test_a_zero_range_bar_with_no_previous_close_refuses_both_sides() -> None:
    """Fail-closed. The bar's own shape says 'board'; nothing says which one.

    This is the case the limit masks structurally cannot answer: with no usable
    `prev_close` both `flag_up` and `flag_down` are False, so before this
    criterion existed the engine read an unknown board as tradable.
    """
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    prices = _prices([
        # first bar of the symbol: no previous close to measure a band against.
        {"trade_date": dates[0], "open": 10.0, "close": 10.0, "high": 10.0, "low": 10.0},
        {"trade_date": dates[1], "open": 10.0, "close": 10.0, "high": 10.5, "low": 9.5},
        {"trade_date": dates[2], "open": 10.0, "close": 10.0, "high": 10.5, "low": 9.5},
    ])
    # Same-bar fill so the order lands on the bar that has no reference.
    result = _run(prices, _weights(dates, [0.10, 0.0, 0.0]), next_day_fill=False)

    first = pd.Timestamp(dates[0])
    assert result.trades.empty, "filled into a board whose direction is unknown"
    blocked = result.rejects[result.rejects["trade_date"] == first]
    assert list(blocked["reason"]) == ["one_word_board_no_buy"]
    assert bool(blocked.iloc[0]["one_word_board"]) is True


def test_one_word_criterion_is_not_applicable_on_a_tape_that_carries_no_ranges() -> None:
    """`high == low` everywhere means the range was never measured, not sealed.

    Reporting every bar of such a tape as a board would be manufacturing a
    measurement the source does not contain — the mirror of the defect this
    repo keeps finding. The criterion says 'not applicable'; the limit masks
    still gate execution.
    """
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    prices = _prices([
        {"trade_date": d, "open": 10.0, "close": 10.0, "high": 10.0, "low": 10.0}
        for d in dates
    ])
    assert not one_word_board_mask(prices).any()

    result = _run(prices, _weights(dates, [0.10, 0.10, 0.10]))
    assert not result.trades.empty, "a flat synthetic tape was blocked as a board"


# ---------------------------------------------------------------------------
# 3. The masks themselves
# ---------------------------------------------------------------------------
def test_limit_masks_answer_differently_per_price_basis() -> None:
    frame = _prices([
        {"trade_date": "2026-01-05", "open": 10.0, "close": 10.0, "high": None, "low": None},
        {"trade_date": "2026-01-06", "open": 10.2, "close": 11.0, "high": 11.0, "low": 10.1},
        # prev_close is now 11.00, so the band top is 12.10: sealed at the open,
        # off the board by the close.
        {"trade_date": "2026-01-07", "open": 12.1, "close": 10.45, "high": 12.1, "low": 10.4},
    ])
    close_up = limit_up_mask(frame).tolist()
    open_up = limit_up_mask(frame, price_column="open").tolist()
    assert close_up == [False, True, False]
    assert open_up == [False, False, True]
    assert limit_down_mask(frame, price_column="open").tolist() == [False, False, False]


def test_a_board_test_on_a_price_the_tape_does_not_carry_is_fail_loud() -> None:
    """Silently answering on some other column is how a board flag lies."""
    frame = _prices([
        {"trade_date": "2026-01-05", "open": 10.0, "close": 10.0, "high": None, "low": None},
        {"trade_date": "2026-01-06", "open": 10.0, "close": 10.0, "high": None, "low": None},
    ])
    with pytest.raises(KeyError, match="price_column"):
        limit_up_mask(frame, price_column="vwap")
    with pytest.raises(KeyError, match="price_column"):
        limit_down_mask(frame, price_column="vwap")
