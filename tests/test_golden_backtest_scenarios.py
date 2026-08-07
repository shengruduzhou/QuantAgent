"""Golden backtest scenarios: every number here is computed by hand.

The existing engine test asserted `nav_curve.notna().all()` and
`trades.shape[0] > 0`. That cannot distinguish a correct simulator from one
that charges the wrong stamp duty, settles T+0, or fills into a limit-up. These
scenarios are small enough to verify with a calculator and assert exact values,
so a change to the cost model or the A-share rules has to announce itself.

Every expected figure below is derived in the comment above it from the shipped
defaults in `CostModelConfig`:

    commission      2.5 bps on gross, minimum 5.00 CNY
    transfer fee    0.1 bps on gross
    stamp duty      5.0 bps on gross, SELL SIDE ONLY
    lot size        100 shares
    fills           next day, at that day's open

Slippage is *not* a fee. `AShareFillModel` moves the execution price away from
the reference by `slippage_bps (2.0) + impact_bps x filled/volume`, and the
trade log records the resulting cash difference. The engine used to charge
`cost.slippage_bps (5.0)` on top of that already-moved price, so its effective
slippage was ~7 bps while every configuration declared 5. Fees are therefore
computed on the *fill* price, never the reference price.
"""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.backtest.engine import BacktestConfig, EventDrivenBacktester
from quantagent.quant_math.transaction_cost import CostModelConfig

SYMBOL = "600000.SH"          # main board: 10% price limit
INITIAL_NAV = 1_000_000.0


def _prices(rows: list[dict]) -> pd.DataFrame:
    """Long-form OHLCV. `rows` carries date/open/close and optional overrides."""
    frame = pd.DataFrame(rows)
    frame["symbol"] = frame.get("symbol", SYMBOL)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    for column, default in (("high", None), ("low", None), ("volume", 100_000_000.0)):
        if column not in frame.columns:
            frame[column] = default
    frame["high"] = frame["high"].fillna(frame[["open", "close"]].max(axis=1))
    frame["low"] = frame["low"].fillna(frame[["open", "close"]].min(axis=1))
    frame["amount"] = frame["close"] * frame["volume"]
    return frame


def _run(prices: pd.DataFrame, weights: pd.DataFrame, **config) -> object:
    engine = EventDrivenBacktester(BacktestConfig(initial_nav=INITIAL_NAV, **config))
    return engine.run(weights, prices)


def _weights(dates, values, symbol: str = SYMBOL) -> pd.DataFrame:
    return pd.DataFrame({symbol: values}, index=pd.to_datetime(dates))


# --------------------------------------------------------------------------
# 1. Buy-side costs, to the cent
# --------------------------------------------------------------------------
def test_buy_costs_match_hand_calculation():
    """10% of 1,000,000 NAV at a 10.00 open, volume 100,000,000.

    impact bps  = 1.0 x 10,000 / 100,000,000  =  0.0001
    fill price  = 10.00 x (1 + 2.0001/10000)  = 10.0020001
    raw target  = 100,000 / 10.00             = 10,000 shares (100 lots)
    gross       = 10,000 x 10.0020001         = 100,020.001
    commission  = gross x 2.5/10000           =      25.00500  (above 5.00 floor)
    transfer    = gross x 0.1/10000           =       1.00020
    stamp duty  = 0 (buy side)
    slippage    = |10.0020001 - 10.00| x 10,000 = 20.001  (recorded, in the price)
    cash out    = gross + 25.00500 + 1.00020  = 100,046.00620
    """
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    prices = _prices([
        {"trade_date": dates[0], "open": 10.0, "close": 10.0},
        {"trade_date": dates[1], "open": 10.0, "close": 10.0},
        {"trade_date": dates[2], "open": 10.0, "close": 10.0},
    ])
    result = _run(prices, _weights(dates, [0.10, 0.10, 0.10]))

    buys = result.trades[result.trades["side"] == "buy"]
    assert len(buys) == 1
    trade = buys.iloc[0]
    assert trade["shares"] == 10_000
    assert trade["price"] == pytest.approx(10.0020001, abs=1e-7)
    assert trade["commission"] == pytest.approx(25.00500, abs=1e-4)
    assert trade["transfer_fee"] == pytest.approx(1.00020, abs=1e-4)
    assert trade["slippage"] == pytest.approx(20.001, abs=1e-3)

    # cash 899,953.9938 + 10,000 shares marked at the 10.00 close.
    assert result.nav_curve.loc[pd.Timestamp(dates[1])] == pytest.approx(999_953.9938, abs=0.01)


def test_the_commission_floor_applies_to_tiny_orders():
    """A 2,000 CNY order: 2.5 bps = 0.50, below the 5.00 minimum.

    gross      = 200 x 10.00        = 2,000.00
    commission = max(0.50, 5.00)    =     5.00   <- floor binds
    """
    dates = ["2026-01-05", "2026-01-06"]
    prices = _prices([
        {"trade_date": dates[0], "open": 10.0, "close": 10.0},
        {"trade_date": dates[1], "open": 10.0, "close": 10.0},
    ])
    result = _run(prices, _weights(dates, [0.002, 0.002]))

    trade = result.trades.iloc[0]
    assert trade["shares"] == 200
    assert trade["commission"] == pytest.approx(5.0)


# --------------------------------------------------------------------------
# 2. Sell-side costs — stamp duty is one-directional
# --------------------------------------------------------------------------
def test_sell_pays_stamp_duty_and_buy_does_not():
    """Stamp duty is charged on the sell side only, at 5 bps of gross."""
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
    prices = _prices([{"trade_date": d, "open": 10.0, "close": 10.0} for d in dates])
    result = _run(prices, _weights(dates, [0.10, 0.10, 0.0, 0.0, 0.0]))

    buys = result.trades[result.trades["side"] == "buy"]
    sells = result.trades[result.trades["side"] == "sell"]
    assert len(buys) == 1 and len(sells) == 1

    buy = buys.iloc[0]
    assert pd.isna(buy.get("stamp_duty")) or buy.get("stamp_duty", 0.0) == 0.0

    sell = sells.iloc[0]
    gross = sell["shares"] * sell["price"]
    assert sell["stamp_duty"] == pytest.approx(gross * 5.0 / 10_000, rel=1e-9)
    assert sell["commission"] == pytest.approx(gross * 2.5 / 10_000, rel=1e-9)
    assert sell["transfer_fee"] == pytest.approx(gross * 0.1 / 10_000, rel=1e-9)


def test_a_round_trip_at_a_flat_price_loses_exactly_its_costs():
    """NAV must fall by precisely the fees plus the two price slips.

    Reconstructed from the trade log rather than asserted as a round number, so
    the identity holds whatever the configured rates are.
    """
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
    prices = _prices([{"trade_date": d, "open": 10.0, "close": 10.0} for d in dates])
    result = _run(prices, _weights(dates, [0.10, 0.10, 0.0, 0.0, 0.0]))

    trades = result.trades
    fees = (
        trades["commission"].sum()
        + trades["transfer_fee"].sum()
        + trades.get("stamp_duty", pd.Series(0.0, index=trades.index)).fillna(0.0).sum()
    )
    slips = trades["slippage"].sum()
    assert result.nav_curve.iloc[-1] == pytest.approx(INITIAL_NAV - fees - slips, abs=0.01)


# --------------------------------------------------------------------------
# 3. T+1 settlement
# --------------------------------------------------------------------------
def test_shares_bought_today_cannot_be_sold_today():
    """T+1: a same-day exit must not produce a sell on the fill date."""
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    prices = _prices([{"trade_date": d, "open": 10.0, "close": 10.0} for d in dates])
    # Ask for a position on day 1 and a full exit on day 2. The buy fills on
    # day 2 (next-day fill), so the shares are frozen that day.
    result = _run(prices, _weights(dates, [0.10, 0.0, 0.0]))

    fills = result.trades.sort_values("trade_date")
    buy = fills[fills["side"] == "buy"].iloc[0]
    sells = fills[fills["side"] == "sell"]
    if not sells.empty:
        assert sells.iloc[0]["trade_date"] > buy["trade_date"], (
            "a sell on the buy's own fill date would be a T+0 settlement"
        )


def test_orders_fill_on_the_next_session_not_the_signal_date():
    """A day-1 signal executes at day 2's open, never at day 1's price."""
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    prices = _prices([
        {"trade_date": dates[0], "open": 10.0, "close": 10.0},
        # +2% keeps this well inside the 10% limit so nothing blocks the fill.
        {"trade_date": dates[1], "open": 10.2, "close": 10.2},
        {"trade_date": dates[2], "open": 10.2, "close": 10.2},
    ])
    result = _run(prices, _weights(dates, [0.10, 0.10, 0.10]))

    trade = result.trades.iloc[0]
    assert trade["trade_date"] == pd.Timestamp(dates[1])
    assert trade["price"] == pytest.approx(10.2, rel=1e-3)
    assert trade["price"] != pytest.approx(10.0, rel=1e-4)


# --------------------------------------------------------------------------
# 4. Price limits
# --------------------------------------------------------------------------
def test_a_buy_into_limit_up_is_rejected_not_filled():
    """Main board +10%: a 10.00 close then an 11.00 bar is limit-up."""
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    prices = _prices([
        {"trade_date": dates[0], "open": 10.0, "close": 10.0},
        # open == high == low == close == 11.00 == +10% -> sealed limit up
        {"trade_date": dates[1], "open": 11.0, "close": 11.0, "high": 11.0, "low": 11.0},
        {"trade_date": dates[2], "open": 11.0, "close": 11.0},
    ])
    result = _run(prices, _weights(dates, [0.10, 0.10, 0.10]))

    on_limit_day = result.trades[result.trades["trade_date"] == pd.Timestamp(dates[1])]
    assert on_limit_day.empty, "no shares may be bought into a sealed limit-up"
    # Blocking is not enough: the refusal has to be attributable. The block used
    # to be applied by `enforce_tradability` before the order loop, so the order
    # left no fill and no reason — it simply disappeared.
    assert not result.rejects.empty, "a sealed limit-up buy must be recorded as a reject"
    blocked = result.rejects[result.rejects["trade_date"] == pd.Timestamp(dates[1])]
    assert list(blocked["reason"]) == ["limit_up_no_buy"]
    assert list(blocked["symbol"]) == [SYMBOL]


def test_a_sell_into_limit_down_is_rejected_not_filled():
    """The exit signal has to land *on* the limit-down session.

    Buying into a limit-down is legal, and fills are next-day, so the position
    is established early and held past T+1 before the exit is signalled.
    """
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
    prices = _prices([
        {"trade_date": dates[0], "open": 10.0, "close": 10.0},
        {"trade_date": dates[1], "open": 10.0, "close": 10.0},
        {"trade_date": dates[2], "open": 10.0, "close": 10.0},
        # -10% sealed limit down, and this is the session the exit fills into.
        {"trade_date": dates[3], "open": 9.0, "close": 9.0, "high": 9.0, "low": 9.0},
        {"trade_date": dates[4], "open": 9.0, "close": 9.0},
    ])
    #                              d0    d1    d2    d3(exit)  d4
    result = _run(prices, _weights(dates, [0.10, 0.10, 0.0, 0.0, 0.0]))

    limit_day = pd.Timestamp(dates[3])
    limit_day_sells = result.trades[
        (result.trades["side"] == "sell") & (result.trades["trade_date"] == limit_day)
    ]
    assert limit_day_sells.empty, "no shares may be sold into a sealed limit-down"
    blocked = result.rejects[result.rejects["trade_date"] == limit_day]
    assert "limit_down_no_sell" in set(blocked["reason"])


# --------------------------------------------------------------------------
# 5. Lot rounding
# --------------------------------------------------------------------------
def test_buy_quantities_round_down_to_whole_lots():
    """0.10 NAV at 10.55 = 9,478.7 shares -> 9,400, never 9,478."""
    dates = ["2026-01-05", "2026-01-06"]
    prices = _prices([
        {"trade_date": dates[0], "open": 10.55, "close": 10.55},
        {"trade_date": dates[1], "open": 10.55, "close": 10.55},
    ])
    result = _run(prices, _weights(dates, [0.10, 0.10]))

    shares = int(result.trades.iloc[0]["shares"])
    assert shares % 100 == 0
    assert shares == 9_400


# --------------------------------------------------------------------------
# 6. Cash conservation — the identity that catches most accounting bugs
# --------------------------------------------------------------------------
def test_nav_equals_cash_plus_holdings_after_every_event():
    """NAV must never be produced by anything but cash + marked positions."""
    dates = pd.bdate_range("2026-01-05", periods=8).strftime("%Y-%m-%d").tolist()
    closes = [10.0, 10.4, 9.9, 10.2, 10.6, 10.1, 10.3, 10.5]
    prices = _prices([
        {"trade_date": d, "open": c, "close": c} for d, c in zip(dates, closes)
    ])
    weights = _weights(dates, [0.20, 0.20, 0.0, 0.15, 0.15, 0.0, 0.10, 0.10])
    result = _run(prices, weights)

    trades = result.trades
    cash = INITIAL_NAV
    for _, row in trades.iterrows():
        # Slippage is already inside `price`; adding it again would double-count.
        gross = row["shares"] * row["price"]
        fees = row["commission"] + row["transfer_fee"]
        if row["side"] == "buy":
            cash -= gross + fees
        else:
            stamp = row.get("stamp_duty", 0.0)
            cash += gross - fees - (0.0 if pd.isna(stamp) else stamp)

    last_date = pd.Timestamp(dates[-1])
    shares_held = (
        trades.assign(signed=lambda f: f.apply(
            lambda r: r["shares"] if r["side"] == "buy" else -r["shares"], axis=1
        ))["signed"].sum()
    )
    expected_nav = cash + shares_held * closes[-1]
    assert result.nav_curve.loc[last_date] == pytest.approx(expected_nav, abs=0.01)


# --------------------------------------------------------------------------
# 7. Determinism
# --------------------------------------------------------------------------
def test_identical_inputs_produce_byte_identical_results():
    dates = pd.bdate_range("2026-01-05", periods=10).strftime("%Y-%m-%d").tolist()
    closes = [10.0, 10.4, 9.9, 10.2, 10.6, 10.1, 10.3, 10.5, 10.2, 10.7]
    prices = _prices([
        {"trade_date": d, "open": c, "close": c} for d, c in zip(dates, closes)
    ])
    weights = _weights(dates, [0.2, 0.2, 0.0, 0.15, 0.15, 0.0, 0.1, 0.1, 0.0, 0.0])

    first = _run(prices.copy(), weights.copy())
    second = _run(prices.copy(), weights.copy())

    pd.testing.assert_series_equal(first.nav_curve, second.nav_curve)
    pd.testing.assert_frame_equal(first.trades, second.trades)


# --------------------------------------------------------------------------
# 8. Cost-model sensitivity — costs must actually reach the NAV
# --------------------------------------------------------------------------
def test_raising_costs_lowers_nav_by_exactly_the_extra_charge():
    """Doubling stamp duty costs exactly one more 5 bps charge on the sell."""
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
    prices = _prices([{"trade_date": d, "open": 10.0, "close": 10.0} for d in dates])
    weights = _weights(dates, [0.10, 0.10, 0.0, 0.0, 0.0])

    base = _run(prices, weights)
    doubled = _run(prices, weights, cost=CostModelConfig(sell_stamp_duty_bps=10.0))

    sell = base.trades[base.trades["side"] == "sell"].iloc[0]
    expected_extra = sell["shares"] * sell["price"] * 5.0 / 10_000

    delta = base.nav_curve.iloc[-1] - doubled.nav_curve.iloc[-1]
    assert delta == pytest.approx(expected_extra, rel=1e-6)
