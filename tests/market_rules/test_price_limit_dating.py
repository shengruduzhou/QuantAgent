"""Price-limit rules must be dated by trade_date (round-29 R1-F09).

ChiNext moved 10% -> 20% on 2020-08-24 (SZSE 创业板交易特别规定). Before the
main-board registration reform (2023-04-10) only the FIRST listing day had the
44%/-36% band; days 2-5 used the ordinary 10% band..
"""
import pandas as pd

from quantagent.market_rules import ashare as rules
from quantagent.quant_math.ashare import board_price_limit_vector


def test_chinext_before_2020_08_24_is_10pct():
    lim = rules.price_limits(board=rules.CHINEXT, previous_close=10.0, trade_date="2019-06-03")
    assert lim.ratio == 0.10, lim
    assert lim.limit_up == 11.00


def test_chinext_after_2020_08_24_is_20pct():
    lim = rules.price_limits(board=rules.CHINEXT, previous_close=10.0, trade_date="2020-08-24")
    assert lim.ratio == 0.20


def test_vectorised_panel_flag_path_is_dated():
    s = pd.Series(["300750.SZ", "300750.SZ"])
    d = pd.Series(pd.to_datetime(["2019-06-03", "2021-06-03"]))
    r = board_price_limit_vector(s, False, trade_dates=d)
    assert list(r) == [0.10, 0.20], list(r)


def test_pre_reform_main_board_ipo_day3_uses_ordinary_band():
    lim = rules.price_limits(board=rules.SH_MAIN, previous_close=10.0, trade_date="2019-06-05",
                             sessions_since_listing=2)
    assert lim.ratio == 0.10, lim
