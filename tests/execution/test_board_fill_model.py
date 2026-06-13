"""Tests for 竞价/打板 fill models — limit bands, 一字板, break-only fills."""

from __future__ import annotations

import pandas as pd
import pytest

from quantagent.execution.board_fill_model import (
    auction_fill,
    board_chase_fill,
    detect_board_day,
    limit_up_price,
)


class TestLimitPrice:
    def test_main_board_10pct(self):
        assert limit_up_price(10.0, "600000") == 11.0
        assert limit_up_price(10.0, "000001") == 11.0

    def test_st_5pct(self):
        assert limit_up_price(10.0, "600000", is_st=True) == 10.5

    def test_chinext_star_20pct(self):
        assert limit_up_price(10.0, "300750") == 12.0
        assert limit_up_price(10.0, "688981") == 12.0

    def test_bse_30pct(self):
        assert limit_up_price(10.0, "830799") == 13.0


class TestAuctionFill:
    def test_normal_open_fills_at_open(self):
        r = auction_fill("buy", 1000, open_price=10.2, prev_close=10.0,
                         symbol="600000", auction_volume=100000)
        assert r.filled_quantity == 1000 and r.fill_price == 10.2

    def test_one_word_board_unfilled(self):
        r = auction_fill("buy", 1000, open_price=11.0, prev_close=10.0, symbol="600000")
        assert r.filled_quantity == 0 and r.reject_reason == "open_at_limit_up"

    def test_participation_cap(self):
        r = auction_fill("buy", 10000, open_price=10.2, prev_close=10.0,
                         symbol="600000", auction_volume=5000, participation_cap=0.10)
        assert r.filled_quantity == 500  # 5000·10% → lot-rounded
        assert r.fill_ratio == pytest.approx(0.05)

    def test_limit_down_open_blocks_sell(self):
        r = auction_fill("sell", 1000, open_price=9.0, prev_close=10.0, symbol="600000")
        assert r.filled_quantity == 0 and r.reject_reason == "open_at_limit_down"


def _board_bars(seq: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"trade_time": f"2026-06-01 {t}", "open": c, "high": h, "low": l, "close": c,
         "volume": 1000.0}
        for t, h, l, c in seq
    ])


class TestBoardLifecycle:
    LIM = 11.0  # prev_close 10.0, main board

    def test_sealed_all_day_no_fill(self):
        bars = _board_bars([("09:31:00", 10.5, 10.2, 10.5),
                            ("09:40:00", 11.0, 10.5, 11.0),    # seal
                            ("10:00:00", 11.0, 11.0, 11.0),
                            ("14:55:00", 11.0, 11.0, 11.0)])
        st = detect_board_day(bars, prev_close=10.0, symbol="600000")
        assert st.first_seal_time == "09:40:00" and not st.broke_after_seal
        assert st.closed_sealed
        fill = board_chase_fill(st)
        assert not fill.filled and fill.reason == "unfilled_sealed"

    def test_break_fills_at_limit(self):
        bars = _board_bars([("09:31:00", 10.5, 10.2, 10.5),
                            ("09:40:00", 11.0, 10.5, 11.0),    # seal
                            ("10:30:00", 11.0, 10.8, 10.9),    # break (开板)
                            ("11:00:00", 11.0, 10.9, 11.0),    # re-seal
                            ("14:55:00", 11.0, 11.0, 11.0)])
        st = detect_board_day(bars, prev_close=10.0, symbol="600000")
        assert st.broke_after_seal and st.n_breaks == 1
        fill = board_chase_fill(st)
        assert fill.filled and fill.fill_price == 11.0
        assert fill.fill_time == "10:30:00"
        assert fill.closed_sealed  # re-sealed into close: filled AND strong board

    def test_failed_board_filled_weak_close(self):
        bars = _board_bars([("09:40:00", 11.0, 10.5, 11.0),    # seal
                            ("10:30:00", 11.0, 10.5, 10.6),    # break
                            ("14:55:00", 10.7, 10.4, 10.5)])   # fades, no re-seal
        st = detect_board_day(bars, prev_close=10.0, symbol="600000")
        fill = board_chase_fill(st)
        assert fill.filled and not fill.closed_sealed  # adverse selection case

    def test_never_touched(self):
        bars = _board_bars([("09:31:00", 10.5, 10.2, 10.4),
                            ("14:55:00", 10.6, 10.3, 10.5)])
        st = detect_board_day(bars, prev_close=10.0, symbol="600000")
        assert not st.touched
        assert not board_chase_fill(st).filled

    def test_order_after_break_misses_it(self):
        bars = _board_bars([("09:40:00", 11.0, 10.5, 11.0),    # seal
                            ("10:30:00", 11.0, 10.8, 10.9),    # break
                            ("11:00:00", 11.0, 10.9, 11.0),    # re-seal — order joins here
                            ("14:55:00", 11.0, 11.0, 11.0)])
        st = detect_board_day(bars, prev_close=10.0, symbol="600000")
        fill = board_chase_fill(st, order_time="11:05:00")
        assert not fill.filled and fill.reason == "unfilled_sealed"
