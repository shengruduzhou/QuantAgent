"""Tests for quantagent.universe.filters.

Validates the user's "至少 90% ST 不能买, 停牌硬排除" policy:

* Suspended → hard exclude when ``suspended_block_new=True``.
* Limit-up → hard exclude when ``limit_up_block_new=True``.
* ST → at least 90% blocked per date (configurable via
  ``st_min_block_rate``), with the top fraction (1 - 90% = 10%)
  ranked by prediction passing through.
* Derived market flags from OHLCV match hand-computed expectations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantagent.universe.filters import (
    UniverseFilterConfig,
    apply_universe_filter,
    compute_high_chase_flags,
    derive_market_flags,
)


def _make_market_panel() -> pd.DataFrame:
    """3 symbols × 5 days panel. Symbol A trades normally, B is
    suspended day 3 (volume=0), C hits limit-up day 4."""
    dates = pd.bdate_range("2024-01-02", periods=5)
    rows = []
    # Symbol A: normal trader
    a_close = [10.0, 10.2, 10.5, 10.6, 10.7]
    for i, d in enumerate(dates):
        rows.append({"trade_date": d, "symbol": "A.SZ", "close": a_close[i], "volume": 1_000_000, "amount": 1e7})
    # Symbol B: normal except day 3 suspended
    b_close = [20.0, 20.5, 20.5, 20.5, 20.7]
    b_volume = [500_000, 500_000, 0, 500_000, 500_000]
    b_amount = [1e7, 1e7, 0, 1e7, 1e7]
    for i, d in enumerate(dates):
        rows.append({"trade_date": d, "symbol": "B.SZ", "close": b_close[i], "volume": b_volume[i], "amount": b_amount[i]})
    # Symbol C: hits +10% limit-up on day 4 (10.0 → 11.0 = +10.0%; threshold 9.9%)
    c_close = [10.0, 10.0, 10.0, 11.0, 11.0]
    for i, d in enumerate(dates):
        rows.append({"trade_date": d, "symbol": "C.SZ", "close": c_close[i], "volume": 800_000, "amount": 8e6})
    # Pad with 100+ extra symbols so derive_market_flags treats these as active days
    for sid in range(200):
        sym = f"X{sid:03d}.SZ"
        for i, d in enumerate(dates):
            rows.append({"trade_date": d, "symbol": sym, "close": 5.0 + sid * 0.01, "volume": 100_000, "amount": 1e6})
    return pd.DataFrame(rows)


def test_derive_market_flags_detects_suspended_and_limit_up():
    mp = _make_market_panel()
    flags = derive_market_flags(mp)
    # B.SZ day 3 should be suspended
    b_day3 = flags[(flags["symbol"] == "B.SZ") & (flags["trade_date"] == "2024-01-04")]
    assert len(b_day3) == 1
    assert bool(b_day3.iloc[0]["is_suspended"]) is True
    # B.SZ other days are NOT suspended
    b_other = flags[(flags["symbol"] == "B.SZ") & (flags["trade_date"] != pd.Timestamp("2024-01-04"))]
    assert b_other["is_suspended"].sum() == 0
    # C.SZ day 4 is limit-up
    c_day4 = flags[(flags["symbol"] == "C.SZ") & (flags["trade_date"] == "2024-01-05")]
    assert bool(c_day4.iloc[0]["is_limit_up"]) is True
    # C.SZ other days are not
    c_other = flags[(flags["symbol"] == "C.SZ") & (flags["trade_date"] != pd.Timestamp("2024-01-05"))]
    assert c_other["is_limit_up"].sum() == 0


def test_derive_market_flags_is_board_aware():
    """derive_market_flags uses board-aware limits, not a flat 10%."""
    dates = pd.bdate_range("2024-01-02", periods=2)
    rows = []
    for sym, closes in (
        ("600000.SH", [10.0, 11.0]),   # main +10% → seal
        ("300001.SZ", [10.0, 11.0]),   # chinext +10% → NOT seal (limit 20%)
        ("300002.SZ", [10.0, 12.0]),   # chinext +20% → seal
        ("830001.BJ", [10.0, 13.0]),   # bse +30% → seal
    ):
        for d, c in zip(dates, closes):
            rows.append({"trade_date": d, "symbol": sym, "close": c, "volume": 8e5, "amount": 8e6})
    flags = derive_market_flags(pd.DataFrame(rows))

    def lu(sym: str) -> bool:
        r = flags[(flags["symbol"] == sym) & (flags["trade_date"] == dates[1])]
        return bool(r.iloc[0]["is_limit_up"])

    assert lu("600000.SH")        # main 10%
    assert not lu("300001.SZ")    # chinext +10% is not a seal
    assert lu("300002.SZ")        # chinext 20%
    assert lu("830001.BJ")        # bse 30%


def test_apply_filter_blocks_suspended_and_limit_up():
    mp = _make_market_panel()
    preds = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-04"]),
        "symbol": ["B.SZ", "C.SZ", "A.SZ"],
        "prediction": [0.5, 0.9, 0.1],
    })
    result = apply_universe_filter(preds, market_panel=mp)
    # B.SZ on 2024-01-04 is suspended → blocked
    b_row = result.filtered_predictions[
        (result.filtered_predictions["symbol"] == "B.SZ")
        & (result.filtered_predictions["trade_date"] == "2024-01-04")
    ].iloc[0]
    assert bool(b_row["universe_pass"]) is False
    assert b_row["universe_reason"] == "suspended"
    # C.SZ on 2024-01-05 is limit-up → blocked
    c_row = result.filtered_predictions[
        (result.filtered_predictions["symbol"] == "C.SZ")
        & (result.filtered_predictions["trade_date"] == "2024-01-05")
    ].iloc[0]
    assert bool(c_row["universe_pass"]) is False
    assert c_row["universe_reason"] == "limit_up_at_close"
    # A.SZ on 2024-01-04 is normal → passes
    a_row = result.filtered_predictions[
        (result.filtered_predictions["symbol"] == "A.SZ")
        & (result.filtered_predictions["trade_date"] == "2024-01-04")
    ].iloc[0]
    assert bool(a_row["universe_pass"]) is True


def test_st_soft_filter_keeps_at_least_top_10_percent():
    """Per user policy: at least 90% of ST stocks per day must be blocked."""
    # 20 ST candidates on a single day with descending prediction.
    dates = [pd.Timestamp("2024-01-04")] * 20
    symbols = [f"ST{i:02d}.SZ" for i in range(20)]
    preds = pd.DataFrame({
        "trade_date": dates,
        "symbol": symbols,
        "prediction": np.linspace(1.0, -1.0, num=20),
    })
    st_flags = pd.DataFrame({
        "trade_date": dates,
        "symbol": symbols,
        "is_st": [True] * 20,
    })
    cfg = UniverseFilterConfig(st_min_block_rate=0.90, suspended_block_new=False, limit_up_block_new=False)
    result = apply_universe_filter(preds, market_panel=None, st_flags=st_flags, config=cfg)
    passed = result.filtered_predictions[result.filtered_predictions["universe_pass"]]
    # 20 stocks × (1 - 0.90) = 2 should pass; floor → 2
    assert len(passed) == 2
    # The 2 survivors should be the highest-predicted ones
    assert set(passed["symbol"]) == {"ST00.SZ", "ST01.SZ"}


def test_st_soft_filter_respects_higher_block_rate():
    """st_min_block_rate=0.95 means at most 5% pass (1 of 20)."""
    dates = [pd.Timestamp("2024-01-04")] * 20
    symbols = [f"ST{i:02d}.SZ" for i in range(20)]
    preds = pd.DataFrame({
        "trade_date": dates,
        "symbol": symbols,
        "prediction": np.linspace(1.0, -1.0, num=20),
    })
    st_flags = pd.DataFrame({
        "trade_date": dates,
        "symbol": symbols,
        "is_st": [True] * 20,
    })
    cfg = UniverseFilterConfig(st_min_block_rate=0.95, suspended_block_new=False, limit_up_block_new=False)
    result = apply_universe_filter(preds, st_flags=st_flags, config=cfg)
    passed = result.filtered_predictions[result.filtered_predictions["universe_pass"]]
    assert len(passed) == 1
    assert passed.iloc[0]["symbol"] == "ST00.SZ"


def test_st_hard_exclude_when_block_rate_is_1():
    """st_min_block_rate=1.0 → no ST stock ever passes (hard exclude)."""
    dates = [pd.Timestamp("2024-01-04")] * 5
    preds = pd.DataFrame({
        "trade_date": dates,
        "symbol": [f"ST{i}.SZ" for i in range(5)],
        "prediction": [1.0, 0.8, 0.6, 0.4, 0.2],
    })
    st_flags = pd.DataFrame({
        "trade_date": dates,
        "symbol": [f"ST{i}.SZ" for i in range(5)],
        "is_st": [True] * 5,
    })
    cfg = UniverseFilterConfig(st_min_block_rate=1.0, suspended_block_new=False, limit_up_block_new=False)
    result = apply_universe_filter(preds, st_flags=st_flags, config=cfg)
    passed = result.filtered_predictions[result.filtered_predictions["universe_pass"]]
    assert len(passed) == 0


def test_no_st_data_passes_through_with_warning():
    preds = pd.DataFrame({
        "trade_date": [pd.Timestamp("2024-01-04")] * 3,
        "symbol": ["A.SZ", "B.SZ", "C.SZ"],
        "prediction": [0.5, 0.3, 0.1],
    })
    result = apply_universe_filter(preds, market_panel=None, st_flags=None)
    # No ST data → all pass through (the ST soft filter is skipped)
    assert result.filtered_predictions["universe_pass"].all()
    assert "warnings" in result.summary
    assert any("st_flags_missing" in w for w in result.summary["warnings"])


def test_empty_input_returns_empty_result():
    empty = pd.DataFrame(columns=["trade_date", "symbol", "prediction"])
    result = apply_universe_filter(empty)
    assert result.summary["status"] == "empty_input"


def test_high_chase_or_mode_blocks_grinding_rally():
    """Legacy OR mode: cum return alone is enough to block."""
    dates = pd.bdate_range("2024-01-02", periods=15)
    closes = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.2, 13.5, 13.8, 14.0, 14.0, 14.0, 14.0, 14.0]
    rows = [{"trade_date": d, "symbol": "RUN.SZ", "close": closes[i], "volume": 1e6, "amount": 1e7}
            for i, d in enumerate(dates)]
    for sid in range(200):
        for i, d in enumerate(dates):
            rows.append({"trade_date": d, "symbol": f"Q{sid:03d}.SZ", "close": 5.0, "volume": 1e5, "amount": 1e6})
    mp = pd.DataFrame(rows)
    flags = compute_high_chase_flags(mp, lookback=10, max_cum_return=0.30, max_limit_ups=3, combine="or")
    run_late = flags[flags["symbol"] == "RUN.SZ"].sort_values("trade_date").iloc[-3:]
    assert run_late["is_high_chase"].any()


def test_high_chase_and_mode_passes_grinding_rally():
    """AND mode: a grinding +35% rally with NO limit-ups should NOT be flagged."""
    dates = pd.bdate_range("2024-01-02", periods=15)
    # Smooth +35% climb: 10.0 → 13.5, no day > +9.9%
    closes = [10.0, 10.2, 10.4, 10.7, 11.0, 11.3, 11.6, 11.9, 12.2, 12.5, 12.8, 13.0, 13.2, 13.4, 13.5]
    rows = [{"trade_date": d, "symbol": "GRIND.SZ", "close": closes[i], "volume": 1e6, "amount": 1e7}
            for i, d in enumerate(dates)]
    for sid in range(200):
        for i, d in enumerate(dates):
            rows.append({"trade_date": d, "symbol": f"Q{sid:03d}.SZ", "close": 5.0, "volume": 1e5, "amount": 1e6})
    mp = pd.DataFrame(rows)
    flags = compute_high_chase_flags(mp, lookback=10, max_cum_return=0.30, max_limit_ups=2, combine="and")
    # No limit-ups in the smooth rally, so AND-mode must NOT flag any GRIND day
    grind_flags = flags[flags["symbol"] == "GRIND.SZ"]
    assert not grind_flags["is_high_chase"].any()


def test_high_chase_and_mode_blocks_parabolic_with_limit_ups():
    """AND mode: parabolic rally with consecutive limit-ups → flagged."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    # 3 limit-ups followed by gentle holds. cum return from close[-1] back 5 days = 1.331/1.0 - 1 = 33%
    closes = [1.0, 1.10, 1.21, 1.331, 1.331, 1.34, 1.35, 1.36, 1.37, 1.38]
    rows = [{"trade_date": d, "symbol": "PARA.SZ", "close": closes[i], "volume": 1e6, "amount": 1e7}
            for i, d in enumerate(dates)]
    for sid in range(200):
        for i, d in enumerate(dates):
            rows.append({"trade_date": d, "symbol": f"Q{sid:03d}.SZ", "close": 5.0, "volume": 1e5, "amount": 1e6})
    mp = pd.DataFrame(rows)
    flags = compute_high_chase_flags(mp, lookback=5, max_cum_return=0.30, max_limit_ups=2, combine="and")
    para_flags = flags[flags["symbol"] == "PARA.SZ"].sort_values("trade_date")
    # Day 4 (index 3): close = 1.331; close[3-1-1] = 1.10 (close at idx 1)
    #   actually lookback=5 means use close.shift(6) for window start
    # Just verify SOME day is flagged
    assert para_flags["is_high_chase"].any()


def test_high_chase_blocks_after_multiple_limit_ups():
    """3 limit-ups in lookback window should flag even with modest cum return."""
    dates = pd.bdate_range("2024-01-02", periods=12)
    # 4 +10% days then flat; cum return ~46% but limit-ups also trigger
    closes = [10.0, 11.0, 12.1, 13.31, 14.64, 14.64, 14.64, 14.64, 14.64, 14.64, 14.64, 14.64]
    rows = [{"trade_date": d, "symbol": "JUMP.SZ", "close": closes[i], "volume": 1e6, "amount": 1e7}
            for i, d in enumerate(dates)]
    for sid in range(200):
        for d in dates:
            rows.append({"trade_date": d, "symbol": f"Q{sid:03d}.SZ", "close": 5.0, "volume": 1e5, "amount": 1e6})
    mp = pd.DataFrame(rows)
    # Set lookback=10 so all 4 limit-ups fall inside the window from day 5 onwards.
    # Use combine="or" — this test isolates the limit-up-count rule by setting
    # cum_return threshold absurdly high so only limit-ups can fire.
    flags = compute_high_chase_flags(mp, lookback=10, max_cum_return=10.0, max_limit_ups=3, combine="or")
    # By day 5 (index 4) we should see >=3 limit-ups in the trailing window
    jump_late = flags[flags["symbol"] == "JUMP.SZ"].sort_values("trade_date").iloc[-3:]
    assert jump_late["is_high_chase"].any()


def test_apply_filter_blocks_high_chase_and_mode():
    """Default AND-mode filter blocks parabolic + multi-限ttup, not grinding rally."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    # 3 limit-ups → cum return ~33% in 5 days, multi限tt up
    closes = [1.0, 1.10, 1.21, 1.331, 1.331, 1.34, 1.35, 1.36, 1.37, 1.38]
    rows = [{"trade_date": d, "symbol": "PARA.SZ", "close": closes[i], "volume": 1e6, "amount": 1e7}
            for i, d in enumerate(dates)]
    for sid in range(200):
        for i, d in enumerate(dates):
            rows.append({"trade_date": d, "symbol": f"Q{sid:03d}.SZ", "close": 5.0, "volume": 1e5, "amount": 1e6})
    mp = pd.DataFrame(rows)
    # Eval at index 6 — that's the day where the trailing-5-day cum
    # return (close[5]/close[0]-1 = 0.34) AND limit_ups_n=3 both
    # exceed the AND-mode thresholds.
    preds = pd.DataFrame({
        "trade_date": [dates[6]] * 2,
        "symbol": ["PARA.SZ", "Q000.SZ"],
        "prediction": [0.9, 0.1],
    })
    cfg = UniverseFilterConfig(
        high_chase_enabled=True,
        high_chase_lookback=5,
        high_chase_max_cum_return=0.30,
        high_chase_max_limit_ups=2,
        high_chase_combine="and",
        suspended_block_new=False,
        limit_up_block_new=False,
    )
    result = apply_universe_filter(preds, market_panel=mp, config=cfg)
    para_row = result.filtered_predictions[result.filtered_predictions["symbol"] == "PARA.SZ"].iloc[0]
    assert bool(para_row["universe_pass"]) is False
    assert para_row["universe_reason"] == "high_chase_block"
    quiet_row = result.filtered_predictions[result.filtered_predictions["symbol"] == "Q000.SZ"].iloc[0]
    assert bool(quiet_row["universe_pass"]) is True


def test_apply_filter_handles_non_range_index_predictions():
    """Review fix #2: caller may pass a frame with a non-range index
    (e.g., post-filter sliced). The ST filter must not KeyError."""
    preds = pd.DataFrame({
        "trade_date": [pd.Timestamp("2024-01-04")] * 3,
        "symbol": ["ST_A.SZ", "ST_B.SZ", "ST_C.SZ"],
        "prediction": [0.9, 0.5, 0.1],
    }, index=[100, 200, 300])  # deliberately non-range
    st_flags = pd.DataFrame({
        "trade_date": [pd.Timestamp("2024-01-04")] * 3,
        "symbol": ["ST_A.SZ", "ST_B.SZ", "ST_C.SZ"],
        "is_st": [True, True, True],
    })
    # Should not raise
    cfg = UniverseFilterConfig(st_min_block_rate=0.90, suspended_block_new=False, limit_up_block_new=False)
    result = apply_universe_filter(preds, market_panel=None, st_flags=st_flags, config=cfg)
    # Top 10% of 3 = ceil(2.7) blocked = 3 - 1 = 0... actually n_block = ceil(0.9*3) = 3, n_pass = 0
    # All ST stocks blocked because we can't pass less than 1 stock through.
    # That's fine — confirms no crash.
    assert len(result.filtered_predictions) == 3


def test_apply_filter_st_flags_missing_is_st_column_warns_not_crashes():
    """Review fix #3: st_flags without is_st column should warn, not crash."""
    preds = pd.DataFrame({
        "trade_date": [pd.Timestamp("2024-01-04")] * 2,
        "symbol": ["A.SZ", "B.SZ"],
        "prediction": [0.5, 0.3],
    })
    bad_st_flags = pd.DataFrame({
        "trade_date": [pd.Timestamp("2024-01-04")] * 2,
        "symbol": ["A.SZ", "B.SZ"],
        # is_st column intentionally missing
    })
    result = apply_universe_filter(preds, market_panel=None, st_flags=bad_st_flags)
    assert result.filtered_predictions["universe_pass"].all()
    warnings = result.summary["warnings"]
    assert any("st_flags_missing_is_st_column" in w for w in warnings)


def test_apply_filter_dedups_duplicate_market_panel_rows():
    """Review fix #1: a market_panel with duplicate (date, symbol)
    rows must not multiply the prediction frame."""
    mp = _make_market_panel()
    # Inject a duplicate row
    dup = mp.iloc[0].copy()
    mp = pd.concat([mp, dup.to_frame().T], ignore_index=True)
    preds = pd.DataFrame({
        "trade_date": [pd.Timestamp("2024-01-08")],  # arbitrary mid-window date
        "symbol": ["A.SZ"],
        "prediction": [0.5],
    })
    cfg = UniverseFilterConfig(suspended_block_new=False, limit_up_block_new=False, high_chase_enabled=False)
    result = apply_universe_filter(preds, market_panel=mp, config=cfg)
    # Exactly one row in the output — no expansion from the dup
    assert len(result.filtered_predictions) == 1


def test_st_top_passes_only_when_not_also_suspended():
    """A stock flagged BOTH ST and suspended must be excluded by the
    suspended rule even if it would have ranked top of the ST cohort."""
    mp = _make_market_panel()
    # Override B.SZ to be both ST and suspended on 2024-01-04
    preds = pd.DataFrame({
        "trade_date": [pd.Timestamp("2024-01-04")] * 5,
        "symbol": ["B.SZ", "ST00.SZ", "ST01.SZ", "ST02.SZ", "ST03.SZ"],
        "prediction": [10.0, 1.0, 0.8, 0.6, 0.4],
    })
    st_flags = pd.DataFrame({
        "trade_date": [pd.Timestamp("2024-01-04")] * 5,
        "symbol": ["B.SZ", "ST00.SZ", "ST01.SZ", "ST02.SZ", "ST03.SZ"],
        "is_st": [True, True, True, True, True],
    })
    result = apply_universe_filter(preds, market_panel=mp, st_flags=st_flags)
    b_row = result.filtered_predictions[result.filtered_predictions["symbol"] == "B.SZ"].iloc[0]
    # B.SZ has the highest prediction in the ST cohort but is suspended
    assert bool(b_row["universe_pass"]) is False
    assert b_row["universe_reason"] == "suspended"
