from __future__ import annotations

import pandas as pd
import pytest

from quantagent.quant_math.ashare import (
    ASharePriceLimit,
    AshareRuleEngine,
    board_price_limit_vector,
    daily_price_limit,
    limit_up_mask,
)
from quantagent.universe.filters import derive_market_flags


def test_rule_engine_preserves_main_board_st_reform_boundary() -> None:
    engine = AshareRuleEngine()

    before = engine.price_limit_rule(
        "600000.SH", 10.0, trade_date="2026-07-03", is_st=True
    )
    after = engine.price_limit_rule(
        "600000.SH", 10.0, trade_date="2026-07-06", is_st=True
    )

    assert before["ratio"] == pytest.approx(0.05)
    assert before["limit_up"] == pytest.approx(10.50)
    assert after["ratio"] == pytest.approx(0.10)
    assert after["limit_up"] == pytest.approx(11.00)


def test_rule_engine_preserves_underlying_board_for_risk_warning_names() -> None:
    engine = AshareRuleEngine()

    assert engine.price_limit_rule(
        "000001.SZ", 10.0, trade_date="2026-08-07", is_st=True
    )["ratio"] == pytest.approx(0.10)
    assert engine.price_limit_rule(
        "300001.SZ", 10.0, trade_date="2026-08-07", is_st=True
    )["ratio"] == pytest.approx(0.20)
    assert engine.price_limit_rule(
        "688001.SH", 10.0, trade_date="2026-08-07", is_st=True
    )["ratio"] == pytest.approx(0.20)


def test_daily_main_board_st_limit_requires_date_in_canonical_mode() -> None:
    with pytest.raises(ValueError, match="valid trade_date"):
        daily_price_limit("600000.SH", True)


def test_explicit_scenario_limits_remain_available_without_claiming_exchange_truth() -> None:
    scenario = ASharePriceLimit(st=0.07)
    assert daily_price_limit("600000.SH", True, scenario) == pytest.approx(0.07)
    assert daily_price_limit("300001.SZ", True, scenario) == pytest.approx(0.07)


def test_vector_uses_each_rows_trade_date_and_board() -> None:
    symbols = pd.Series(["600000.SH", "600000.SH", "300001.SZ"])
    is_st = pd.Series([True, True, True])
    dates = pd.to_datetime(["2026-07-03", "2026-07-06", "2026-07-06"])

    ratios = board_price_limit_vector(
        symbols,
        is_st,
        trade_dates=dates,
    )

    assert ratios.tolist() == pytest.approx([0.05, 0.10, 0.20])


def test_vector_fails_closed_if_historical_main_board_st_date_is_missing() -> None:
    with pytest.raises(ValueError, match="valid trade_date"):
        board_price_limit_vector(
            pd.Series(["600000.SH"]),
            pd.Series([True]),
        )


def test_limit_up_mask_changes_at_exact_reform_date() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH", "600000.SH"],
            "trade_date": pd.to_datetime(["2026-07-02", "2026-07-03", "2026-07-06"]),
            "close": [10.00, 10.50, 11.02],
            "is_st": [True, True, True],
            "volume": [1_000_000, 1_000_000, 1_000_000],
        }
    )

    mask = limit_up_mask(frame, tolerance=1e-4)

    assert bool(mask.iloc[1]) is True  # +5% was the legal ST band on Friday.
    assert bool(mask.iloc[2]) is False  # +~5% is not limit-up after 10% reform.


def test_universe_flag_derivation_passes_trade_date_to_canonical_rule() -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH", "600000.SH"],
            "trade_date": pd.to_datetime(["2026-07-02", "2026-07-03", "2026-07-06"]),
            "close": [10.00, 10.50, 11.02],
            "volume": [1_000_000, 1_000_000, 1_000_000],
            "amount": [10_000_000, 10_000_000, 10_000_000],
            "is_st": [True, True, True],
        }
    )

    flags = derive_market_flags(panel, tolerance=1e-4)

    assert bool(flags.loc[1, "is_limit_up"]) is True
    assert bool(flags.loc[2, "is_limit_up"]) is False
