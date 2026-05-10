import pandas as pd

from quantagent.quant_math.ashare import AshareRuleEngine, AshareRuleEngineConfig


def test_v4_board_inference_and_price_limits_are_configurable():
    engine = AshareRuleEngine(AshareRuleEngineConfig(st_price_limit_ratio=0.04, star_minimum_buy_quantity=200))
    assert engine.infer_board("600519.SH") == "main_board"
    assert engine.infer_board("300750.SZ") == "chinext"
    assert engine.infer_board("688981.SH") == "star"
    assert engine.infer_board("920001.BJ") == "bse"
    assert engine.infer_board("510300.SH") == "etf"
    assert engine.infer_board("123001.SZ") == "convertible_bond"
    assert engine.price_limit_rule("600519.SH", 10.0, is_st=True)["ratio"] == 0.04


def test_v4_lot_rules_cover_star_etf_odd_lot_and_t_plus_one():
    engine = AshareRuleEngine()
    assert engine.round_order_quantity("688981.SH", "buy", 199) == 0
    assert engine.round_order_quantity("688981.SH", "buy", 250) == 200
    assert engine.round_order_quantity("510300.SH", "buy", 99) == 0
    assert engine.round_order_quantity("600519.SH", "sell", 37) == 37
    valid, reason = engine.validate_order_intent(
        {"symbol": "600519.SH", "side": "sell", "quantity": 200},
        {"symbol": "600519.SH", "available_shares": 100, "volume": 1000},
    )
    assert not valid
    assert reason == "t_plus_one_insufficient_available_shares"


def test_v4_tradability_blocks_suspension_limit_up_and_limit_down():
    engine = AshareRuleEngine()
    assert not engine.is_tradable({"symbol": "600519.SH", "is_suspended": True})
    assert engine.validate_order_intent(
        {"symbol": "600519.SH", "side": "buy", "quantity": 100},
        {"symbol": "600519.SH", "is_limit_up": True, "volume": 1000},
    )[1] == "limit_up_no_buy"
    assert engine.validate_order_intent(
        {"symbol": "600519.SH", "side": "sell", "quantity": 100},
        {"symbol": "600519.SH", "is_limit_down": True, "available_shares": 100, "volume": 1000},
    )[1] == "limit_down_no_sell"
    panel = pd.DataFrame(
        [
            {"trade_date": "2026-01-01", "symbol": "A", "volume": 100},
            {"trade_date": "2026-01-01", "symbol": "B", "volume": 0},
        ]
    )
    assert engine.filter_tradable(panel)["symbol"].tolist() == ["A"]
