import pandas as pd

from quantagent.agents.flow_agent import multi_source_flow_signals
from quantagent.ashare.fund_flow import build_flow_feature_frame


def test_flow_feature_frame_combines_sources():
    dates = pd.date_range("2026-01-01", periods=30)
    northbound = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": "600519.SH",
            "holding_value": [1e9 + i * 1e6 for i in range(29)] + [1.2e9],
        }
    )
    dragon = pd.DataFrame(
        {
            "trade_date": [dates[-1]],
            "symbol": ["600519.SH"],
            "inst_buy": [1e8],
            "inst_sell": [1e7],
            "retail_buy": [2e7],
            "retail_sell": [5e7],
        }
    )
    features = build_flow_feature_frame({"northbound_holding": northbound, "dragon_tiger": dragon})
    assert "flow_zscore" in features.frame.columns
    assert set(features.source_columns) == {"dragon_tiger", "northbound_holding"}


def test_multi_source_flow_agent_emits_structured_signals_only():
    dates = pd.date_range("2026-01-01", periods=30)
    frame = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": "600519.SH",
            "holding_value": [1e9 + i * 1e6 for i in range(29)] + [1.5e9],
        }
    )
    signals = multi_source_flow_signals({"northbound_holding": frame}, z_threshold=1.0)
    assert signals
    assert signals[0].agent_name.endswith("_flow_agent")

