import pandas as pd

from quantagent.data.event_store import EventRecord, EventStore
from quantagent.data.feature_store import FeatureStore
from quantagent.data.point_in_time import PITJoiner
from quantagent.services.build_features_service import build_synthetic_v4_inputs


def test_v4_feature_store_builds_sorted_point_in_time_view():
    inputs = build_synthetic_v4_inputs(symbol_count=6, periods=35)
    result = FeatureStore().build_training_view(
        inputs.prices,
        benchmark=inputs.benchmark,
        fundamentals=inputs.fundamentals,
        events=inputs.events,
        fund_flow=inputs.fund_flow,
        universe=inputs.universe,
    )
    frame = result.frame
    assert {"trade_date", "symbol", "feature_version", "asof_time", "future_1d_return"}.issubset(frame.columns)
    assert frame[["trade_date", "symbol"]].equals(frame[["trade_date", "symbol"]].sort_values(["trade_date", "symbol"]).reset_index(drop=True))
    assert "alpha001" in frame.columns


def test_v4_pit_joiner_prevents_future_fundamental_and_event_leakage():
    panel = pd.DataFrame({"trade_date": pd.to_datetime(["2026-01-02", "2026-01-03"]), "symbol": ["A", "A"], "close": [10, 11]})
    fundamentals = pd.DataFrame(
        {
            "symbol": ["A"],
            "announcement_time": [pd.Timestamp("2026-01-03 16:00:00")],
            "roe": [0.2],
        }
    )
    joined = PITJoiner().join_fundamentals(panel, fundamentals, value_columns=["roe"])
    assert pd.isna(joined.loc[joined["trade_date"] == pd.Timestamp("2026-01-03"), "roe"]).all()
    events = EventStore([EventRecord("A", "2026-01-03 16:00:00", "risk", "test", "late", sentiment_score=-1.0)])
    daily = events.aggregate_daily(panel, event_cutoff="15:00:00")
    assert daily.loc[daily["trade_date"] == pd.Timestamp("2026-01-03"), "event_count"].iloc[0] == 0.0
