import pandas as pd

from quantagent.data.feature_store import FeatureStore, FeatureStoreConfig


def test_v6_pit_feature_store_respects_event_cutoff():
    prices = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "symbol": ["600000.SH"] * 3,
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.8, 9.9, 10.0],
            "close": [10.0, 10.1, 10.2],
            "volume": [1000, 1000, 1000],
            "amount": [10000, 10100, 10200],
        }
    )
    fundamentals = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "announcement_time": [pd.Timestamp("2026-01-05 16:00:00")],
            "report_period": ["2025Q4"],
            "roe": [0.15],
        }
    )
    store = FeatureStore(FeatureStoreConfig(feature_version="v6.test", event_cutoff="15:00:00", enable_alpha101=False, enable_cicc_high_freq=False, enable_sector_rotation=False))
    result = store.build_live_view(prices, fundamentals=fundamentals)
    jan5 = result.frame[result.frame["trade_date"] == pd.Timestamp("2026-01-05")].iloc[0]
    jan6 = result.frame[result.frame["trade_date"] == pd.Timestamp("2026-01-06")].iloc[0]
    assert pd.isna(jan5["roe"])
    assert jan6["roe"] == 0.15

