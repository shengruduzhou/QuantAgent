from __future__ import annotations

import pandas as pd
import pytest

from quantagent.data.feature_store import FeatureStore, FeatureStoreConfig
from quantagent.data.point_in_time import PITConfig, PITJoiner


def _prices() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-04-29", "2026-04-30", "2026-05-06"])
    return pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": ["600000.SH"] * len(dates),
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "volume": [1_000_000, 1_100_000, 1_200_000],
            "amount": [10_100_000.0, 11_220_000.0, 12_360_000.0],
        }
    )


def _config(*, cutoff: str = "15:00:00") -> FeatureStoreConfig:
    return FeatureStoreConfig(
        event_cutoff=cutoff,
        enable_alpha101=False,
        enable_cicc_high_freq=False,
        enable_sector_rotation=False,
        enable_event_policy=False,
    )


def test_feature_store_propagates_configured_decision_cutoff_to_fundamentals() -> None:
    fundamentals = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "announcement_time": [pd.Timestamp("2026-04-30 14:45:00")],
            "report_period": ["2026Q1"],
            "roe": [0.12],
        }
    )

    early = FeatureStore(config=_config(cutoff="14:30:00")).build_view(
        _prices(), fundamentals=fundamentals
    ).frame
    normal = FeatureStore(config=_config(cutoff="15:00:00")).build_view(
        _prices(), fundamentals=fundamentals
    ).frame

    early_t = early.loc[early["trade_date"] == pd.Timestamp("2026-04-30")].iloc[0]
    early_next = early.loc[early["trade_date"] == pd.Timestamp("2026-05-06")].iloc[0]
    normal_t = normal.loc[normal["trade_date"] == pd.Timestamp("2026-04-30")].iloc[0]

    assert pd.isna(early_t["roe"])
    assert early_next["roe"] == pytest.approx(0.12)
    assert normal_t["roe"] == pytest.approx(0.12)
    assert early_t["asof_time"] == pd.Timestamp("2026-04-30 14:30:00")


def test_fund_flow_is_joined_by_available_at_not_same_day_observation() -> None:
    flow = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-04-30")],
            "symbol": ["600000.SH"],
            # T-day observation is not released until after the T-day feature frontier.
            "available_at": [pd.Timestamp("2026-04-30 18:00:00")],
            "northbound_net_cny": [123_000_000.0],
        }
    )

    frame = FeatureStore(config=_config()).build_view(_prices(), fund_flow=flow).frame
    t_row = frame.loc[frame["trade_date"] == pd.Timestamp("2026-04-30")].iloc[0]
    next_row = frame.loc[frame["trade_date"] == pd.Timestamp("2026-05-06")].iloc[0]

    assert pd.isna(t_row["northbound_net_cny"])
    assert next_row["northbound_net_cny"] == pytest.approx(123_000_000.0)


def test_fund_flow_without_first_knowable_timestamp_fails_closed() -> None:
    flow = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-04-30")],
            "symbol": ["600000.SH"],
            "northbound_net_cny": [123_000_000.0],
        }
    )

    with pytest.raises(ValueError, match="availability column"):
        FeatureStore(config=_config()).build_view(_prices(), fund_flow=flow)


def test_generic_pit_join_rejects_invalid_availability_timestamps() -> None:
    panel = _prices()[["trade_date", "symbol"]]
    features = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "available_at": ["not-a-timestamp"],
            "value": [1.0],
        }
    )

    joiner = PITJoiner(PITConfig(event_cutoff="15:00:00"))
    with pytest.raises(ValueError, match="invalid/null available_at"):
        joiner.join_available_features(panel, features)
