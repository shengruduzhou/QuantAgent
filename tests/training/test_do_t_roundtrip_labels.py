from __future__ import annotations

import pandas as pd

from quantagent.training.do_t_roundtrip_labels import (
    ROUND_TRIP_LABEL_COLUMNS,
    RoundTripLabelConfig,
    build_round_trip_labels,
)


def _minute_panel(prices):
    rows = []
    for i, price in enumerate(prices):
        rows.append(
            {
                "symbol": "000001.SZ",
                "trade_date": "2026-06-01",
                "trade_time": pd.Timestamp("2026-06-01 09:30:00") + pd.Timedelta(minutes=i),
                "open": price,
                "high": price * 1.002,
                "low": price * 0.998,
                "close": price,
                "volume": 100_000,
                "pre_close": 10.0,
                "open_sell_price": 10.4,
            }
        )
    return pd.DataFrame(rows)


def test_round_trip_labels_emit_required_columns_and_sell_high_success():
    panel = _minute_panel([10.0, 10.4, 10.2, 9.8, 9.9, 10.1])
    labels = build_round_trip_labels(panel, config=RoundTripLabelConfig(horizon_minutes=4, min_required_edge_bps=5))

    for col in ROUND_TRIP_LABEL_COLUMNS:
        assert col in labels.columns
    row = labels.iloc[1]
    assert row["label_sell_high_success"] == 1
    assert row["label_sell_high_net_edge_bps"] > 0
    assert row["label_sell_high_eod_restore"] == 0
    assert row["label_time_to_buyback"] > 0


def test_round_trip_labels_mark_new_high_failure_and_eod_restore():
    panel = _minute_panel([10.0, 10.4, 10.6, 10.7, 10.8])
    labels = build_round_trip_labels(panel, config=RoundTripLabelConfig(horizon_minutes=3, min_required_edge_bps=5))

    row = labels.iloc[1]
    assert row["label_sell_high_success"] == 0
    assert row["label_sell_high_fail_new_high"] == 1
    assert row["label_sell_high_eod_restore"] == 1


def test_round_trip_labels_buy_low_success_and_buyback_now():
    panel = _minute_panel([10.4, 9.8, 10.0, 10.5, 10.3])
    labels = build_round_trip_labels(panel, config=RoundTripLabelConfig(horizon_minutes=3, min_required_edge_bps=5))

    low_row = labels.iloc[1]
    assert low_row["label_buy_low_success"] == 1
    assert low_row["label_sell_after_buy_success"] == 1
    assert low_row["label_buy_low_net_edge_bps"] > 0
    assert low_row["label_buyback_now_edge_bps"] > 0
