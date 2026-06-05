from __future__ import annotations

import pandas as pd

from quantagent.portfolio.do_t_overlay import DoTOverlayConfig
from quantagent.training.do_t_labels import build_do_t_training_labels


def test_build_do_t_training_labels_from_legal_overlay():
    start = pd.Timestamp("2024-01-02 09:30")
    minute_panel = pd.DataFrame([
        {
            "trade_date": pd.Timestamp("2024-01-02"),
            "symbol": "000001.SZ",
            "datetime": start + pd.Timedelta(minutes=i),
            "close": price,
        }
        for i, price in enumerate([10.0, 10.8, 10.9, 10.2, 9.8, 9.9])
    ])
    inventory = pd.DataFrame([
        {"trade_date": pd.Timestamp("2024-01-02"), "symbol": "000001.SZ", "available_shares": 1000}
    ])

    labels = build_do_t_training_labels(
        minute_panel,
        inventory,
        config=DoTOverlayConfig(trade_fraction=0.5, min_edge_pct=0.05, min_minutes_between_legs=2),
    )

    assert len(labels) == 1
    assert labels.iloc[0]["do_t_label"] == 1
    assert labels.iloc[0]["do_t_net_pnl"] > 0
