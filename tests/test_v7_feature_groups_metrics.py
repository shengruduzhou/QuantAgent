"""Feature-group selection and extended metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.data.v7_feature_groups import join_pit_features, select_v7_feature_columns


def test_select_feature_columns_only_keeps_present_columns():
    frame = pd.DataFrame(
        {
            "momentum_5d": [0.1, 0.2],
            "valuation_history_zscore_120d": [-1.0, 0.5],
            "unused": [0.0, 0.0],
        }
    )
    selection = select_v7_feature_columns(frame, groups=("short_term", "valuation"))
    assert "momentum_5d" in selection.selected
    assert "valuation_history_zscore_120d" in selection.selected
    assert "unused" not in selection.selected
    assert selection.group_to_columns["short_term"] == ("momentum_5d",)


def test_join_pit_features_is_strict_asof():
    base = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "available_at": pd.to_datetime(["2026-05-01", "2026-05-03", "2026-05-05"]),
            "trade_date": pd.to_datetime(["2026-05-01", "2026-05-03", "2026-05-05"]),
        }
    )
    extra = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "available_at": pd.to_datetime(["2026-05-02", "2026-05-04"]),
            "fund_metric": [1.0, 2.0],
        }
    )
    out = join_pit_features(base, [extra])
    assert out.loc[out["trade_date"] == pd.Timestamp("2026-05-01"), "fund_metric"].isna().all()
    assert out.loc[out["trade_date"] == pd.Timestamp("2026-05-03"), "fund_metric"].iloc[0] == 1.0
    assert out.loc[out["trade_date"] == pd.Timestamp("2026-05-05"), "fund_metric"].iloc[0] == 2.0








