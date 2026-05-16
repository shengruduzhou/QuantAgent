"""Feature-group selection and extended metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantagent.data.v7_feature_groups import join_pit_features, select_v7_feature_columns
from quantagent.training.metrics import (
    capacity_proxy,
    compose_alpha_metrics,
    max_drawdown,
    sortino_ratio,
    top_minus_bottom_spread,
)


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


def test_top_minus_bottom_spread_handles_ties():
    rng = np.random.default_rng(0)
    rows = []
    for day in range(8):
        for sidx in range(8):
            value = rng.standard_normal()
            rows.append({"trade_date": pd.Timestamp("2026-05-01") + pd.Timedelta(days=day), "symbol": f"S{sidx}", "alpha": value, "target": value + 0.5 * rng.standard_normal()})
    frame = pd.DataFrame(rows)
    summary = top_minus_bottom_spread(frame, "alpha", "target")
    assert set(summary) == {"top_minus_bottom_mean", "top_minus_bottom_std", "hit_rate"}


def test_sortino_and_max_drawdown_are_finite():
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])
    assert np.isfinite(sortino_ratio(returns))
    assert max_drawdown(returns) <= 0


def test_capacity_proxy_uses_amount_and_weights():
    frame = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-05-01")] * 3,
            "symbol": ["A", "B", "C"],
            "weight": [0.5, 0.25, 0.25],
            "amount": [1_000_000, 5_000_000, 8_000_000],
        }
    )
    cap = capacity_proxy(frame, weight_column="weight", amount_column="amount", participation_cap=0.10)
    assert cap == 0.10 * 1_000_000 / 0.5


def test_compose_alpha_metrics_returns_expected_keys():
    rng = np.random.default_rng(1)
    rows = []
    for day in range(10):
        for sidx in range(10):
            x = rng.standard_normal()
            rows.append({"trade_date": pd.Timestamp("2026-05-01") + pd.Timedelta(days=day), "symbol": f"S{sidx}", "alpha": x, "target": x + 0.3 * rng.standard_normal()})
    frame = pd.DataFrame(rows)
    summary = compose_alpha_metrics(frame, "alpha", "target", cost_bps=12.0)
    for key in (
        "rank_ic_mean",
        "icir",
        "top_minus_bottom_mean",
        "sharpe",
        "sortino",
        "max_drawdown",
        "net_return",
    ):
        assert key in summary
