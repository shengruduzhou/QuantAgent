"""Build supervised Do-T labels from legal intraday overlay simulations."""

from __future__ import annotations

import pandas as pd

from quantagent.portfolio.do_t_overlay import DoTOverlayConfig, simulate_do_t_overlay
from quantagent.training.do_t_roundtrip_labels import (
    ROUND_TRIP_LABEL_COLUMNS,
    RoundTripLabelConfig,
    build_round_trip_labels,
)


def build_do_t_training_labels(
    minute_panel: pd.DataFrame,
    available_inventory: pd.DataFrame,
    *,
    config: DoTOverlayConfig | None = None,
    min_net_pnl: float = 0.0,
) -> pd.DataFrame:
    """Create one label row per (trade_date, symbol) with legal Do-T outcome."""
    trades = simulate_do_t_overlay(minute_panel, available_inventory, config=config)
    if trades.empty:
        return pd.DataFrame(columns=[
            "trade_date", "symbol", "do_t_label", "do_t_mode",
            "do_t_edge_pct", "do_t_net_pnl", "do_t_cost",
        ])
    out = trades[[
        "trade_date", "symbol", "mode", "edge_pct", "net_pnl", "cost",
    ]].copy()
    out = out.rename(columns={
        "mode": "do_t_mode",
        "edge_pct": "do_t_edge_pct",
        "net_pnl": "do_t_net_pnl",
        "cost": "do_t_cost",
    })
    out["do_t_label"] = (pd.to_numeric(out["do_t_net_pnl"], errors="coerce") > float(min_net_pnl)).astype(int)
    return out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


__all__ = [
    "ROUND_TRIP_LABEL_COLUMNS",
    "RoundTripLabelConfig",
    "build_do_t_training_labels",
    "build_round_trip_labels",
]
