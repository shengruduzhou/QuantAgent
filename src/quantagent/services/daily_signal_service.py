from __future__ import annotations

import pandas as pd


def infer_v4_alpha(features: pd.DataFrame, date: str | None = None) -> pd.DataFrame:
    data = features.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    if date is None:
        date_value = data["trade_date"].max()
    else:
        date_value = pd.Timestamp(date)
    latest = data[data["trade_date"] == date_value].copy()
    numeric = [c for c in ["ret_5d", "ma_gap_20d", "event_sentiment", "northbound_flow"] if c in latest.columns]
    if numeric:
        latest["alpha"] = latest[numeric].fillna(0.0).mean(axis=1)
    else:
        latest["alpha"] = 0.0
    latest["confidence"] = 1.0 / (1.0 + latest["alpha"].abs())
    return latest[["trade_date", "symbol", "alpha", "confidence"]].sort_values("symbol").reset_index(drop=True)
