from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventRecord:
    symbol: str
    event_time: pd.Timestamp | str
    event_type: str
    source: str
    title: str
    summary: str = ""
    sentiment_score: float = 0.0
    policy_exposure: float = 0.0
    confidence: float = 0.5
    decay_half_life: float = 5.0
    raw_payload: dict[str, Any] = field(default_factory=dict)


class EventStore:
    """Normalized structured event store with deterministic daily aggregation."""

    def __init__(self, records: list[EventRecord] | pd.DataFrame | None = None) -> None:
        self.frame = self.normalize(records) if records is not None else self.empty_frame()

    @staticmethod
    def empty_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "symbol",
                "event_time",
                "event_type",
                "source",
                "title",
                "summary",
                "sentiment_score",
                "policy_exposure",
                "confidence",
                "decay_half_life",
                "raw_payload",
            ]
        )

    @classmethod
    def normalize(cls, records: list[EventRecord] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        else:
            frame = pd.DataFrame([asdict(record) for record in records])
        if frame.empty:
            return cls.empty_frame()
        required = set(cls.empty_frame().columns)
        for column in required:
            if column not in frame.columns:
                frame[column] = None
        frame["event_time"] = pd.to_datetime(frame["event_time"])
        numeric = ["sentiment_score", "policy_exposure", "confidence", "decay_half_life"]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["confidence"] = frame["confidence"].fillna(0.5).clip(0.0, 1.0)
        frame["decay_half_life"] = frame["decay_half_life"].fillna(5.0).clip(lower=1e-6)
        return frame[list(cls.empty_frame().columns)].sort_values(["event_time", "symbol", "event_type"]).reset_index(drop=True)

    def add(self, records: list[EventRecord] | pd.DataFrame) -> None:
        self.frame = pd.concat([self.frame, self.normalize(records)], ignore_index=True).sort_values(
            ["event_time", "symbol", "event_type"]
        ).reset_index(drop=True)

    def aggregate_daily(
        self,
        panel: pd.DataFrame,
        event_cutoff: str = "15:00:00",
        event_type_map: dict[str, int] | None = None,
    ) -> pd.DataFrame:
        if self.frame.empty:
            base = panel[["trade_date", "symbol"]].copy()
            return _add_empty_event_features(base)
        cutoff = pd.to_datetime(event_cutoff).time()
        base = panel[["trade_date", "symbol"]].copy()
        base["trade_date"] = pd.to_datetime(base["trade_date"])
        rows: list[dict[str, Any]] = []
        mapping = event_type_map or {}
        for _, row in base.sort_values(["trade_date", "symbol"]).iterrows():
            asof_time = row["trade_date"].normalize() + pd.to_timedelta(
                cutoff.hour * 3600 + cutoff.minute * 60 + cutoff.second,
                unit="s",
            )
            events = self.frame[(self.frame["symbol"] == row["symbol"]) & (self.frame["event_time"] <= asof_time)]
            if events.empty:
                rows.append(_empty_event_row(row["trade_date"], row["symbol"]))
                continue
            age_days = (asof_time - events["event_time"]).dt.total_seconds() / 86400.0
            decay = np.power(0.5, age_days / events["decay_half_life"].to_numpy())
            weight = decay * events["confidence"].to_numpy()
            denom = max(float(weight.sum()), 1e-12)
            latest = events.iloc[-1]
            rows.append(
                {
                    "trade_date": row["trade_date"],
                    "symbol": row["symbol"],
                    "event_count": float(len(events)),
                    "event_type_id": float(mapping.get(str(latest["event_type"]), len(mapping) + 1)),
                    "event_sentiment": float(np.dot(weight, events["sentiment_score"].fillna(0.0)) / denom),
                    "event_policy_exposure": float(np.dot(weight, events["policy_exposure"].fillna(0.0)) / denom),
                    "event_confidence": float(events["confidence"].mean()),
                    "event_decay": float(decay.iloc[-1]),
                    "event_recency": float(age_days.iloc[-1]),
                }
            )
        return pd.DataFrame(rows).sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _empty_event_row(trade_date: pd.Timestamp, symbol: str) -> dict[str, Any]:
    return {
        "trade_date": trade_date,
        "symbol": symbol,
        "event_count": 0.0,
        "event_type_id": 0.0,
        "event_sentiment": 0.0,
        "event_policy_exposure": 0.0,
        "event_confidence": 0.0,
        "event_decay": 0.0,
        "event_recency": np.inf,
    }


def _add_empty_event_features(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [_empty_event_row(pd.Timestamp(row["trade_date"]), str(row["symbol"])) for _, row in frame.iterrows()]
    return pd.DataFrame(rows)
