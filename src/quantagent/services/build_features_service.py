from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantagent.data.event_store import EventRecord, EventStore
from quantagent.data.feature_store import FeatureStore, FeatureStoreConfig, FeatureStoreResult


@dataclass(frozen=True)
class SyntheticV4Inputs:
    prices: pd.DataFrame
    benchmark: pd.DataFrame
    fundamentals: pd.DataFrame
    events: EventStore
    fund_flow: pd.DataFrame
    universe: pd.DataFrame


def build_synthetic_v4_inputs(symbol_count: int = 8, periods: int = 60, seed: int = 7) -> SyntheticV4Inputs:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-02", periods=periods, freq="B")
    symbols = [f"600{500 + i:03d}.SH" for i in range(symbol_count - 3)] + ["300750.SZ", "688981.SH", "920001.BJ"]
    sectors = ["consumer", "tech", "semi", "finance"]
    rows: list[dict[str, object]] = []
    fund_rows: list[dict[str, object]] = []
    fundamental_rows: list[dict[str, object]] = []
    universe_rows: list[dict[str, object]] = []
    for j, symbol in enumerate(symbols):
        close = 20 + np.cumsum(rng.normal(0.02 + j * 0.001, 0.18, len(dates)))
        volume = np.maximum(1000, 800_000 + rng.normal(0, 20_000, len(dates)).cumsum())
        sector = sectors[j % len(sectors)]
        for i, date in enumerate(dates):
            suspended = i == 10 and j == 0
            is_limit_up = i == 20 and j == 1
            is_limit_down = i == 21 and j == 2
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": close[i] * 0.995,
                    "high": close[i] * 1.02,
                    "low": close[i] * 0.98,
                    "close": close[i],
                    "volume": 0.0 if suspended else volume[i],
                    "amount": close[i] * (0.0 if suspended else volume[i]),
                    "is_suspended": suspended,
                    "is_limit_up": is_limit_up,
                    "is_limit_down": is_limit_down,
                    "is_st": j == symbol_count - 1,
                    "listed_days": 300 + i,
                    "sector": sector,
                }
            )
            fund_rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "northbound_flow": rng.normal(0, 1e6),
                    "main_money_flow": rng.normal(0, 5e6),
                }
            )
            universe_rows.append({"trade_date": date, "symbol": symbol, "is_member": True})
        for q, ann_idx in enumerate([5, 30]):
            fundamental_rows.append(
                {
                    "symbol": symbol,
                    "announcement_time": dates[ann_idx] + pd.Timedelta(hours=16),
                    "report_period": f"2025Q{q + 3}",
                    "roe": 0.08 + 0.01 * j + q * 0.005,
                    "debt_to_asset": 0.4 + 0.01 * j,
                }
            )
    benchmark_close = 4000 + np.cumsum(rng.normal(0.5, 8.0, len(dates)))
    benchmark = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": "000300.SH",
            "open": benchmark_close * 0.998,
            "high": benchmark_close * 1.005,
            "low": benchmark_close * 0.995,
            "close": benchmark_close,
            "volume": 1_000_000,
        }
    )
    events = EventStore(
        [
            EventRecord(symbol=symbols[0], event_time=dates[12] + pd.Timedelta(hours=10), event_type="policy", source="synthetic", title="policy support", sentiment_score=0.6, policy_exposure=0.8, confidence=0.7),
            EventRecord(symbol=symbols[1], event_time=dates[25] + pd.Timedelta(hours=16), event_type="risk", source="synthetic", title="post market risk", sentiment_score=-0.7, policy_exposure=0.1, confidence=0.8),
        ]
    )
    return SyntheticV4Inputs(
        prices=pd.DataFrame(rows),
        benchmark=benchmark,
        fundamentals=pd.DataFrame(fundamental_rows),
        events=events,
        fund_flow=pd.DataFrame(fund_rows),
        universe=pd.DataFrame(universe_rows),
    )


def build_features_v4(inputs: SyntheticV4Inputs | None = None, config: FeatureStoreConfig | None = None) -> FeatureStoreResult:
    data = inputs or build_synthetic_v4_inputs()
    store = FeatureStore(config or FeatureStoreConfig())
    return store.build_training_view(
        data.prices,
        benchmark=data.benchmark,
        fundamentals=data.fundamentals,
        events=data.events,
        fund_flow=data.fund_flow,
        universe=data.universe,
    )
