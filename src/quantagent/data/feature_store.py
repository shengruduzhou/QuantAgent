from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from quantagent.data.event_store import EventStore
from quantagent.data.features import add_benchmark_features, add_technical_features
from quantagent.data.labels import add_forward_return_labels
from quantagent.data.point_in_time import PITJoiner
from quantagent.data.universe import UniverseBuilder


@dataclass(frozen=True)
class FeatureStoreConfig:
    cache_dir: str | Path = "data/cache"
    feature_version: str = "v4.0"
    event_cutoff: str = "15:00:00"
    enable_alpha101: bool = True
    enable_cicc_high_freq: bool = True
    enable_sector_rotation: bool = True
    enable_fund_flow: bool = True
    enable_event_policy: bool = True


@dataclass(frozen=True)
class FeatureStoreResult:
    frame: pd.DataFrame
    feature_version: str
    data_source_metadata: dict[str, Any] = field(default_factory=dict)


class FeatureStore:
    """Unified point-in-time feature builder for V4 offline and live views."""

    def __init__(
        self,
        config: FeatureStoreConfig | None = None,
        pit_joiner: PITJoiner | None = None,
        universe_builder: UniverseBuilder | None = None,
    ) -> None:
        self.config = config or FeatureStoreConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.pit_joiner = pit_joiner or PITJoiner()
        self.universe_builder = universe_builder or UniverseBuilder()

    def build_view(
        self,
        prices: pd.DataFrame,
        benchmark: pd.DataFrame | None = None,
        benchmark_symbol: str = "000300.SH",
        fundamentals: pd.DataFrame | None = None,
        events: EventStore | pd.DataFrame | None = None,
        fund_flow: pd.DataFrame | None = None,
        universe: pd.DataFrame | None = None,
        include_labels: bool = False,
    ) -> FeatureStoreResult:
        frame = add_technical_features(prices)
        if benchmark is not None and not benchmark.empty:
            frame = add_benchmark_features(frame, benchmark, benchmark_symbol)
        else:
            frame["benchmark_symbol"] = benchmark_symbol
            frame["benchmark_ret_1d"] = 0.0
        if fundamentals is not None and not fundamentals.empty:
            frame = self.pit_joiner.join_fundamentals(frame, fundamentals)
        if self.config.enable_fund_flow and fund_flow is not None and not fund_flow.empty:
            frame = self._join_fund_flow(frame, fund_flow)
        if self.config.enable_event_policy and events is not None:
            event_store = events if isinstance(events, EventStore) else EventStore(events)
            event_features = event_store.aggregate_daily(frame, event_cutoff=self.config.event_cutoff)
            frame = frame.merge(event_features, on=["trade_date", "symbol"], how="left")
        if self.config.enable_alpha101:
            frame = self._merge_long_factor_frame(frame, self._compute_alpha101(frame))
        if self.config.enable_cicc_high_freq:
            frame = self._merge_long_factor_frame(frame, self._compute_cicc(frame))
        if self.config.enable_sector_rotation and "sector" in frame.columns:
            frame = self._merge_sector_rotation(frame)
        if universe is not None and not universe.empty:
            frame = self.universe_builder.filter(frame, snapshots=universe)
        if include_labels:
            frame = add_forward_return_labels(frame)
        frame = self._finalize(frame)
        metadata = {
            "feature_version": self.config.feature_version,
            "point_in_time": True,
            "event_cutoff": self.config.event_cutoff,
        }
        return FeatureStoreResult(frame=frame, feature_version=self.config.feature_version, data_source_metadata=metadata)

    def build_training_view(self, *args: Any, **kwargs: Any) -> FeatureStoreResult:
        kwargs["include_labels"] = True
        return self.build_view(*args, **kwargs)

    def build_live_view(self, *args: Any, **kwargs: Any) -> FeatureStoreResult:
        kwargs["include_labels"] = False
        return self.build_view(*args, **kwargs)

    def write_cache(self, name: str, frame: pd.DataFrame) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = self.cache_dir / f"{name}.parquet"
        try:
            frame.to_parquet(parquet_path, index=False)
            return parquet_path
        except Exception:
            csv_path = self.cache_dir / f"{name}.csv"
            frame.to_csv(csv_path, index=False)
            return csv_path

    def read_cache(self, name: str) -> pd.DataFrame:
        parquet_path = self.cache_dir / f"{name}.parquet"
        csv_path = self.cache_dir / f"{name}.csv"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(f"No cached feature frame named {name}")

    def _compute_alpha101(self, frame: pd.DataFrame) -> pd.DataFrame:
        try:
            from quantagent.factors.alpha101 import compute_alpha101
        except Exception:
            return pd.DataFrame()
        return compute_alpha101(frame)

    def _compute_cicc(self, frame: pd.DataFrame) -> pd.DataFrame:
        try:
            from quantagent.factors.cicc_high_freq import compute_cicc_high_freq_factors
        except Exception:
            return pd.DataFrame()
        return compute_cicc_high_freq_factors(frame).factors

    def _merge_sector_rotation(self, frame: pd.DataFrame) -> pd.DataFrame:
        try:
            from quantagent.factors.sector_rotation import compute_sector_rotation_factors
        except Exception:
            return frame
        sector = compute_sector_rotation_factors(frame)
        return frame.merge(sector, on=["trade_date", "sector"], how="left")

    def _merge_long_factor_frame(self, frame: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        if factors.empty:
            return frame
        wide = factors.pivot_table(
            index=["trade_date", "symbol"],
            columns="factor_name",
            values="factor_value",
            aggfunc="last",
        ).reset_index()
        wide.columns = [str(column) for column in wide.columns]
        return frame.merge(wide, on=["trade_date", "symbol"], how="left")

    def _join_fund_flow(self, frame: pd.DataFrame, fund_flow: pd.DataFrame) -> pd.DataFrame:
        data = fund_flow.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        flow_columns = [c for c in data.columns if c not in {"trade_date", "symbol"}]
        return frame.merge(data[["trade_date", "symbol", *flow_columns]], on=["trade_date", "symbol"], how="left")

    def _finalize(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        out["feature_version"] = self.config.feature_version
        out["asof_time"] = self.pit_joiner.panel_timestamp(out["trade_date"])
        return out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
