from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from quantagent.data.providers.base import ProviderRequest, ProviderResult


@dataclass(frozen=True)
class V7ResearchDataBundle:
    policies: ProviderResult
    theme_metrics: ProviderResult
    base_universe: ProviderResult
    company_profiles: ProviderResult
    company_theme_map: ProviderResult
    fundamentals: ProviderResult
    news: ProviderResult
    market_state: ProviderResult
    market_panel: ProviderResult
    factors: ProviderResult
    positions: ProviderResult
    announcements: ProviderResult
    metadata: dict[str, Any] = field(default_factory=dict)


class LocalV7ResearchProvider:
    """Load V7 research inputs from local PIT files before falling back to synthetic data."""

    FILES = {
        "policies": "policies.csv",
        "theme_metrics": "theme_metrics.csv",
        "base_universe": "base_universe.csv",
        "company_profiles": "company_profiles.csv",
        "company_theme_map": "company_theme_map.csv",
        "fundamentals": "fundamentals.csv",
        "news": "news.csv",
        "market_state": "market_state.csv",
        "market_panel": "market_panel.csv",
        "factors": "factors.csv",
        "positions": "positions.csv",
        "announcements": "announcements.csv",
    }

    def __init__(self, root: str | Path = "data/v7") -> None:
        self.root = Path(root)

    def load_bundle(self, request: ProviderRequest, as_of_date: str) -> V7ResearchDataBundle:
        results = {name: self._read(name, request, as_of_date) for name in self.FILES}
        return V7ResearchDataBundle(
            policies=results["policies"],
            theme_metrics=results["theme_metrics"],
            base_universe=results["base_universe"],
            company_profiles=results["company_profiles"],
            company_theme_map=results["company_theme_map"],
            fundamentals=results["fundamentals"],
            news=results["news"],
            market_state=results["market_state"],
            market_panel=results["market_panel"],
            factors=results["factors"],
            positions=results["positions"],
            announcements=results["announcements"],
            metadata={"root": str(self.root), "as_of_date": as_of_date},
        )

    def _read(self, name: str, request: ProviderRequest, as_of_date: str) -> ProviderResult:
        path = self.root / self.FILES[name]
        if not path.exists():
            return ProviderResult(
                pd.DataFrame(),
                source=f"local_v7_missing:{path}",
                quality_score=0.0,
                warnings=(f"missing_v7_file:{path}",),
                metadata={"path": str(path)},
            )
        frame = pd.read_csv(path)
        frame = filter_point_in_time(frame, as_of_date)
        frame = filter_request_dates(frame, request)
        if request.symbols and "symbol" in frame.columns:
            frame = frame[frame["symbol"].astype(str).isin(request.symbols)]
        return ProviderResult(
            frame.reset_index(drop=True),
            source=f"local_v7_csv:{path}",
            point_in_time=True,
            quality_score=quality_score(frame),
            warnings=(),
            metadata={"path": str(path)},
        )


def filter_point_in_time(frame: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame.copy()
    cutoff = pd.Timestamp(as_of_date)
    if "available_at" in data.columns:
        data = data[pd.to_datetime(data["available_at"]) <= cutoff]
    elif "published_at" in data.columns:
        data = data[pd.to_datetime(data["published_at"]) <= cutoff]
    elif "announcement_time" in data.columns:
        data = data[pd.to_datetime(data["announcement_time"]) <= cutoff]
    return data


def filter_request_dates(frame: pd.DataFrame, request: ProviderRequest) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame.copy()
    date_column = "trade_date" if "trade_date" in data.columns else "published_at" if "published_at" in data.columns else None
    if date_column is None:
        return data
    dates = pd.to_datetime(data[date_column])
    start = pd.Timestamp(request.start_date)
    end = pd.Timestamp(request.end_date)
    return data[(dates >= start) & (dates <= end)]


def quality_score(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    required = {"available_at", "published_at"} & set(frame.columns)
    if required:
        return 1.0
    return 0.75
