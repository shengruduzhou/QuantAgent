from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import json
import pandas as pd

from quantagent.replay.scenario_registry import ReplayScenario


@dataclass(frozen=True)
class ReplayDayResult:
    trade_date: str
    target_weight_count: int
    order_count: int
    rejected_count: int
    account_value: float


@dataclass(frozen=True)
class HistoricalLiveReplayResult:
    scenario: ReplayScenario
    days: tuple[ReplayDayResult, ...]
    data_quality_warnings: tuple[str, ...] = field(default_factory=tuple)

    def write_report(self, output_dir: str | Path) -> Path:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"{self.scenario.name}_replay_report.json"
        out.write_text(json.dumps(asdict(self), indent=2, sort_keys=True, default=str), encoding="utf-8")
        return out


class HistoricalLiveReplay:
    def __init__(self, service: Any) -> None:
        self.service = service

    def run(self, scenario: ReplayScenario, config: dict[str, Any] | None = None) -> HistoricalLiveReplayResult:
        cfg = config or {}
        features = self.service.build_features_v6(cfg, scenario.start_date, scenario.end_date, scenario.universe)
        dates = pd.to_datetime(features.frame["trade_date"]).drop_duplicates().sort_values().tail(5)
        day_results: list[ReplayDayResult] = []
        for date in dates:
            trade_date = str(pd.Timestamp(date).date())
            portfolio = self.service.build_portfolio_v6(cfg, trade_date, feature_frame=features.frame)
            paper = self.service.run_paper_trade_v6(cfg, trade_date, target_weights=portfolio["target_weights"], feature_frame=features.frame)
            day_results.append(
                ReplayDayResult(
                    trade_date=trade_date,
                    target_weight_count=int(len(portfolio["target_weights"])),
                    order_count=int(len(paper["order_states"])),
                    rejected_count=int(sum(str(getattr(getattr(state, "status", ""), "value", getattr(state, "status", ""))).lower() == "rejected" for state in paper["order_states"])),
                    account_value=float(paper["account_value"]),
                )
            )
        warnings = tuple(features.data_source_metadata.get("warnings", ()))
        provider = str(features.data_source_metadata.get("provider", "unknown"))
        if scenario.data_mode != "mock" and provider == "mock":
            warnings = warnings + ("real_external_scenario_ran_with_mock_provider",)
        return HistoricalLiveReplayResult(scenario, tuple(day_results), warnings)
