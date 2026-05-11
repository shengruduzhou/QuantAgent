from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ReplayScenario:
    name: str
    start_date: str
    end_date: str
    universe: str
    regime_hint: str
    account_nav: float
    data_mode: str = "mock"
    frequency: str = "daily"
    cost_model: str = "ashare_default"


class ScenarioRegistry:
    def __init__(self, scenarios: list[ReplayScenario]) -> None:
        self.scenarios = {scenario.name: scenario for scenario in scenarios}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScenarioRegistry":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        scenarios = [ReplayScenario(**item) for item in data.get("scenarios", [])]
        return cls(scenarios)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ScenarioRegistry":
        return cls([ReplayScenario(**item) for item in config.get("scenarios", [])])

    def get(self, name: str) -> ReplayScenario:
        if name not in self.scenarios:
            raise KeyError(f"Unknown replay scenario: {name}")
        return self.scenarios[name]

    def names(self) -> list[str]:
        return sorted(self.scenarios)

