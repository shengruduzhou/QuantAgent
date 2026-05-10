from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AblationResult:
    name: str
    metrics: dict[str, float]


def summarize_ablations(results: list[AblationResult]) -> pd.DataFrame:
    rows = [{"name": result.name, **result.metrics} for result in results]
    return pd.DataFrame(rows).sort_values("name").reset_index(drop=True)
