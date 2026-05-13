from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValuationScore:
    symbol: str
    valuation_score: float
    valuation_percentile: float
    margin_of_safety: float
    overextension_risk: float
    rationale: str


def score_valuation(frame: pd.DataFrame) -> list[ValuationScore]:
    if frame.empty:
        return []
    data = frame.copy()
    data["pe_ttm"] = data.get("pe_ttm", pd.Series(25.0, index=data.index)).astype(float)
    data["pb"] = data.get("pb", pd.Series(3.0, index=data.index)).astype(float)
    data["ps"] = data.get("ps", pd.Series(5.0, index=data.index)).astype(float)
    data["fcf_yield"] = data.get("fcf_yield", pd.Series(0.03, index=data.index)).astype(float)
    data["valuation_raw"] = -0.45 * data["pe_ttm"].rank(pct=True) - 0.25 * data["pb"].rank(pct=True) - 0.15 * data["ps"].rank(pct=True) + 0.15 * data["fcf_yield"].rank(pct=True)
    data["valuation_percentile"] = data["valuation_raw"].rank(pct=True) * 100.0
    scores: list[ValuationScore] = []
    for _, row in data.iterrows():
        percentile = float(row["valuation_percentile"])
        margin = float(row.get("margin_of_safety", (100.0 - percentile) / 100.0 - 0.5))
        score = float(np.clip(percentile, 0.0, 100.0))
        risk = float(np.clip(100.0 - score + max(0.0, -margin) * 100.0, 0.0, 100.0))
        scores.append(
            ValuationScore(
                symbol=str(row["symbol"]),
                valuation_score=score,
                valuation_percentile=percentile,
                margin_of_safety=margin,
                overextension_risk=risk,
                rationale=f"valuation_percentile={percentile:.1f}, margin_of_safety={margin:.2f}",
            )
        )
    return scores
