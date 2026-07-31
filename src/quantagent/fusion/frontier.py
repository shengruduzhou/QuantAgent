"""Pareto frontier over the four declared research objectives.

The platform's stated selection criteria — maximum excess return, minimum
drawdown, maximum annualised return, strong out-of-sample robustness — conflict.
There is no single optimum, so the search reports the **non-dominated set** and
lets the operator's preference weights order that set. Preference weights change
the ranking only; they never change which candidates were generated or which are
on the frontier. That separation is what stops a preference slider from
silently becoming a selection bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

# Objective key -> whether higher is better.
OBJECTIVES: dict[str, bool] = {
    "excessReturn": True,
    "annualReturn": True,
    "maxDrawdown": False,
    "robustness": True,
}


@dataclass(frozen=True)
class ObjectivePreference:
    """Operator preference over the four objectives. Normalised on construction."""

    excess_return: float = 0.40
    annual_return: float = 0.20
    drawdown_control: float = 0.25
    robustness: float = 0.15

    def normalised(self) -> dict[str, float]:
        raw = {
            "excessReturn": max(0.0, float(self.excess_return)),
            "annualReturn": max(0.0, float(self.annual_return)),
            "maxDrawdown": max(0.0, float(self.drawdown_control)),
            "robustness": max(0.0, float(self.robustness)),
        }
        total = sum(raw.values())
        if total <= 0:
            return {key: 0.25 for key in raw}
        return {key: value / total for key, value in raw.items()}


def _objective_vector(metrics: Mapping[str, float]) -> np.ndarray:
    """Orient every objective so that larger is better."""
    return np.array(
        [
            float(metrics.get(key, 0.0)) * (1.0 if higher_is_better else -1.0)
            for key, higher_is_better in OBJECTIVES.items()
        ],
        dtype=float,
    )


def dominates(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    """True when ``left`` is at least as good everywhere and strictly better once."""
    a = _objective_vector(left)
    b = _objective_vector(right)
    return bool(np.all(a >= b) and np.any(a > b))


def pareto_front(candidates: Sequence[Mapping[str, object]]) -> list[str]:
    """Return the ids of non-dominated candidates.

    Each element must expose ``id`` and ``metrics``. Candidates with no usable
    observations are excluded: an empty evaluation is absence of evidence, not a
    zero-drawdown result, and letting it onto the frontier would be a lie.
    """
    usable = [
        item
        for item in candidates
        if int(_metric(item, "observations", 0)) > 0
    ]
    front: list[str] = []
    for item in usable:
        item_metrics = _metrics(item)
        if any(
            dominates(_metrics(other), item_metrics)
            for other in usable
            if other is not item
        ):
            continue
        front.append(str(item["id"]))
    return front


def _metrics(item: Mapping[str, object]) -> Mapping[str, float]:
    metrics = item.get("metrics")
    return metrics if isinstance(metrics, Mapping) else {}


def _metric(item: Mapping[str, object], key: str, default: float) -> float:
    value = _metrics(item).get(key, default)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _minmax(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if array.size == 0:
        return 0.0, 1.0
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-12:
        return low, low + 1.0
    return low, high


def rank_by_preference(
    candidates: Sequence[Mapping[str, object]],
    preference: ObjectivePreference | None = None,
) -> list[dict[str, object]]:
    """Score candidates against operator preference and sort best-first.

    Each objective is min-max normalised **across the evaluated set** so the
    weights are comparable; ``maxDrawdown`` is inverted first so that larger is
    always better. The returned dicts carry a ``preferenceScore`` in ``[0, 1]``
    and the per-objective normalised contributions, so the UI can explain the
    ranking instead of asserting it.
    """
    weights = (preference or ObjectivePreference()).normalised()
    usable = [item for item in candidates if int(_metric(item, "observations", 0)) > 0]
    if not usable:
        return []

    bounds = {
        key: _minmax(_metric(item, key, 0.0) for item in usable)
        for key in OBJECTIVES
    }

    scored: list[dict[str, object]] = []
    for item in usable:
        contributions: dict[str, float] = {}
        total = 0.0
        for key, higher_is_better in OBJECTIVES.items():
            low, high = bounds[key]
            raw = _metric(item, key, 0.0)
            unit = (raw - low) / (high - low)
            if not higher_is_better:
                unit = 1.0 - unit
            unit = float(np.clip(unit, 0.0, 1.0))
            contributions[key] = round(unit * weights[key], 6)
            total += unit * weights[key]
        scored.append(
            {
                "id": str(item["id"]),
                "preferenceScore": round(float(np.clip(total, 0.0, 1.0)), 6),
                "contributions": contributions,
            }
        )
    scored.sort(key=lambda row: float(row["preferenceScore"]), reverse=True)
    return scored


__all__ = [
    "OBJECTIVES",
    "ObjectivePreference",
    "dominates",
    "pareto_front",
    "rank_by_preference",
]
