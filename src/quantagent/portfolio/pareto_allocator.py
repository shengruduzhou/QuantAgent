"""Hard-constrained Pareto portfolio construction.

Return, drawdown/risk, turnover, cost and capacity are not collapsed into one
unbounded optimizer loss.  Hard constraints are applied first; the remaining
feasible candidates form a Pareto frontier.  The returned choice is a policy
selection from that frontier, not proof that one scalar objective is optimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from quantagent.quant_math.optimizer import ContinuousMeanVarianceOptimizer, OptimizerConfig


REGIME_GROSS_RANGE: dict[str, tuple[float, float]] = {
    "crisis": (0.00, 0.15),
    "bear_capitulation": (0.05, 0.30),
    "caution": (0.20, 0.50),
    "normal": (0.35, 0.65),
    "bull_consolidation": (0.45, 0.72),
    "bull_expansion": (0.55, 0.80),
}


@dataclass(frozen=True)
class PortfolioHardConstraints:
    target_book_cny: float = 10_000_000.0
    max_name_weight: float = 0.08
    max_sector_weight: float = 0.30
    max_style_exposure: float = 0.25
    max_turnover: float = 0.35
    max_adv_participation: float = 0.10
    min_cash_weight: float = 0.20
    max_gross_weight: float = 0.80
    min_names: int = 10


@dataclass(frozen=True)
class ParetoSearchConfig:
    risk_aversion_grid: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
    turnover_penalty_grid: tuple[float, ...] = (0.05, 0.20, 0.50)
    cost_penalty_grid: tuple[float, ...] = (0.5, 1.0, 2.0)
    gross_scale_grid: tuple[float, ...] = (0.60, 0.80, 1.00)
    selection_policy: str = "balanced"


@dataclass
class PortfolioCandidate:
    candidate_id: str
    weights: pd.Series
    expected_alpha: float
    expected_volatility: float
    turnover: float
    expected_cost: float
    concentration_hhi: float
    max_sector_weight: float
    max_style_exposure: float
    min_capacity_multiple: float
    cash_weight: float
    feasible: bool
    violations: list[str] = field(default_factory=list)
    optimizer_status: str = "unknown"

    def objectives(self) -> tuple[float, float, float, float, float]:
        # First coordinate is negated because Pareto comparison minimises all.
        return (
            -self.expected_alpha,
            self.expected_volatility,
            self.turnover,
            self.expected_cost,
            self.concentration_hhi,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "expected_alpha": self.expected_alpha,
            "expected_volatility": self.expected_volatility,
            "turnover": self.turnover,
            "expected_cost": self.expected_cost,
            "concentration_hhi": self.concentration_hhi,
            "max_sector_weight": self.max_sector_weight,
            "max_style_exposure": self.max_style_exposure,
            "min_capacity_multiple": self.min_capacity_multiple,
            "cash_weight": self.cash_weight,
            "feasible": self.feasible,
            "violations": list(self.violations),
            "optimizer_status": self.optimizer_status,
        }


@dataclass
class ParetoAllocationResult:
    selected: PortfolioCandidate
    frontier: list[PortfolioCandidate]
    rejected: list[PortfolioCandidate]
    gross_budget: float


def gross_exposure_budget(
    regime: str,
    *,
    blend_confidence: float,
    breadth_score: float = 0.5,
    liquidity_score: float = 0.5,
) -> float:
    """Map market state and calibrated confidence to a bounded gross budget."""
    lower, upper = REGIME_GROSS_RANGE.get(str(regime), REGIME_GROSS_RANGE["normal"])
    confidence = float(np.clip(blend_confidence, 0.0, 1.0))
    breadth = float(np.clip(breadth_score, 0.0, 1.0))
    liquidity = float(np.clip(liquidity_score, 0.0, 1.0))
    quality = 0.50 * confidence + 0.30 * breadth + 0.20 * liquidity
    return float(lower + (upper - lower) * quality)


def _capacity_upper_bounds(
    alpha_index: pd.Index,
    adv20_cny: pd.Series | None,
    constraints: PortfolioHardConstraints,
) -> pd.Series:
    hard_cap = pd.Series(constraints.max_name_weight, index=alpha_index, dtype=float)
    if adv20_cny is None:
        return hard_cap
    adv = pd.to_numeric(adv20_cny.reindex(alpha_index), errors="coerce").fillna(0.0)
    capacity_cap = adv * constraints.max_adv_participation / max(constraints.target_book_cny, 1.0)
    return pd.concat([hard_cap.rename("hard"), capacity_cap.rename("capacity")], axis=1).min(axis=1).clip(lower=0.0)


def _sector_max(weights: pd.Series, sector: pd.Series | None) -> float:
    if sector is None or weights.empty:
        return 0.0
    grouped = weights.groupby(sector.reindex(weights.index).fillna("UNKNOWN")).sum()
    return float(grouped.max()) if not grouped.empty else 0.0


def _style_max(weights: pd.Series, style_exposures: pd.DataFrame | None) -> float:
    if style_exposures is None or weights.empty:
        return 0.0
    exposures = style_exposures.reindex(index=weights.index).fillna(0.0)
    portfolio = exposures.T @ weights
    return float(portfolio.abs().max()) if not portfolio.empty else 0.0


def _capacity_multiple(
    weights: pd.Series,
    adv20_cny: pd.Series | None,
    constraints: PortfolioHardConstraints,
) -> float:
    active = weights[weights > 1e-12]
    if active.empty:
        return 0.0
    if adv20_cny is None:
        return 0.0
    adv = pd.to_numeric(adv20_cny.reindex(active.index), errors="coerce").fillna(0.0)
    executable_cny = adv * constraints.max_adv_participation
    required_cny = active * constraints.target_book_cny
    ratio = executable_cny / required_cny.replace(0.0, np.nan)
    return float(ratio.min()) if not ratio.dropna().empty else 0.0


def _candidate_metrics(
    *,
    candidate_id: str,
    weights: pd.Series,
    alpha: pd.Series,
    covariance: pd.DataFrame,
    current_weights: pd.Series,
    cost: pd.Series,
    sector: pd.Series | None,
    style_exposures: pd.DataFrame | None,
    adv20_cny: pd.Series | None,
    constraints: PortfolioHardConstraints,
    optimizer_status: str,
) -> PortfolioCandidate:
    weights = weights.reindex(alpha.index).fillna(0.0).clip(lower=0.0)
    expected_alpha = float((weights * alpha).sum())
    cov = covariance.reindex(index=weights.index, columns=weights.index).fillna(0.0)
    variance = float(weights.to_numpy() @ cov.to_numpy() @ weights.to_numpy())
    turnover = float((weights - current_weights).abs().sum())
    expected_cost = float(((weights - current_weights).abs() * cost).sum())
    hhi = float(np.square(weights.to_numpy()).sum())
    sector_max = _sector_max(weights, sector)
    style_max = _style_max(weights, style_exposures)
    capacity_multiple = _capacity_multiple(weights, adv20_cny, constraints)
    cash = float(max(0.0, 1.0 - weights.sum()))
    active_names = int((weights > 1e-6).sum())

    violations: list[str] = []
    if float(weights.max()) > constraints.max_name_weight + 1e-9:
        violations.append("max_name_weight")
    if sector_max > constraints.max_sector_weight + 1e-9:
        violations.append("max_sector_weight")
    if style_max > constraints.max_style_exposure + 1e-9:
        violations.append("max_style_exposure")
    if turnover > constraints.max_turnover + 1e-9:
        violations.append("max_turnover")
    if cash + 1e-9 < constraints.min_cash_weight:
        violations.append("min_cash_weight")
    if weights.sum() > constraints.max_gross_weight + 1e-9:
        violations.append("max_gross_weight")
    if active_names < constraints.min_names:
        violations.append("min_names")
    if capacity_multiple < 1.0:
        violations.append("adv_capacity")

    return PortfolioCandidate(
        candidate_id=candidate_id,
        weights=weights,
        expected_alpha=expected_alpha,
        expected_volatility=float(np.sqrt(max(variance, 0.0))),
        turnover=turnover,
        expected_cost=expected_cost,
        concentration_hhi=hhi,
        max_sector_weight=sector_max,
        max_style_exposure=style_max,
        min_capacity_multiple=capacity_multiple,
        cash_weight=cash,
        feasible=not violations,
        violations=violations,
        optimizer_status=optimizer_status,
    )


def _dominates(left: PortfolioCandidate, right: PortfolioCandidate) -> bool:
    a = left.objectives()
    b = right.objectives()
    return all(x <= y + 1e-12 for x, y in zip(a, b)) and any(
        x < y - 1e-12 for x, y in zip(a, b)
    )


def pareto_frontier(candidates: Iterable[PortfolioCandidate]) -> list[PortfolioCandidate]:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    return [
        candidate
        for candidate in feasible
        if not any(_dominates(other, candidate) for other in feasible if other is not candidate)
    ]


def _select_frontier(frontier: list[PortfolioCandidate], policy: str) -> PortfolioCandidate:
    if not frontier:
        raise ValueError("Pareto frontier is empty")
    metrics = pd.DataFrame(
        [
            {
                "alpha": item.expected_alpha,
                "risk": item.expected_volatility,
                "turnover": item.turnover,
                "cost": item.expected_cost,
                "hhi": item.concentration_hhi,
            }
            for item in frontier
        ]
    )
    normalised = pd.DataFrame(index=metrics.index)
    for column in metrics.columns:
        lo, hi = float(metrics[column].min()), float(metrics[column].max())
        normalised[column] = 0.5 if hi - lo <= 1e-12 else (metrics[column] - lo) / (hi - lo)
    if policy == "return_first":
        utility = normalised["alpha"] - 0.15 * normalised[["risk", "turnover", "cost", "hhi"]].mean(axis=1)
    elif policy == "risk_first":
        utility = 0.35 * normalised["alpha"] - normalised[["risk", "turnover", "cost", "hhi"]].mean(axis=1)
    else:
        utility = normalised["alpha"] - 0.60 * normalised[["risk", "turnover", "cost", "hhi"]].mean(axis=1)
    return frontier[int(utility.idxmax())]


def allocate_pareto_portfolio(
    *,
    alpha: pd.Series,
    covariance: pd.DataFrame,
    current_weights: pd.Series | None = None,
    cost: pd.Series | None = None,
    sector: pd.Series | None = None,
    style_exposures: pd.DataFrame | None = None,
    adv20_cny: pd.Series | None = None,
    regime: str = "normal",
    blend_confidence: float = 0.5,
    breadth_score: float = 0.5,
    liquidity_score: float = 0.5,
    constraints: PortfolioHardConstraints | None = None,
    search: ParetoSearchConfig | None = None,
) -> ParetoAllocationResult:
    constraints = constraints or PortfolioHardConstraints()
    search = search or ParetoSearchConfig()
    clean_alpha = pd.to_numeric(alpha, errors="coerce").dropna().astype(float)
    if clean_alpha.empty:
        raise ValueError("alpha is empty")
    current = (
        pd.to_numeric(current_weights.reindex(clean_alpha.index), errors="coerce").fillna(0.0)
        if current_weights is not None
        else pd.Series(0.0, index=clean_alpha.index)
    )
    cost_series = (
        pd.to_numeric(cost.reindex(clean_alpha.index), errors="coerce").fillna(0.0)
        if cost is not None
        else pd.Series(0.0, index=clean_alpha.index)
    )
    upper = _capacity_upper_bounds(clean_alpha.index, adv20_cny, constraints)
    regime_budget = min(
        constraints.max_gross_weight,
        gross_exposure_budget(
            regime,
            blend_confidence=blend_confidence,
            breadth_score=breadth_score,
            liquidity_score=liquidity_score,
        ),
    )

    candidates: list[PortfolioCandidate] = []
    for risk_aversion, turnover_penalty, cost_penalty, gross_scale in product(
        search.risk_aversion_grid,
        search.turnover_penalty_grid,
        search.cost_penalty_grid,
        search.gross_scale_grid,
    ):
        gross_target = min(regime_budget * gross_scale, constraints.max_gross_weight)
        optimizer = ContinuousMeanVarianceOptimizer(
            OptimizerConfig(
                risk_aversion=float(risk_aversion),
                turnover_penalty=float(turnover_penalty),
                cost_penalty=float(cost_penalty),
                max_position_weight=constraints.max_name_weight,
                max_total_weight=float(gross_target),
                max_turnover=constraints.max_turnover,
                long_only=True,
            )
        )
        result = optimizer.solve(
            clean_alpha,
            covariance,
            current_weights=current,
            cost=cost_series,
            sector=sector,
            max_sector_weight=constraints.max_sector_weight,
            upper_bounds=upper,
        )
        candidate_id = (
            f"ra{risk_aversion:g}_tp{turnover_penalty:g}_cp{cost_penalty:g}_g{gross_scale:g}"
        )
        candidates.append(
            _candidate_metrics(
                candidate_id=candidate_id,
                weights=result.weights,
                alpha=clean_alpha,
                covariance=covariance,
                current_weights=current,
                cost=cost_series,
                sector=sector,
                style_exposures=style_exposures,
                adv20_cny=adv20_cny,
                constraints=constraints,
                optimizer_status=result.status,
            )
        )

    frontier = pareto_frontier(candidates)
    if not frontier:
        summary = sorted({violation for candidate in candidates for violation in candidate.violations})
        raise ValueError(f"no feasible portfolio candidates; violations={summary}")
    selected = _select_frontier(frontier, search.selection_policy)
    return ParetoAllocationResult(
        selected=selected,
        frontier=frontier,
        rejected=[candidate for candidate in candidates if not candidate.feasible],
        gross_budget=regime_budget,
    )
