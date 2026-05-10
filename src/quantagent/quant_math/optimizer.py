from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptimizerConfig:
    risk_aversion: float = 8.0
    turnover_penalty: float = 0.2
    cost_penalty: float = 1.0
    max_position_weight: float = 0.08
    max_total_weight: float = 0.95
    max_turnover: float = 0.20
    long_only: bool = True
    solver: str | None = None


@dataclass(frozen=True)
class OptimizerResult:
    weights: pd.Series
    status: str
    objective_value: float | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class V4PortfolioConfig:
    mode: str = "long_only_enhancement"
    max_name_weight: float = 0.05
    max_sector_weight: float = 0.30
    max_turnover: float = 0.30
    target_beta: float = 1.0
    beta_limit: float = 0.2
    cost_aware: bool = True
    no_buy_limit_up: bool = True
    no_sell_limit_down: bool = True


@dataclass(frozen=True)
class V4PortfolioResult:
    target_weights: pd.Series
    expected_turnover: float
    expected_cost: float
    active_risk_proxy: float
    constraint_diagnostics: dict[str, float | str]
    rejected_symbols: dict[str, str]
    status: str


class ContinuousMeanVarianceOptimizer:
    """Mean-variance optimizer with cost, turnover, and exposure constraints."""

    def __init__(self, config: OptimizerConfig | None = None) -> None:
        self.config = config or OptimizerConfig()

    def solve(
        self,
        alpha: pd.Series,
        covariance: pd.DataFrame,
        current_weights: pd.Series | None = None,
        cost: pd.Series | None = None,
        sector: pd.Series | None = None,
        max_sector_weight: float | None = None,
        beta: pd.Series | None = None,
        beta_target: float | None = None,
        beta_limit: float | None = None,
        upper_bounds: pd.Series | None = None,
    ) -> OptimizerResult:
        symbols = alpha.dropna().index
        if len(symbols) == 0:
            return OptimizerResult(weights=pd.Series(dtype=float), status="empty_universe")
        cov = covariance.reindex(index=symbols, columns=symbols).fillna(0.0)
        current = _align(current_weights, symbols, 0.0)
        cost = _align(cost, symbols, 0.0)
        upper = _align(upper_bounds, symbols, self.config.max_position_weight).clip(
            upper=self.config.max_position_weight
        )
        try:
            return self._solve_cvxpy(
                alpha.loc[symbols],
                cov,
                current,
                cost,
                sector.reindex(symbols) if sector is not None else None,
                max_sector_weight,
                beta.reindex(symbols) if beta is not None else None,
                beta_target,
                beta_limit,
                upper,
            )
        except ImportError:
            return self._solve_fallback(alpha.loc[symbols], cov, current, cost, upper)
        except Exception:
            return self._solve_fallback(
                alpha.loc[symbols],
                cov,
                current,
                cost,
                upper,
                status="fallback_optimizer_error",
            )

    def _solve_cvxpy(
        self,
        alpha: pd.Series,
        covariance: pd.DataFrame,
        current: pd.Series,
        cost: pd.Series,
        sector: pd.Series | None,
        max_sector_weight: float | None,
        beta: pd.Series | None,
        beta_target: float | None,
        beta_limit: float | None,
        upper: pd.Series,
    ) -> OptimizerResult:
        try:
            import cvxpy as cp
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("cvxpy is not installed") from exc

        n = len(alpha)
        weights = cp.Variable(n)
        delta = weights - current.to_numpy()
        cov_matrix = _positive_semidefinite(covariance.to_numpy())
        objective = cp.Maximize(
            alpha.to_numpy() @ weights
            - self.config.risk_aversion * cp.quad_form(weights, cp.psd_wrap(cov_matrix))
            - self.config.turnover_penalty * cp.norm1(delta)
            - self.config.cost_penalty * cp.sum(cp.multiply(cost.to_numpy(), cp.abs(delta)))
        )
        constraints = [cp.sum(weights) <= self.config.max_total_weight]
        constraints.append(cp.norm1(delta) <= self.config.max_turnover)
        constraints.append(weights <= upper.to_numpy())
        if self.config.long_only:
            constraints.append(weights >= 0)
        if sector is not None and max_sector_weight is not None:
            sector_array = sector.to_numpy()
            for sector_label in pd.unique(sector_array):
                idx = np.where(sector_array == sector_label)[0]
                if idx.size > 0:
                    constraints.append(cp.sum(weights[idx]) <= max_sector_weight)
        if beta is not None and beta_target is not None and beta_limit is not None:
            portfolio_beta = beta.to_numpy() @ weights
            constraints.append(portfolio_beta <= beta_target + beta_limit)
            constraints.append(portfolio_beta >= beta_target - beta_limit)

        problem = cp.Problem(objective, constraints)
        if self.config.solver:
            problem.solve(solver=self.config.solver)
        else:
            problem.solve()
        if weights.value is None:
            return self._solve_fallback(alpha, covariance, current, cost, upper, status="fallback_no_solution")
        result = pd.Series(np.maximum(weights.value, 0.0), index=alpha.index)
        return OptimizerResult(
            weights=result,
            status=str(problem.status),
            objective_value=float(problem.value) if problem.value is not None else None,
            diagnostics=_diagnostics(result, current, covariance),
        )

    def _solve_fallback(
        self,
        alpha: pd.Series,
        covariance: pd.DataFrame,
        current: pd.Series,
        cost: pd.Series,
        upper: pd.Series,
        status: str = "fallback_no_cvxpy",
    ) -> OptimizerResult:
        risk = pd.Series(np.sqrt(np.clip(np.diag(covariance.to_numpy()), 1e-12, None)), index=alpha.index)
        score = (alpha - cost).clip(lower=0.0) / risk
        if score.sum() <= 0:
            weights = pd.Series(0.0, index=alpha.index)
        else:
            weights = score / score.sum() * self.config.max_total_weight
            weights = weights.clip(upper=upper)
            if weights.sum() > self.config.max_total_weight:
                weights = weights / weights.sum() * self.config.max_total_weight
        turnover = (weights - current).abs().sum()
        if turnover > self.config.max_turnover:
            scale = self.config.max_turnover / turnover
            weights = current + (weights - current) * scale
        return OptimizerResult(
            weights=weights,
            status=status,
            objective_value=None,
            diagnostics=_diagnostics(weights, current, covariance),
        )


def _align(series: pd.Series | None, index: pd.Index, fill_value: float) -> pd.Series:
    if series is None:
        return pd.Series(fill_value, index=index)
    return series.reindex(index).fillna(fill_value)


def _diagnostics(weights: pd.Series, current: pd.Series, covariance: pd.DataFrame) -> dict[str, float]:
    cov = covariance.reindex(index=weights.index, columns=weights.index).fillna(0.0)
    variance = float(weights.to_numpy() @ cov.to_numpy() @ weights.to_numpy())
    return {
        "gross_weight": float(weights.abs().sum()),
        "net_weight": float(weights.sum()),
        "turnover": float((weights - current.reindex(weights.index).fillna(0.0)).abs().sum()),
        "portfolio_volatility": float(np.sqrt(max(variance, 0.0))),
    }


def _positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    matrix = (matrix + matrix.T) / 2.0
    eig_min = np.linalg.eigvalsh(matrix).min()
    if eig_min < 1e-10:
        matrix = matrix + np.eye(matrix.shape[0]) * (abs(eig_min) + 1e-8)
    return matrix


def solve_v4_portfolio(
    alpha: pd.Series,
    covariance: pd.DataFrame,
    current_weights: pd.Series | None = None,
    cost: pd.Series | None = None,
    sector: pd.Series | None = None,
    beta: pd.Series | None = None,
    tradability: pd.DataFrame | None = None,
    config: V4PortfolioConfig | None = None,
) -> V4PortfolioResult:
    cfg = config or V4PortfolioConfig()
    rejected: dict[str, str] = {}
    clean_alpha = alpha.dropna().astype(float)
    if tradability is not None and not tradability.empty:
        for _, row in tradability.iterrows():
            symbol = str(row["symbol"])
            if symbol not in clean_alpha.index:
                continue
            if cfg.no_buy_limit_up and bool(row.get("is_limit_up", False)) and clean_alpha.loc[symbol] > 0:
                rejected[symbol] = "limit_up_no_buy"
            if cfg.no_sell_limit_down and bool(row.get("is_limit_down", False)) and clean_alpha.loc[symbol] < 0:
                rejected[symbol] = "limit_down_no_sell"
            if bool(row.get("is_suspended", row.get("suspended", False))):
                rejected[symbol] = "suspended"
        clean_alpha = clean_alpha.drop(index=[s for s in rejected if s in clean_alpha.index])
    if cfg.mode == "hedged_alpha":
        clean_alpha = clean_alpha - clean_alpha.mean()
    elif cfg.mode == "market_neutral_placeholder":
        clean_alpha = clean_alpha - clean_alpha.mean()
    optimizer = ContinuousMeanVarianceOptimizer(
        OptimizerConfig(
            max_position_weight=cfg.max_name_weight,
            max_total_weight=1.0 if cfg.mode == "long_only_enhancement" else 0.5,
            max_turnover=cfg.max_turnover,
            long_only=cfg.mode == "long_only_enhancement",
            cost_penalty=1.0 if cfg.cost_aware else 0.0,
        )
    )
    upper = pd.Series(cfg.max_name_weight, index=clean_alpha.index)
    result = optimizer.solve(
        clean_alpha,
        covariance,
        current_weights=current_weights,
        cost=cost,
        sector=sector,
        max_sector_weight=cfg.max_sector_weight,
        beta=beta,
        beta_target=cfg.target_beta if beta is not None and cfg.mode == "long_only_enhancement" else None,
        beta_limit=cfg.beta_limit,
        upper_bounds=upper,
    )
    weights = result.weights
    if cfg.mode in {"hedged_alpha", "market_neutral_placeholder"} and not weights.empty:
        weights = weights - weights.mean()
        gross = weights.abs().sum()
        if gross > 1.0:
            weights = weights / gross
    current = current_weights.reindex(weights.index).fillna(0.0) if current_weights is not None else pd.Series(0.0, index=weights.index)
    turnover = float((weights - current).abs().sum())
    cost_series = cost.reindex(weights.index).fillna(0.0) if cost is not None else pd.Series(0.0, index=weights.index)
    expected_cost = float(((weights - current).abs() * cost_series).sum())
    cov = covariance.reindex(index=weights.index, columns=weights.index).fillna(0.0)
    active_risk = float(np.sqrt(max(weights.to_numpy() @ cov.to_numpy() @ weights.to_numpy(), 0.0))) if len(weights) else 0.0
    diagnostics: dict[str, float | str] = {
        **result.diagnostics,
        "mode": cfg.mode,
        "expected_turnover": turnover,
        "expected_cost": expected_cost,
        "active_risk_proxy": active_risk,
        "rejected_count": float(len(rejected)),
    }
    return V4PortfolioResult(
        target_weights=weights.sort_index(),
        expected_turnover=turnover,
        expected_cost=expected_cost,
        active_risk_proxy=active_risk,
        constraint_diagnostics=diagnostics,
        rejected_symbols=rejected,
        status=result.status,
    )
