"""Self-optimisation utilities for V7 research loops."""

from quantagent.optimization.optuna_search import (
    OptunaSearchConfig,
    OptunaSearchResult,
    build_search_space,
    run_optuna_hp_search,
)
from quantagent.optimization.factor_evolution import (
    FactorEvolutionConfig,
    FactorEvolutionResult,
    run_factor_evolution,
)
from quantagent.optimization.multi_objective_loss import (
    LossWeights,
    LossComponents,
    compute_multi_objective_loss,
    score_backtest,
)

__all__ = [
    "OptunaSearchConfig",
    "OptunaSearchResult",
    "build_search_space",
    "run_optuna_hp_search",
    "FactorEvolutionConfig",
    "FactorEvolutionResult",
    "run_factor_evolution",
    "LossWeights",
    "LossComponents",
    "compute_multi_objective_loss",
    "score_backtest",
]
