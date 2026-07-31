"""Factor-fusion search: discover and rank factor blends out of sample.

This package implements layer L2 of the ATLAS architecture described in
``docs/research/institutional_quant_reference_architecture.md``. It answers a
single question: *given a reviewed factor library, which blend of those factors
survives purged walk-forward validation on the four declared objectives —
excess return, drawdown, annualised return and out-of-sample robustness?*

The package is deliberately a **research** optimiser. It never emits orders and
never claims a tradable result: the production acceptance path remains
``scripts/baseline_protocol.py`` variant C with the full A-share execution
model. What this package adds over ad-hoc searching is bookkeeping that makes
the search statistically honest:

* every evaluated blend is counted into ``n_trials``, and ``n_trials`` feeds the
  deflated Sharpe ratio — the operator cannot set it by hand;
* fitted schemes see only the training segment of each fold;
* the Pareto frontier is reported instead of a single "best" configuration,
  because the four objectives genuinely conflict.
"""

from quantagent.fusion.evaluation import (
    SegmentEvaluation,
    cross_sectional_scores,
    evaluate_blend,
)
from quantagent.fusion.frontier import (
    ObjectivePreference,
    pareto_front,
    rank_by_preference,
)
from quantagent.fusion.robustness import (
    RobustnessInputs,
    RobustnessWeights,
    fold_consistency,
    robustness_score,
)
from quantagent.fusion.schemes import (
    BlendScheme,
    SchemeSpec,
    build_scheme_specs,
    derive_weights,
)
from quantagent.fusion.search import (
    FusionCandidate,
    FusionSearchConfig,
    FusionSearchResult,
    run_fusion_search,
    save_fusion_artifacts,
)

__all__ = [
    "BlendScheme",
    "FusionCandidate",
    "FusionSearchConfig",
    "FusionSearchResult",
    "ObjectivePreference",
    "RobustnessInputs",
    "RobustnessWeights",
    "SchemeSpec",
    "SegmentEvaluation",
    "build_scheme_specs",
    "cross_sectional_scores",
    "derive_weights",
    "evaluate_blend",
    "fold_consistency",
    "pareto_front",
    "rank_by_preference",
    "robustness_score",
    "run_fusion_search",
    "save_fusion_artifacts",
]
