from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pandas as pd

from quantagent.research.model_comparison import (
    ComparisonConfig,
    ComparisonReport,
    run_model_comparison,
)
from quantagent.research.nonlinear_promotion import (
    NonlinearPromotionConfig,
    NonlinearPromotionReport,
    evaluate_nonlinear_promotion,
)


@dataclass(frozen=True)
class GovernedModelComparison:
    """Raw model comparison plus the only production-eligibility verdict.

    ``comparison.verdict`` is a research result: it means a nonlinear arm
    cleared the paired IC/materiality checks and the comparison's own PBO rule.
    It is *not* sufficient for production. ``promotion.final_verdict`` applies
    the repository-wide PBO/DSR/SPA policy against the measured linear control
    and is the authoritative promotion result.
    """

    comparison: ComparisonReport
    promotion: NonlinearPromotionReport

    @property
    def production_eligible(self) -> bool:
        return bool(self.promotion.accepted)

    def to_dict(self) -> dict[str, object]:
        return {
            "productionEligible": self.production_eligible,
            "comparison": self.comparison.as_dict(),
            "promotion": self.promotion.to_dict(),
            "note": (
                "Raw model-comparison verdicts are research evidence only. "
                "Production eligibility is controlled by promotion.final_verdict."
            ),
        }


def run_governed_model_comparison(
    panel: pd.DataFrame,
    factor_columns: Sequence[str],
    *,
    comparison_config: ComparisonConfig | None = None,
    promotion_config: NonlinearPromotionConfig | None = None,
    regime_by_date: pd.Series | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> GovernedModelComparison:
    """Measure nonlinear challengers, then apply fail-closed promotion gates."""
    comparison = run_model_comparison(
        panel,
        factor_columns,
        config=comparison_config,
        regime_by_date=regime_by_date,
        progress=progress,
    )
    promotion = evaluate_nonlinear_promotion(
        comparison,
        config=promotion_config,
    )
    return GovernedModelComparison(comparison=comparison, promotion=promotion)


__all__ = ["GovernedModelComparison", "run_governed_model_comparison"]
