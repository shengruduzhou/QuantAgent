from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quantagent.quant_math.performance import spa_test
from quantagent.research.model_comparison import ComparisonReport, LINEAR_BASELINE


@dataclass(frozen=True)
class NonlinearPromotionConfig:
    """Fail-closed research-to-production gates for nonlinear factor models."""

    max_pbo: float = 0.25
    min_dsr_probability: float = 0.95
    max_spa_pvalue: float = 0.05
    spa_bootstrap: int = 1000

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_pbo <= 1.0:
            raise ValueError("max_pbo must be in [0,1]")
        if not 0.0 <= self.min_dsr_probability <= 1.0:
            raise ValueError("min_dsr_probability must be in [0,1]")
        if not 0.0 <= self.max_spa_pvalue <= 1.0:
            raise ValueError("max_spa_pvalue must be in [0,1]")
        if self.spa_bootstrap < 100:
            raise ValueError("spa_bootstrap must be >= 100")


@dataclass(frozen=True)
class NonlinearPromotionReport:
    raw_verdict: str
    final_verdict: str
    champion: str
    pbo: float
    dsr_probability: float
    spa_pvalue: float
    accepted: bool
    rejection_reasons: tuple[str, ...]
    config: NonlinearPromotionConfig

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["config"] = asdict(self.config)
        return payload


def evaluate_nonlinear_promotion(
    comparison: ComparisonReport,
    *,
    config: NonlinearPromotionConfig | None = None,
) -> NonlinearPromotionReport:
    """Apply PBO/DSR/SPA to a frozen nonlinear champion versus linear control."""
    cfg = config or NonlinearPromotionConfig()
    reasons: list[str] = []
    raw = str(comparison.verdict)
    champion = str(comparison.champion or "")

    if raw != "production_accepted" or not champion or champion == LINEAR_BASELINE:
        reasons.append(
            f"model comparison did not freeze a promotable nonlinear champion: verdict={raw}, champion={champion or 'none'}"
        )

    pbo = float(comparison.pbo)
    dsr = float(comparison.dsr_probability)
    if not np.isfinite(pbo) or pbo > cfg.max_pbo:
        reasons.append(f"pbo={pbo:.4f} exceeds {cfg.max_pbo:.4f}" if np.isfinite(pbo) else "pbo is unavailable")
    if not np.isfinite(dsr) or dsr < cfg.min_dsr_probability:
        reasons.append(f"dsr={dsr:.4f} below {cfg.min_dsr_probability:.4f}" if np.isfinite(dsr) else "dsr is unavailable")

    baseline = next((arm for arm in comparison.arms if arm.name == LINEAR_BASELINE and arm.status == "measured"), None)
    challengers = [
        arm for arm in comparison.arms
        if arm.status == "measured" and arm.name != LINEAR_BASELINE and not arm.daily_returns.empty
    ]
    spa_pvalue = float("nan")
    if baseline is None or baseline.daily_returns.empty or not challengers:
        reasons.append("SPA cannot be computed: measured linear baseline and nonlinear candidate returns are required")
    else:
        candidate_returns = pd.DataFrame({arm.name: arm.daily_returns for arm in challengers}).sort_index()
        spa = spa_test(candidate_returns, baseline.daily_returns.sort_index(), n_bootstrap=cfg.spa_bootstrap, rng_seed=0)
        spa_pvalue = float(spa.get("p_consistent", float("nan")))
        if not np.isfinite(spa_pvalue) or spa_pvalue > cfg.max_spa_pvalue:
            reasons.append(
                f"spa_pvalue={spa_pvalue:.4f} exceeds {cfg.max_spa_pvalue:.4f}"
                if np.isfinite(spa_pvalue) else "SPA p-value is unavailable"
            )

    accepted = not reasons
    return NonlinearPromotionReport(
        raw_verdict=raw,
        final_verdict="production_accepted" if accepted else "hypothesis_rejected",
        champion=champion,
        pbo=pbo,
        dsr_probability=dsr,
        spa_pvalue=spa_pvalue,
        accepted=accepted,
        rejection_reasons=tuple(reasons),
        config=cfg,
    )


__all__ = ["NonlinearPromotionConfig", "NonlinearPromotionReport", "evaluate_nonlinear_promotion"]
