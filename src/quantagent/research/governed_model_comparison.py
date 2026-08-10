from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from quantagent.research.experiment_governance import (
    ExperimentEvent,
    ExperimentLedger,
    ExperimentSpec,
    FinalHoldoutLedger,
    FinalHoldoutSpec,
    with_cumulative_trial_count,
)
from quantagent.research.model_comparison import (
    ComparisonConfig,
    ComparisonReport,
    LINEAR_BASELINE,
    run_model_comparison,
)
from quantagent.research.nonlinear_promotion import (
    NonlinearPromotionConfig,
    NonlinearPromotionReport,
    evaluate_nonlinear_promotion,
)


@dataclass(frozen=True)
class HoldoutQualification:
    """One-shot final-holdout veto for an already frozen nonlinear champion.

    The holdout never chooses among challengers.  It only asks whether the champion
    frozen on selection folds generalises in the same direction versus the linear
    control on both prediction quality and after-cost economic value.
    """

    accepted: bool
    champion: str
    ic_delta: float
    net_return_delta: float
    reasons: tuple[str, ...] = ()
    holdout_start: str = ""
    holdout_end: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "champion": self.champion,
            "icDelta": None if not np.isfinite(self.ic_delta) else float(self.ic_delta),
            "netReturnDelta": (
                None if not np.isfinite(self.net_return_delta) else float(self.net_return_delta)
            ),
            "reasons": list(self.reasons),
            "holdoutStart": self.holdout_start,
            "holdoutEnd": self.holdout_end,
        }


@dataclass(frozen=True)
class GovernedModelComparison:
    """Raw model comparison plus authoritative promotion/holdout eligibility.

    ``comparison.verdict`` is research evidence. ``promotion.final_verdict`` applies
    PBO/DSR/SPA.  When ``holdout`` is present, production eligibility additionally
    requires the one-shot final holdout to confirm the already-frozen champion.
    """

    comparison: ComparisonReport
    promotion: NonlinearPromotionReport
    holdout: HoldoutQualification | None = None
    governance: dict[str, object] = field(default_factory=dict)

    @property
    def production_eligible(self) -> bool:
        holdout_ok = self.holdout is None or self.holdout.accepted
        return bool(self.promotion.accepted and holdout_ok)

    def to_dict(self) -> dict[str, object]:
        return {
            "productionEligible": self.production_eligible,
            "comparison": self.comparison.as_dict(),
            "promotion": self.promotion.to_dict(),
            "holdout": self.holdout.to_dict() if self.holdout is not None else None,
            "governance": dict(self.governance),
            "note": (
                "Raw model-comparison verdicts are research evidence only. "
                "Production eligibility requires promotion gates and, on the Stage-4 path, "
                "a one-shot final-holdout qualification."
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
    """Legacy-compatible governed path: measure challengers, then apply PBO/DSR/SPA.

    New production research should use :func:`run_stage4_governed_model_comparison`,
    which additionally binds cumulative trial history and an atomic one-shot holdout.
    """
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


def run_stage4_governed_model_comparison(
    panel: pd.DataFrame,
    factor_columns: Sequence[str],
    *,
    experiment: ExperimentSpec,
    experiment_ledger: ExperimentLedger,
    final_holdout: FinalHoldoutSpec,
    final_holdout_ledger: FinalHoldoutLedger,
    comparison_config: ComparisonConfig | None = None,
    promotion_config: NonlinearPromotionConfig | None = None,
    regime_by_date: pd.Series | None = None,
    progress: Callable[[str, int, int], None] | None = None,
    run_id: str = "",
) -> GovernedModelComparison:
    """Stage-4 path with deterministic lineage, cumulative trials and one-shot holdout.

    Policy order is deliberate:

    1. validate that search and final-holdout identities cannot overlap;
    2. atomically burn the final holdout before any code capable of observing it runs;
    3. append the research attempt to the experiment ledger;
    4. run the existing fold-purged comparison;
    5. raise ``n_trials`` to the cumulative family attempt count before DSR;
    6. apply PBO/DSR/SPA to the frozen champion;
    7. use the final holdout only as a veto, never as a selector.

    A crash after step 2 intentionally leaves the holdout consumed.  Once final data may
    have been observed, silently reusing it as a fresh final test would be false evidence.
    """

    _validate_stage4_contract(experiment, final_holdout)
    seal = final_holdout_ledger.consume(
        final_holdout,
        candidate_fingerprint=experiment.fingerprint,
        git_hash=experiment.git_hash,
        run_id=run_id,
    )
    event = experiment_ledger.append(
        ExperimentEvent(
            spec=experiment,
            status="executed_with_final_holdout",
            metadata={"holdout_key": final_holdout.holdout_key, "run_id": run_id},
        )
    )

    comparison = run_model_comparison(
        panel,
        factor_columns,
        config=comparison_config,
        regime_by_date=regime_by_date,
        progress=progress,
    )
    cumulative_trials = experiment_ledger.attempt_count(family=experiment.family)
    comparison = with_cumulative_trial_count(comparison, cumulative_trials)
    promotion = evaluate_nonlinear_promotion(comparison, config=promotion_config)
    holdout = _qualify_final_holdout(comparison, expected=final_holdout)
    governance = {
        "experimentFingerprint": experiment.fingerprint,
        "eventHash": event["event_hash"],
        "holdoutKey": final_holdout.holdout_key,
        "holdoutSealHash": seal["seal_hash"],
        "cumulativeFamilyTrials": cumulative_trials,
        "uniqueFamilyFingerprints": experiment_ledger.unique_fingerprint_count(
            family=experiment.family
        ),
    }
    return GovernedModelComparison(
        comparison=comparison,
        promotion=promotion,
        holdout=holdout,
        governance=governance,
    )


def _validate_stage4_contract(experiment: ExperimentSpec, holdout: FinalHoldoutSpec) -> None:
    if experiment.family != holdout.family:
        raise ValueError("experiment and final holdout must belong to the same family")
    if experiment.dataset_hash != holdout.dataset_hash:
        raise ValueError("experiment and final holdout dataset_hash must match")
    search_end = pd.Timestamp(experiment.search_window[1])
    holdout_start = pd.Timestamp(holdout.holdout_window[0])
    if search_end >= holdout_start:
        raise ValueError(
            "search_window must end strictly before final holdout starts; "
            "final holdout cannot participate in candidate selection"
        )


def _qualify_final_holdout(
    comparison: ComparisonReport,
    *,
    expected: FinalHoldoutSpec,
) -> HoldoutQualification:
    champion = str(comparison.champion or "")
    reasons: list[str] = []
    holdout_windows = [w for w in comparison.fold_windows if w.get("role") == "holdout"]
    actual_start = min((str(w["validStart"]) for w in holdout_windows), default="")
    actual_end = max((str(w["validEnd"]) for w in holdout_windows), default="")
    if not holdout_windows:
        reasons.append("comparison produced no final holdout folds")
    else:
        expected_start = str(pd.Timestamp(expected.holdout_window[0]).date())
        expected_end = str(pd.Timestamp(expected.holdout_window[1]).date())
        if actual_start != expected_start or actual_end != expected_end:
            reasons.append(
                "executed holdout window does not match the pre-registered final holdout: "
                f"actual={actual_start}..{actual_end}, expected={expected_start}..{expected_end}"
            )

    measured = {arm.name: arm for arm in comparison.arms if arm.status == "measured"}
    baseline = measured.get(LINEAR_BASELINE)
    winner = measured.get(champion)
    if not champion or champion == LINEAR_BASELINE:
        reasons.append("no frozen nonlinear champion is available for final holdout qualification")
    if baseline is None or winner is None:
        reasons.append("final holdout requires measured linear baseline and frozen champion")
        return HoldoutQualification(
            accepted=False,
            champion=champion,
            ic_delta=float("nan"),
            net_return_delta=float("nan"),
            reasons=tuple(reasons),
            holdout_start=actual_start,
            holdout_end=actual_end,
        )

    winner_ic = float(winner.holdout_metrics.get("rank_ic_mean", float("nan")))
    baseline_ic = float(baseline.holdout_metrics.get("rank_ic_mean", float("nan")))
    winner_net = float(winner.holdout_metrics.get("net_annual_return", float("nan")))
    baseline_net = float(baseline.holdout_metrics.get("net_annual_return", float("nan")))
    ic_delta = winner_ic - baseline_ic
    net_delta = winner_net - baseline_net

    if not np.isfinite(winner_ic) or not np.isfinite(baseline_ic):
        reasons.append("final holdout rank IC evidence is unavailable")
    elif winner_ic <= 0.0:
        reasons.append(f"final holdout champion rank IC is not positive: {winner_ic:.6f}")
    elif ic_delta <= 0.0:
        reasons.append(f"final holdout rank IC does not beat linear baseline: delta={ic_delta:+.6f}")

    if not np.isfinite(winner_net) or not np.isfinite(baseline_net):
        reasons.append("final holdout after-cost return evidence is unavailable")
    elif net_delta <= 0.0:
        reasons.append(
            "final holdout after-cost annual return does not beat linear baseline: "
            f"delta={net_delta:+.6f}"
        )

    return HoldoutQualification(
        accepted=not reasons,
        champion=champion,
        ic_delta=ic_delta,
        net_return_delta=net_delta,
        reasons=tuple(reasons),
        holdout_start=actual_start,
        holdout_end=actual_end,
    )


__all__ = [
    "GovernedModelComparison",
    "HoldoutQualification",
    "run_governed_model_comparison",
    "run_stage4_governed_model_comparison",
]
