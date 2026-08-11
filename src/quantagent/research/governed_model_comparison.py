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
    """One-shot final-holdout veto for an already frozen nonlinear champion."""

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
    """Raw comparison plus explicit research-to-production governance state.

    Production eligibility is intentionally fail-closed. A statistical promotion
    decision alone is never sufficient: the result must come from the strict
    Stage-4 path, include an accepted one-shot final holdout, and carry an
    explicit certified economic-backtest flag. Missing governance metadata is a
    blocker rather than an implicit approval.
    """

    comparison: ComparisonReport
    promotion: NonlinearPromotionReport
    holdout: HoldoutQualification | None = None
    governance: dict[str, object] = field(default_factory=dict)

    @property
    def production_eligible(self) -> bool:
        stage4_ok = self.governance.get("stage4Governed") is True
        holdout_ok = self.holdout is not None and self.holdout.accepted
        economic_ok = self.governance.get("economicBacktestCertified") is True
        return bool(self.promotion.accepted and stage4_ok and holdout_ok and economic_ok)

    def to_dict(self) -> dict[str, object]:
        return {
            "productionEligible": self.production_eligible,
            "comparison": self.comparison.as_dict(),
            "promotion": self.promotion.to_dict(),
            "holdout": self.holdout.to_dict() if self.holdout is not None else None,
            "governance": dict(self.governance),
            "note": (
                "Raw model-comparison and statistical-promotion verdicts are research "
                "evidence only. Production eligibility requires the strict Stage-4 "
                "experiment ledger, an accepted one-shot final holdout, and an explicitly "
                "certified executable economic backtest. Missing evidence fails closed."
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
    """Legacy-compatible research path; never grants production eligibility."""
    comparison = run_model_comparison(
        panel,
        factor_columns,
        config=comparison_config,
        regime_by_date=regime_by_date,
        progress=progress,
    )
    promotion = evaluate_nonlinear_promotion(comparison, config=promotion_config)
    governance = {
        "stage4Governed": False,
        "researchOnly": True,
        "economicBacktestCertified": False,
        "productionBlockers": [
            "legacy research path has no ExperimentLedger cumulative multiplicity binding",
            "legacy research path has no one-shot FinalHoldoutLedger qualification",
            "model_comparison economic returns are not yet position-carrying/executable under A-share trading constraints",
        ],
    }
    return GovernedModelComparison(
        comparison=comparison,
        promotion=promotion,
        governance=governance,
    )


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
    """Strict Stage-4 comparison with lineage, multiplicity and one-shot holdout.

    The final block is one trailing fold. Multiple trailing "holdout" folds are forbidden
    here because expanding walk-forward would allow an earlier holdout fold to enter the
    training set of a later holdout fold, so the union would not be an untouched block.
    Increase ``valid_size_days`` when a longer final block is required.

    Stage-4 outcome evidence must use the canonical executable-label contract:
    a T-close signal enters on the next global market session and an H-session
    horizon is named ``forward_executable_return_{H}d``. The final holdout identity
    must also bind a non-empty label-contract hash so a one-shot seal cannot be
    reused under a different outcome clock.

    Statistical governance is strict, but production remains fail-closed while
    ``model_comparison`` still computes economic Top-K returns from idealised daily
    selections rather than carrying positions through direction-specific A-share
    buy/sell constraints. Transaction cost itself is turnover-proportional after
    Stage-4 PR #108; the remaining blocker is executable position state.
    """

    cfg = comparison_config or ComparisonConfig(
        label_column="forward_executable_return_5d",
        horizon_days=5,
        holdout_folds=1,
    )
    _validate_stage4_contract(experiment, final_holdout, cfg)
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
        config=cfg,
        regime_by_date=regime_by_date,
        progress=progress,
    )
    if experiment.declared_trial_count < comparison.n_trials:
        raise RuntimeError(
            "declared_trial_count undercounts the model arms actually evaluated: "
            f"declared={experiment.declared_trial_count}, measured_arms={comparison.n_trials}; "
            "final holdout remains consumed"
        )

    cumulative_trials = experiment_ledger.multiple_testing_trial_count(
        family=experiment.family
    )
    comparison = with_cumulative_trial_count(comparison, cumulative_trials)
    promotion = evaluate_nonlinear_promotion(comparison, config=promotion_config)
    holdout = _qualify_final_holdout(comparison, expected=final_holdout)
    governance = {
        "stage4Governed": True,
        "researchOnly": True,
        "experimentFingerprint": experiment.fingerprint,
        "eventHash": event["event_hash"],
        "holdoutKey": final_holdout.holdout_key,
        "holdoutSealHash": seal["seal_hash"],
        "labelContractHash": final_holdout.label_contract_hash,
        "executableLabelColumn": cfg.label_column,
        "familyAttempts": experiment_ledger.attempt_count(family=experiment.family),
        "cumulativeMultipleTestingTrials": cumulative_trials,
        "uniqueFamilyFingerprints": experiment_ledger.unique_fingerprint_count(
            family=experiment.family
        ),
        "economicBacktestCertified": False,
        "economicBacktestBlockers": [
            "model_comparison economic returns are not yet position-carrying/executable under direction-specific A-share buy/sell constraints",
        ],
    }
    return GovernedModelComparison(
        comparison=comparison,
        promotion=promotion,
        holdout=holdout,
        governance=governance,
    )


def _validate_stage4_contract(
    experiment: ExperimentSpec,
    holdout: FinalHoldoutSpec,
    config: ComparisonConfig,
) -> None:
    if experiment.family != holdout.family:
        raise ValueError("experiment and final holdout must belong to the same family")
    if experiment.dataset_hash != holdout.dataset_hash:
        raise ValueError("experiment and final holdout dataset_hash must match")
    if config.holdout_folds != 1:
        raise ValueError(
            "Stage-4 requires exactly one contiguous final holdout fold; "
            "multiple expanding holdout folds are not an untouched block"
        )

    expected_label = f"forward_executable_return_{int(config.horizon_days)}d"
    if str(config.label_column) != expected_label:
        raise ValueError(
            "Stage-4 requires the canonical executable next-session label for its horizon: "
            f"expected {expected_label!r}, got {config.label_column!r}. "
            "Build labels with build_executable_forward_returns on an explicit global market-session schedule."
        )
    if not str(holdout.label_contract_hash or "").strip():
        raise ValueError(
            "Stage-4 final holdout requires a non-empty label_contract_hash so the one-shot "
            "holdout seal is bound to the executable outcome clock"
        )

    train_start, train_end = map(pd.Timestamp, experiment.train_window)
    search_start, search_end = map(pd.Timestamp, experiment.search_window)
    holdout_start, holdout_end = map(pd.Timestamp, holdout.holdout_window)
    if train_start > train_end or search_start > search_end or holdout_start > holdout_end:
        raise ValueError("train/search/holdout windows must each be ordered start <= end")
    if train_end >= search_start:
        raise ValueError("train_window must end strictly before search_window starts")
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
    """Use final holdout only to veto the frozen champion, never to re-select."""
    champion = str(comparison.champion or "")
    reasons: list[str] = []
    holdout_windows = [w for w in comparison.fold_windows if w.get("role") == "holdout"]
    actual_start = min((str(w["validStart"]) for w in holdout_windows), default="")
    actual_end = max((str(w["validEnd"]) for w in holdout_windows), default="")
    if len(holdout_windows) != 1:
        reasons.append(
            f"Stage-4 final qualification requires exactly one holdout fold, found {len(holdout_windows)}"
        )
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
