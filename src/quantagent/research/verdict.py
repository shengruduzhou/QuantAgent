"""Separate a rejected research hypothesis from a broken pipeline.

A walk-forward run that completes and then fails its pre-registered
overfitting gate has produced a *result*: the candidate is not usable. That is
not the same event as a missing file, an OOM kill, or a bug, yet both used to
surface as ``exit code 1`` with a Python traceback, so an operator could not
tell "the research was rejected" from "the system broke" — and the run's own
evidence was discarded on the way out.

``ResearchRejection`` carries the verdict, the gate that produced it, the
measured values behind it and the artifacts that hold the evidence.
``RESEARCH_REJECTED_EXIT_CODE`` gives the process a distinct exit status so the
job layer can record a terminal *conclusion* rather than a failure.

The rejection never softens the gate. It only makes the refusal legible: what
was measured, which threshold it violated, and which artifact proves it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

# Distinct from 1 (engineering failure) and 2 (CLI usage error, Click's own).
RESEARCH_REJECTED_EXIT_CODE = 3

# A run whose configuration cannot answer its own question. Nothing was
# measured, so it is neither a rejection (which requires a tested hypothesis)
# nor a failure (which requires something to have broken). It is *blocked*, and
# it is knowable before any compute is spent.
CONFIGURATION_BLOCKED_EXIT_CODE = 4

VERDICT_FILENAME = "research_verdict.json"


@dataclass
class ResearchRejection(Exception):
    """A pre-registered research gate refused to promote a candidate."""

    code: str
    title: str
    reasons: tuple[str, ...]
    stage: str
    remediation: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)
    evidence_paths: tuple[str, ...] = ()
    output_dir: Path | None = None
    #: ``rejected`` = a gate tested the candidate and refused it.
    #: ``blocked``  = the run could not be executed as configured; nothing was tested.
    verdict: str = "rejected"

    def __post_init__(self) -> None:
        self.reasons = tuple(str(item) for item in self.reasons)
        self.evidence_paths = tuple(str(item) for item in self.evidence_paths)
        Exception.__init__(self, self.summary())

    def summary(self) -> str:
        joined = "; ".join(self.reasons) if self.reasons else self.title
        return f"{self.title}: {joined}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "quantagent.research_verdict.v1",
            "verdict": self.verdict,
            "code": self.code,
            "title": self.title,
            "stage": self.stage,
            "reasons": list(self.reasons),
            "remediation": self.remediation,
            "metrics": dict(self.metrics),
            "evidencePaths": list(self.evidence_paths),
            "decidedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def persist(self, output_dir: Path | str | None = None) -> Path | None:
        """Write the verdict next to the run's other evidence.

        Returns the path written, or ``None`` when no output directory is known
        — a rejection with nowhere to record itself is still reported on stdout,
        it is simply not filed.
        """
        target = Path(output_dir) if output_dir is not None else self.output_dir
        if target is None:
            return None
        target = Path(target)
        try:
            target.mkdir(parents=True, exist_ok=True)
            path = target / VERDICT_FILENAME
            path.write_text(
                json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            return None
        return path


def rejection_event(rejection: ResearchRejection, verdict_path: Path | None) -> str:
    """A single structured stdout line the job layer parses without a log scrape."""
    payload = rejection.to_dict()
    payload.update({
        "progress": 1.0,
        "message": rejection.summary(),
        "verdictPath": str(verdict_path) if verdict_path else None,
    })
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


#: One OOS trading day is consumed turning a NAV series into daily returns
#: (``nav.pct_change().dropna()``). Overfitting governance counts *return
#: observations*, while the splitter and the pre-flight count *trading days*, so
#: without this allowance a run that produces exactly ``selection + holdout``
#: days hands governance one observation too few and aborts after the whole
#: portfolio search has been paid for.
RETURN_DIFFERENCING_DAYS = 1


def required_oos_days(min_selection_days: int, min_holdout_days: int) -> int:
    """OOS trading days a nested selection + frozen holdout protocol needs."""
    return int(min_selection_days) + int(min_holdout_days) + RETURN_DIFFERENCING_DAYS


def reject_insufficient_oos(
    *,
    observed_days: int,
    min_selection_days: int,
    min_holdout_days: int,
    output_dir: Path | str | None = None,
    evidence_paths: Iterable[str] = (),
) -> ResearchRejection:
    required = required_oos_days(min_selection_days, min_holdout_days)
    return ResearchRejection(
        code="insufficient_oos_dates",
        title="样本外交易日不足，无法执行嵌套选择与冻结 holdout",
        reasons=(
            f"observed OOS trading days = {observed_days}, required = {required} "
            f"(selection {min_selection_days} + holdout {min_holdout_days})",
        ),
        stage="portfolio_selection",
        remediation=(
            "提高 nSplits（每折 20 个 OOS 交易日），或降低 selectionMinOosDays / "
            "selectionMinHoldoutDays。缩短 holdout 会降低结论强度，请在研究记录中说明。"
        ),
        metrics={
            "observedOosDays": int(observed_days),
            "requiredOosDays": int(required),
            "minSelectionDays": int(min_selection_days),
            "minHoldoutDays": int(min_holdout_days),
        },
        evidence_paths=tuple(evidence_paths),
        output_dir=Path(output_dir) if output_dir is not None else None,
    )


def block_infeasible_oos_budget(
    *,
    achievable_oos_days: int,
    requested_splits: int,
    achievable_splits: int,
    valid_size_days: int,
    min_selection_days: int,
    min_holdout_days: int,
    trading_days_available: int,
    trading_days_required: int,
    output_dir: Path | str | None = None,
) -> ResearchRejection:
    """Refuse a run whose fold budget can never satisfy its own protocol.

    Raised *before* training, from numbers that are pure arithmetic over the
    panel's date span. The old behaviour was to discover the same shortfall in
    the portfolio stage, after the whole walk-forward had been paid for, and to
    report it as a research rejection — which read as "the hypothesis failed"
    when in fact no hypothesis had been tested.
    """
    required = required_oos_days(min_selection_days, min_holdout_days)
    needed_splits = -(-required // max(1, valid_size_days))  # ceil
    panel_is_the_constraint = achievable_splits < requested_splits
    if panel_is_the_constraint:
        shortfall = (
            f"requested {requested_splits} folds x {valid_size_days} days = "
            f"{requested_splits * valid_size_days} OOS days, but the data supports only "
            f"{achievable_splits} folds = {achievable_oos_days} OOS days"
        )
    else:
        shortfall = (
            f"requested {requested_splits} folds x {valid_size_days} days = "
            f"{achievable_oos_days} OOS days"
        )
    reasons = [
        shortfall,
        f"the selection + holdout protocol requires {required} OOS days "
        f"(selection {min_selection_days} + holdout {min_holdout_days})",
    ]
    return ResearchRejection(
        code="infeasible_oos_budget",
        title="配置无法产生协议所需的样本外交易日，运行已在训练前中止",
        reasons=tuple(reasons),
        stage="preflight",
        verdict="blocked",
        remediation=(
            (
                f"数据跨度不足：现有 {trading_days_available} 个可用交易日，"
                f"至少需要 {trading_days_required} 个。请扩大数据范围或缩短标签期限；"
                "单纯提高 nSplits 不会有帮助。"
            )
            if panel_is_the_constraint
            else (
                f"把 nSplits 提高到至少 {needed_splits} 折，或降低 selectionMinOosDays / "
                f"selectionMinHoldoutDays。数据本身够用（{trading_days_available} 个可用交易日，"
                f"该折数需要 {trading_days_required} 个）。"
                "缩短 holdout 会降低结论强度，请在研究记录中说明。"
            )
        ),
        metrics={
            "requestedSplits": int(requested_splits),
            "achievableSplits": int(achievable_splits),
            "validSizeDays": int(valid_size_days),
            "achievableOosDays": int(achievable_oos_days),
            "requiredOosDays": int(required),
            "minimumSplits": int(needed_splits),
            "minSelectionDays": int(min_selection_days),
            "minHoldoutDays": int(min_holdout_days),
            "tradingDaysAvailable": int(trading_days_available),
            "tradingDaysRequired": int(trading_days_required),
        },
        output_dir=Path(output_dir) if output_dir is not None else None,
    )


__all__ = [
    "CONFIGURATION_BLOCKED_EXIT_CODE",
    "RESEARCH_REJECTED_EXIT_CODE",
    "ResearchRejection",
    "VERDICT_FILENAME",
    "block_infeasible_oos_budget",
    "reject_insufficient_oos",
    "rejection_event",
    "required_oos_days",
]
