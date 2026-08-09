"""Economic-live model gate kept separate from evidence-bundle verification.

A schema-v2 bundle may be internally consistent and hash-bound while still
lacking the independent assurances required to authorize economic trading.
Downstream order authorization therefore requires an explicit positive economic
eligibility fact **and** trace-proven execution timing. Product Arming cannot
substitute for either requirement.
"""

from __future__ import annotations

from quantagent.execution.live_model_trust import LiveModelTrustReport
from quantagent.execution.live_model_trust_v2_execution_policy import (
    TRACE_PROVEN_EXECUTION_ASSURANCE,
)


ECONOMIC_LIVE_ELIGIBLE_KEY = "economic_live_eligible"
ECONOMIC_LIVE_NOT_PROVEN_REASON = "v2_economic_live_eligibility_not_proven"
EXECUTION_TIMING_NOT_PROVEN_REASON = "execution_timing_trace_not_proven"


def evaluate_economic_model_gate(
    report: LiveModelTrustReport,
) -> tuple[bool, tuple[str, ...]]:
    """Return the economic-live verdict without weakening evidence verification.

    ``report.ok`` only means the presented certificate/evidence is internally
    consistent under its own schema. Economic authorization is stricter:

    1. evidence verification must pass;
    2. an independent promotion process must set literal
       ``economic_live_eligible is True``;
    3. execution timing assurance must be the verifier-derived trace-proven
       contract, not a narrative flag.
    """
    if not report.ok:
        return False, tuple(report.reasons)
    if report.evidence.get(ECONOMIC_LIVE_ELIGIBLE_KEY) is not True:
        return False, (ECONOMIC_LIVE_NOT_PROVEN_REASON,)
    if report.evidence.get("execution_timing_assurance") != TRACE_PROVEN_EXECUTION_ASSURANCE:
        return False, (EXECUTION_TIMING_NOT_PROVEN_REASON,)
    return True, ()


__all__ = [
    "ECONOMIC_LIVE_ELIGIBLE_KEY",
    "ECONOMIC_LIVE_NOT_PROVEN_REASON",
    "EXECUTION_TIMING_NOT_PROVEN_REASON",
    "evaluate_economic_model_gate",
]
