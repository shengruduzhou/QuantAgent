"""Economic-live model gate kept separate from evidence-bundle verification.

A schema-v2 bundle may be internally consistent and hash-bound while still
lacking the independent assurances required to authorize economic trading
(e.g. trace-proven execution timing and protected provenance).  Downstream
order authorization must therefore require an explicit positive economic
eligibility fact instead of treating ``LiveModelTrustReport.ok`` alone as
sufficient.
"""

from __future__ import annotations

from quantagent.execution.live_model_trust import LiveModelTrustReport


ECONOMIC_LIVE_ELIGIBLE_KEY = "economic_live_eligible"
ECONOMIC_LIVE_NOT_PROVEN_REASON = "v2_economic_live_eligibility_not_proven"


def evaluate_economic_model_gate(
    report: LiveModelTrustReport,
) -> tuple[bool, tuple[str, ...]]:
    """Return the economic-live verdict without weakening evidence verification.

    ``report.ok`` means the certificate/evidence verifier found no internal
    inconsistency.  Economic authorization is stricter and requires a separate,
    explicit ``economic_live_eligible is True`` fact.  Missing/false/ambiguous
    evidence always fails closed.
    """
    if not report.ok:
        return False, tuple(report.reasons)
    if report.evidence.get(ECONOMIC_LIVE_ELIGIBLE_KEY) is not True:
        return False, (ECONOMIC_LIVE_NOT_PROVEN_REASON,)
    return True, ()


__all__ = [
    "ECONOMIC_LIVE_ELIGIBLE_KEY",
    "ECONOMIC_LIVE_NOT_PROVEN_REASON",
    "evaluate_economic_model_gate",
]
