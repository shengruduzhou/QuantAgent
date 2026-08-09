from __future__ import annotations

from quantagent.execution.economic_model_gate import (
    ECONOMIC_LIVE_NOT_PROVEN_REASON,
    EXECUTION_TIMING_NOT_PROVEN_REASON,
    evaluate_economic_model_gate,
)
from quantagent.execution.live_model_trust import LiveModelTrustReport
from quantagent.execution.live_model_trust_v2_execution_policy import (
    TRACE_PROVEN_EXECUTION_ASSURANCE,
)


def _report(*, ok: bool, evidence: dict[str, object], reasons: tuple[str, ...] = ()) -> LiveModelTrustReport:
    return LiveModelTrustReport(
        ok=ok,
        status="production_accepted",
        model_id="fixture-v2",
        trust_class="fresh_oos",
        reasons=reasons,
        evidence=evidence,
        manifest_path="fixture.json",
    )


def test_verified_v2_evidence_is_not_automatically_economic_live_eligible() -> None:
    allowed, reasons = evaluate_economic_model_gate(
        _report(
            ok=True,
            evidence={
                "provenance_assurance": "hash_bound_unsigned_v1",
                "execution_timing_assurance": "not_certified_by_model_trust_v2",
            },
        )
    )
    assert allowed is False
    assert reasons == (ECONOMIC_LIVE_NOT_PROVEN_REASON,)


def test_explicit_false_or_ambiguous_economic_eligibility_fails_closed() -> None:
    for value in (False, None, 1, "true"):
        allowed, reasons = evaluate_economic_model_gate(
            _report(ok=True, evidence={"economic_live_eligible": value})
        )
        assert allowed is False
        assert reasons == (ECONOMIC_LIVE_NOT_PROVEN_REASON,)


def test_explicit_true_still_requires_verified_evidence() -> None:
    allowed, reasons = evaluate_economic_model_gate(
        _report(
            ok=False,
            evidence={
                "economic_live_eligible": True,
                "execution_timing_assurance": TRACE_PROVEN_EXECUTION_ASSURANCE,
            },
            reasons=("artifact_digest_mismatch",),
        )
    )
    assert allowed is False
    assert reasons == ("artifact_digest_mismatch",)


def test_explicit_true_without_trace_proven_timing_still_fails_closed() -> None:
    for assurance in (None, "not_certified_by_model_trust_v2", "trace_proven:true"):
        allowed, reasons = evaluate_economic_model_gate(
            _report(
                ok=True,
                evidence={
                    "economic_live_eligible": True,
                    "execution_timing_assurance": assurance,
                },
            )
        )
        assert allowed is False
        assert reasons == (EXECUTION_TIMING_NOT_PROVEN_REASON,)


def test_only_verified_eligible_and_trace_proven_model_passes_economic_gate() -> None:
    allowed, reasons = evaluate_economic_model_gate(
        _report(
            ok=True,
            evidence={
                "economic_live_eligible": True,
                "execution_timing_assurance": TRACE_PROVEN_EXECUTION_ASSURANCE,
            },
        )
    )
    assert allowed is True
    assert reasons == ()
