from __future__ import annotations

from quantagent.execution.economic_model_gate import (
    ECONOMIC_LIVE_NOT_PROVEN_REASON,
    evaluate_economic_model_gate,
)
from quantagent.execution.live_model_trust import LiveModelTrustReport


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
            evidence={"economic_live_eligible": True},
            reasons=("artifact_digest_mismatch",),
        )
    )
    assert allowed is False
    assert reasons == ("artifact_digest_mismatch",)


def test_only_verified_and_explicitly_eligible_model_passes_economic_gate() -> None:
    allowed, reasons = evaluate_economic_model_gate(
        _report(ok=True, evidence={"economic_live_eligible": True})
    )
    assert allowed is True
    assert reasons == ()
