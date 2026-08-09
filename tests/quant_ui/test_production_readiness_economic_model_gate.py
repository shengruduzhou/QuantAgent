from __future__ import annotations

from pathlib import Path

from quantagent.execution.live_model_trust import LiveModelTrustReport
from services.quant_api.services import production_readiness as readiness_module
from services.quant_api.services.production_readiness import ProductionReadinessService


def test_operator_model_card_does_not_show_evidence_only_v2_as_pass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    verified_evidence_only = LiveModelTrustReport(
        ok=True,
        status="governed_evidence_accepted",
        model_id="fixture-v2",
        trust_class="fresh_oos_evidence",
        reasons=(),
        evidence={
            "economic_live_eligible": False,
            "strict_backtest_metric_semantics": "strict_v8_nav_v2_initial_cash",
        },
        manifest_path=str(tmp_path / "configs" / "live_model_trust.json"),
    )
    monkeypatch.setattr(
        readiness_module,
        "evaluate_live_model_trust",
        lambda manifest: verified_evidence_only,
    )

    service = ProductionReadinessService(tmp_path, tmp_path / "runtime")
    card = service._model_trust()

    assert card.state == "BLOCKED"
    assert card.severity == "blocked"
    assert card.reasons == ("v2_economic_live_eligibility_not_proven",)
    assert card.evidence["evidenceVerificationOk"] is True
    assert card.evidence["economicLiveEligible"] is False
