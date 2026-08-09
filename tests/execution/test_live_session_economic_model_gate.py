from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from quantagent.execution import live_session as live_session_module
from quantagent.execution.live_model_trust import LiveModelTrustReport
from quantagent.execution.live_session import LiveTradingSession


class _HealthyGateway:
    def preflight(self) -> dict[str, object]:
        return {"ok": True}

    def health(self) -> dict[str, object]:
        return {"ok": True}


def test_armed_product_still_cannot_trade_on_evidence_only_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    verified_evidence_only = LiveModelTrustReport(
        ok=True,
        status="governed_evidence_accepted",
        model_id="fixture-v2",
        trust_class="fresh_oos_evidence",
        reasons=(),
        evidence={"economic_live_eligible": False},
        manifest_path=str(tmp_path / "live_model_trust.json"),
    )
    monkeypatch.setattr(
        live_session_module,
        "evaluate_live_model_trust",
        lambda manifest: verified_evidence_only,
    )
    monkeypatch.setattr(
        live_session_module,
        "describe_policy",
        lambda: {"liveTradingAvailable": True},
    )

    risk_gate = SimpleNamespace(
        kill_switch=SimpleNamespace(triggered=False, reasons=()),
    )
    session = LiveTradingSession(
        _HealthyGateway(),
        risk_gate=risk_gate,  # type: ignore[arg-type]
        model_trust_manifest=tmp_path / "live_model_trust.json",
    )

    readiness = session.readiness()
    assert readiness.query_only_ready is True
    assert readiness.product_policy_armed is True
    assert readiness.model_trust_ok is False
    assert readiness.economic_submit_allowed is False
    assert "model:v2_economic_live_eligibility_not_proven" in readiness.reasons
