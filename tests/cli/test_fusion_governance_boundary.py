from __future__ import annotations

from types import SimpleNamespace

from quantagent.cli.fusion import _fusion_gate_payload


def _gate(*, eligible: bool) -> SimpleNamespace:
    return SimpleNamespace(
        eligible=eligible,
        as_dict=lambda: {
            "promotionEligible": eligible,
            "checks": [],
            "blockers": [] if eligible else ["research gate failed"],
        },
    )


def test_passing_factor_fusion_research_gate_cannot_grant_production() -> None:
    payload = _fusion_gate_payload(
        _gate(eligible=True),
        {"dsrProbability": 0.99, "spaPValue": 0.01, "preferred": "candidate-a"},
    )

    assert payload["researchPromotionEligible"] is True
    assert payload["promotionEligible"] is False
    assert payload["productionEligible"] is False
    assert payload["stage4Governed"] is False
    assert payload["researchOnly"] is True
    assert payload["productionBlockers"]
    assert any("FinalHoldoutLedger" in blocker for blocker in payload["productionBlockers"])
    assert any("strict position-carrying A-share simulator" in blocker for blocker in payload["productionBlockers"])


def test_blocked_factor_fusion_research_gate_remains_production_blocked() -> None:
    payload = _fusion_gate_payload(
        _gate(eligible=False),
        {"dsrProbability": None, "spaPValue": None, "preferred": None},
    )

    assert payload["researchPromotionEligible"] is False
    assert payload["promotionEligible"] is False
    assert payload["productionEligible"] is False
