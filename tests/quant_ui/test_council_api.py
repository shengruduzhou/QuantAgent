"""Contract for the ATLAS decision council.

These tests encode the three properties that make the council worth having:
role-scoped verdicts, `unknown` never counting as a pass, and overrides that
are recorded rather than silently applied.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from services.quant_api.app import create_app
from services.quant_api.services.container import ServiceContainer
from services.quant_api.services.council import (
    COUNCIL_PROTOCOL_VERSION,
    COUNCIL_ROLES,
    CouncilService,
)


def request(app, method: str, url: str, **kwargs):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def _run_id(app) -> str:
    payload = request(app, "GET", "/api/fusion/runs").json()
    return payload["data"][0]["id"]


def _findings(review: dict) -> dict[str, dict]:
    return {item["roleId"]: item for item in review["findings"]}


def _override_payload(
    review: dict,
    role_id: str,
    *,
    verdict: str = "warn",
    reason: str = "完整复核后记录该人工判断，不改变缺失证据的事实。",
    author: str = "researcher",
) -> dict:
    finding = _findings(review)[role_id]
    return {
        "subjectType": review["subject"]["type"],
        "subjectId": review["subject"]["id"],
        "subjectContentHash": review["subject"]["contentHash"],
        "candidateId": review["subject"].get("candidateId"),
        "roleId": role_id,
        "findingHash": finding["findingHash"],
        "originalVerdict": finding["verdict"],
        "verdict": verdict,
        "reason": reason,
        "author": author,
        "protocolVersion": review["protocolVersion"],
        "policyFingerprint": review["policyFingerprint"],
    }


# --------------------------------------------------------------------------- #
# Roster                                                                      #
# --------------------------------------------------------------------------- #


def test_roster_declares_every_role_with_a_veto_scope(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    payload = request(app, "GET", "/api/council/roster").json()
    roles = payload["data"]["roles"]
    assert len(roles) == 11
    assert len(roles) == len(COUNCIL_ROLES)
    for role in roles:
        assert role["vetoScope"], f"{role['id']} must declare what it can block"
        assert role["domain"]
    assert payload["data"]["protocolVersion"] == COUNCIL_PROTOCOL_VERSION
    assert len(payload["data"]["policyFingerprint"]) == 64
    assert [role["id"] for role in roles] == [
        "data_acquisition", "data_quality", "microstructure",
        "factor_integrity", "model_validation", "fusion_search",
        "portfolio_risk", "execution_realism", "challenger",
        "compliance", "governance",
    ]


def test_roster_exposes_the_promotion_thresholds(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    thresholds = request(app, "GET", "/api/council/roster").json()["data"]["thresholds"]
    assert thresholds["maxPbo"] == 0.25
    assert thresholds["minDeflatedSharpe"] == 0.95
    assert thresholds["maxSpaPValue"] == 0.05
    assert thresholds["minObservations"] == 60


# --------------------------------------------------------------------------- #
# Review                                                                      #
# --------------------------------------------------------------------------- #


def test_every_role_reviews_the_fusion_run(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    review = request(app, "GET", f"/api/council/review/fusion/{_run_id(app)}").json()["data"]
    assert set(_findings(review)) == {role["id"] for role in COUNCIL_ROLES}
    for finding in review["findings"]:
        assert finding["verdict"] in {"pass", "warn", "blocked", "unknown"}
        assert finding["evidence"], "a verdict must name the evidence it used"
        assert finding["headline"]


def test_fixture_run_is_promotable_with_the_expected_evidence(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    review = request(app, "GET", f"/api/council/review/fusion/{_run_id(app)}").json()["data"]
    findings = _findings(review)
    assert findings["data_quality"]["verdict"] == "pass"
    assert findings["fusion_search"]["evidence"]["nTrials"] == 4
    assert findings["fusion_search"]["evidence"]["pbo"] == 0.18
    assert review["decision"]["state"] in {"PROMOTABLE", "PROMOTABLE_WITH_WARNINGS"}


def test_control_candidate_is_flagged_by_governance(quant_ui_settings) -> None:
    """A control winning is a real finding, not a promotable strategy."""
    app = create_app(quant_ui_settings)
    review = request(
        app, "GET", f"/api/council/review/fusion/{_run_id(app)}?candidate=equal"
    ).json()["data"]
    governance = _findings(review)["governance"]
    assert governance["verdict"] == "blocked"
    assert "对照" in governance["headline"]
    assert review["decision"]["eligibleForHumanGate"] is False


def test_a_blend_that_loses_to_its_best_single_factor_is_blocked(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_dir = quant_ui_settings.runtime_root / "reports" / "fusion" / "fixture_search"
    candidates = json.loads((run_dir / "fusion_candidates.json").read_text(encoding="utf-8"))
    candidates.append({
        **candidates[0],
        "id": "single_alpha001",
        "label": "单因子 alpha001",
        "scheme": "single_factor",
        "isControl": True,
        "onFrontier": False,
        "metrics": {**candidates[0]["metrics"], "excessReturn": 0.30},
    })
    (run_dir / "fusion_candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False), encoding="utf-8"
    )
    review = request(app, "GET", f"/api/council/review/fusion/{_run_id(app)}").json()["data"]
    factor = _findings(review)["factor_integrity"]
    assert factor["verdict"] == "blocked"
    assert factor["evidence"]["bestSingleFactorExcessReturn"] == 0.30
    assert review["decision"]["state"] == "BLOCKED"
    assert "factor_integrity" in review["decision"]["blockedRoles"]


def test_missing_pbo_is_reported_as_unknown_not_a_pass(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_dir = quant_ui_settings.runtime_root / "reports" / "fusion" / "fixture_search"
    summary = json.loads((run_dir / "fusion_summary.json").read_text(encoding="utf-8"))
    summary["pbo"] = None
    (run_dir / "fusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    review = request(app, "GET", f"/api/council/review/fusion/{_run_id(app)}").json()["data"]
    search = _findings(review)["fusion_search"]
    assert search["verdict"] == "unknown"
    assert search["verdict"] != "pass"


def test_missing_trial_count_yields_unknown_and_blocks_a_clean_decision(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_dir = quant_ui_settings.runtime_root / "reports" / "fusion" / "fixture_search"
    summary = json.loads((run_dir / "fusion_summary.json").read_text(encoding="utf-8"))
    summary.pop("nTrials")
    (run_dir / "fusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    review = request(app, "GET", f"/api/council/review/fusion/{_run_id(app)}").json()["data"]
    assert _findings(review)["fusion_search"]["verdict"] == "unknown"
    assert review["decision"]["state"] == "INSUFFICIENT_EVIDENCE"


def test_missing_promotion_gate_never_receives_compliance_clearance(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_dir = quant_ui_settings.runtime_root / "reports" / "fusion" / "fixture_search"
    (run_dir / "promotion_gate.json").unlink()
    review = request(app, "GET", f"/api/council/review/fusion/{_run_id(app)}").json()["data"]
    assert _findings(review)["compliance"]["verdict"] == "unknown"
    assert _findings(review)["governance"]["verdict"] == "unknown"
    assert review["decision"]["state"] == "INSUFFICIENT_EVIDENCE"


def test_dsr_below_company_bar_is_blocked(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_dir = quant_ui_settings.runtime_root / "reports" / "fusion" / "fixture_search"
    gate = json.loads((run_dir / "promotion_gate.json").read_text(encoding="utf-8"))
    gate["statisticalEvidence"]["dsrProbability"] = 0.94
    (run_dir / "promotion_gate.json").write_text(json.dumps(gate), encoding="utf-8")
    review = request(app, "GET", f"/api/council/review/fusion/{_run_id(app)}").json()["data"]
    assert _findings(review)["fusion_search"]["verdict"] == "blocked"
    assert review["decision"]["state"] == "BLOCKED"


def test_overlapping_train_and_test_windows_are_blocked(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_dir = quant_ui_settings.runtime_root / "reports" / "fusion" / "fixture_search"
    summary = json.loads((run_dir / "fusion_summary.json").read_text(encoding="utf-8"))
    summary["foldWindows"] = [
        {
            "foldIndex": "0",
            "trainStart": "2024-01-02",
            "trainEnd": "2025-12-31",
            "testStart": "2025-07-10",
            "testEnd": "2025-12-31",
        },
        summary["foldWindows"][0],
        summary["foldWindows"][0],
    ]
    (run_dir / "fusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    review = request(app, "GET", f"/api/council/review/fusion/{_run_id(app)}").json()["data"]
    validation = _findings(review)["model_validation"]
    assert validation["verdict"] == "blocked"
    assert validation["evidence"]["overlappingFolds"]


def test_zero_cost_assumption_is_blocked_by_execution_realism(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_dir = quant_ui_settings.runtime_root / "reports" / "fusion" / "fixture_search"
    summary = json.loads((run_dir / "fusion_summary.json").read_text(encoding="utf-8"))
    summary["transactionCostBps"] = 0.0
    (run_dir / "fusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    review = request(app, "GET", f"/api/council/review/fusion/{_run_id(app)}").json()["data"]
    assert _findings(review)["execution_realism"]["verdict"] == "blocked"


def test_review_404s_on_an_unknown_run(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    assert request(app, "GET", "/api/council/review/fusion/nope").status_code == 404


# --------------------------------------------------------------------------- #
# Overrides                                                                   #
# --------------------------------------------------------------------------- #


def test_override_requires_a_stated_reason(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    review = request(
        app, "GET", f"/api/council/review/fusion/{_run_id(app)}"
    ).json()["data"]
    response = request(
        app,
        "POST",
        "/api/council/overrides",
        json=_override_payload(review, "portfolio_risk", verdict="pass", reason="ok"),
    )
    assert response.status_code == 422


def test_override_rejects_an_unknown_role(quant_ui_settings) -> None:
    container = ServiceContainer.create(quant_ui_settings)
    try:
        container.council.record_override(
            subject_type="fusion_run", subject_id="x", subject_content_hash="a" * 64,
            candidate_id="candidate", role_id="not_a_role", finding_hash="b" * 64,
            original_verdict="blocked", verdict="pass",
            reason="a sufficiently long reason", author="me",
            protocol_version=COUNCIL_PROTOCOL_VERSION,
            policy_fingerprint=container.council.policy_fingerprint,
        )
    except ValueError as exc:
        assert "unknown council role" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected an unknown role to be rejected")


def test_override_replaces_the_verdict_and_is_recorded(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_id = _run_id(app)
    run_dir = quant_ui_settings.runtime_root / "reports" / "fusion" / "fixture_search"
    summary = json.loads((run_dir / "fusion_summary.json").read_text(encoding="utf-8"))
    summary["transactionCostBps"] = 0.0
    (run_dir / "fusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    before = request(app, "GET", f"/api/council/review/fusion/{run_id}").json()["data"]
    assert _findings(before)["execution_realism"]["verdict"] == "blocked"

    created = request(
        app,
        "POST",
        "/api/council/overrides",
        json=_override_payload(
            before,
            "execution_realism",
            verdict="warn",
            reason="成本模型将在下游 A 股回测中重新施加，此处只做因子排序。",
            author="研究员甲",
        ),
    )
    assert created.status_code == 200

    after = request(app, "GET", f"/api/council/review/fusion/{run_id}").json()["data"]
    finding = _findings(after)["execution_realism"]
    # The original verdict is preserved next to the override, never erased.
    assert finding["verdict"] == "blocked"
    assert finding["override"]["verdict"] == "warn"
    assert finding["override"]["replacedVerdict"] == "blocked"
    assert finding["override"]["author"] == "研究员甲"
    assert "execution_realism" in after["decision"]["overriddenRoles"]
    assert "execution_realism" not in after["decision"]["blockedRoles"]


def test_override_log_is_append_only(quant_ui_settings) -> None:
    container = ServiceContainer.create(quant_ui_settings)
    council: CouncilService = container.council
    for index in range(3):
        council.record_override(
            subject_type="fusion_run", subject_id="run-x", subject_content_hash="a" * 64,
            candidate_id="candidate", role_id="portfolio_risk", finding_hash="b" * 64,
            original_verdict="pass", verdict="warn",
            reason=f"第 {index} 次复核，回撤在研究可接受范围内。", author="researcher",
            protocol_version=COUNCIL_PROTOCOL_VERSION,
            policy_fingerprint=council.policy_fingerprint,
        )
    records = council.overrides(subject_type="fusion_run", subject_id="run-x")
    assert len(records) == 3
    assert [record["reason"][:3] for record in records] == ["第 0", "第 1", "第 2"]


def test_overrides_are_scoped_to_their_subject(quant_ui_settings) -> None:
    container = ServiceContainer.create(quant_ui_settings)
    council: CouncilService = container.council
    council.record_override(
        subject_type="fusion_run", subject_id="run-a", subject_content_hash="a" * 64,
        candidate_id="candidate", role_id="governance", finding_hash="b" * 64,
        original_verdict="blocked", verdict="pass",
        reason="研究用途，已确认无实盘意图。", author="researcher",
        protocol_version=COUNCIL_PROTOCOL_VERSION,
        policy_fingerprint=council.policy_fingerprint,
    )
    assert council.overrides(subject_type="fusion_run", subject_id="run-b") == []
    assert len(council.overrides(subject_type="fusion_run", subject_id="run-a")) == 1


def test_a_single_factor_subject_is_not_faulted_for_not_beating_itself(quant_ui_settings) -> None:
    """Reviewing a single-factor candidate must not compare it against itself."""
    app = create_app(quant_ui_settings)
    review = request(
        app,
        "GET",
        f"/api/council/review/fusion/{_run_id(app)}?candidate=single_alpha002",
    ).json()["data"]
    factor = _findings(review)["factor_integrity"]
    assert factor["verdict"] == "blocked"
    assert factor["evidence"]["subjectIsSingleFactor"] is True
    assert "不构成融合" in factor["headline"]


def test_unknown_cannot_be_overridden_into_clearance(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_id = _run_id(app)
    run_dir = quant_ui_settings.runtime_root / "reports" / "fusion" / "fixture_search"
    summary = json.loads((run_dir / "fusion_summary.json").read_text(encoding="utf-8"))
    summary.pop("nTrials")
    (run_dir / "fusion_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    review = request(app, "GET", f"/api/council/review/fusion/{run_id}").json()["data"]

    assert _findings(review)["fusion_search"]["verdict"] == "unknown"
    response = request(
        app,
        "POST",
        "/api/council/overrides",
        json=_override_payload(review, "fusion_search", verdict="pass"),
    )
    assert response.status_code == 422
    assert "missing evidence" in response.text


def test_override_is_scoped_to_candidate_and_current_artifact_hash(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_id = _run_id(app)
    run_dir = quant_ui_settings.runtime_root / "reports" / "fusion" / "fixture_search"
    summary_path = run_dir / "fusion_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["transactionCostBps"] = 0.0
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    candidate_a = request(
        app, "GET", f"/api/council/review/fusion/{run_id}?candidate=ic_weighted"
    ).json()["data"]
    created = request(
        app,
        "POST",
        "/api/council/overrides",
        json=_override_payload(candidate_a, "execution_realism", verdict="warn"),
    )
    assert created.status_code == 200

    candidate_b = request(
        app, "GET", f"/api/council/review/fusion/{run_id}?candidate=equal"
    ).json()["data"]
    assert "override" not in _findings(candidate_b)["execution_realism"]
    assert any(item["scopeStatus"] == "stale_candidate" for item in candidate_b["overrides"])

    summary["transactionCostBps"] = 0.5
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    changed = request(
        app, "GET", f"/api/council/review/fusion/{run_id}?candidate=ic_weighted"
    ).json()["data"]
    assert "override" not in _findings(changed)["execution_realism"]
    assert any(item["scopeStatus"] == "stale_subject" for item in changed["overrides"])


def test_cio_never_passes_when_an_upstream_role_is_unknown(quant_ui_settings) -> None:
    app = create_app(quant_ui_settings)
    run_id = _run_id(app)
    run_dir = quant_ui_settings.runtime_root / "reports" / "fusion" / "fixture_search"
    summary_path = run_dir / "fusion_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("nTrials")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    review = request(app, "GET", f"/api/council/review/fusion/{run_id}").json()["data"]
    assert _findings(review)["fusion_search"]["verdict"] == "unknown"
    assert _findings(review)["governance"]["verdict"] == "unknown"
    assert review["decision"]["eligibleForHumanGate"] is False
    assert review["decision"]["liveEligible"] is False
