"""Multi-agent governance: evidence gates, hard vetoes, audit integrity.

Each test names a way a governance system fails in practice: approvals without
evidence, a majority overriding the one agent that checked the data, a missing
check reading as a pass, a crashed veto-holder counting as consent, or a live
order slipping through a research workflow.
"""

from __future__ import annotations

import json

import pytest

from quantagent.governance import agents as roles
from quantagent.governance.audit import AuditLog
from quantagent.governance.envelopes import (
    APPROVE,
    BLOCK,
    NEEDS_EVIDENCE,
    REJECT,
    DecisionEnvelope,
    EnvelopeError,
)
from quantagent.governance.protocol import (
    OUTCOME_APPROVED,
    OUTCOME_BLOCKED,
    OUTCOME_NEEDS_EVIDENCE,
    OUTCOME_REJECTED,
    DecisionProtocol,
    LiveTradingAttempt,
)

ACTION = "promote the L1 champion book to the fresh-window read"


def _approval(agent: str, *, confidence: float = 0.9) -> DecisionEnvelope:
    return DecisionEnvelope(
        agent=agent, hypothesis_or_action=ACTION, verdict=APPROVE,
        input_artifact_hashes={"panel": "a" * 64},
        quantitative_evidence={"rows": 17_829_080, "match_rate": 1.0},
        method="recomputed from the U0 panel", confidence=confidence,
    )


def _all_approvals(*, involves_intraday: bool = False) -> list[DecisionEnvelope]:
    protocol = DecisionProtocol()
    return [_approval(a) for a in protocol.required_agents(involves_intraday=involves_intraday)]


class TestEnvelopeValidity:
    def test_unknown_verdict_is_refused(self):
        with pytest.raises(EnvelopeError, match="unknown verdict"):
            DecisionEnvelope(agent="risk", hypothesis_or_action=ACTION, verdict="LGTM")

    def test_confidence_must_be_a_probability(self):
        with pytest.raises(EnvelopeError, match="confidence"):
            DecisionEnvelope(agent="risk", hypothesis_or_action=ACTION,
                             verdict=APPROVE, confidence=7.0)

    def test_approve_without_evidence_is_structurally_invalid(self):
        envelope = DecisionEnvelope(
            agent="risk", hypothesis_or_action=ACTION, verdict=APPROVE
        )
        problems = envelope.validate()
        assert any("input_artifact_hashes" in p for p in problems)
        assert any("quantitative_evidence" in p for p in problems)

    def test_block_must_name_its_blockers(self):
        envelope = DecisionEnvelope(
            agent="risk", hypothesis_or_action=ACTION, verdict=BLOCK
        )
        assert any("hard_blockers" in p for p in envelope.validate())

    def test_valid_approval_passes(self):
        assert _approval("risk").is_valid


class TestHardVetoes:
    def test_data_quality_block_beats_every_approval(self):
        """Nine confident approvals must not outvote the agent that checked."""
        envelopes = [
            _approval(a, confidence=0.99)
            for a in roles.APPROVAL_SEQUENCE if a != roles.DATA_QUALITY.name
        ]
        envelopes.append(DecisionEnvelope(
            agent=roles.DATA_QUALITY.name, hypothesis_or_action=ACTION, verdict=BLOCK,
            hard_blockers=["tick semantics are UNKNOWN_SEMANTICS"],
            confidence=0.10,
        ))
        record = DecisionProtocol().decide(ACTION, envelopes, involves_intraday=True)
        assert record.outcome == OUTCOME_BLOCKED
        assert record.approved is False
        assert record.blockers[0]["agent"] == roles.DATA_QUALITY.name
        # High mean confidence must not have changed anything.
        assert record.mean_confidence > 0.8

    def test_risk_block_cannot_be_overridden(self):
        envelopes = [
            _approval(a) for a in roles.APPROVAL_SEQUENCE if a != roles.RISK.name
        ]
        envelopes.append(DecisionEnvelope(
            agent=roles.RISK.name, hypothesis_or_action=ACTION, verdict=BLOCK,
            hard_blockers=["worst drawdown 22.1% exceeds the 20% limit"],
        ))
        record = DecisionProtocol().decide(ACTION, envelopes, involves_intraday=True)
        assert record.outcome == OUTCOME_BLOCKED

    def test_compliance_block_stops_the_decision(self):
        envelopes = [
            _approval(a) for a in roles.APPROVAL_SEQUENCE if a != roles.COMPLIANCE.name
        ]
        envelopes.append(DecisionEnvelope(
            agent=roles.COMPLIANCE.name, hypothesis_or_action=ACTION, verdict=BLOCK,
            hard_blockers=["licensed raw vendor data staged inside the Git tree"],
        ))
        record = DecisionProtocol().decide(ACTION, envelopes, involves_intraday=True)
        assert record.outcome == OUTCOME_BLOCKED

    def test_advisory_agent_block_is_downgraded_to_a_rejection(self):
        """Only veto holders get to end a decision outright."""
        envelopes = _all_approvals()
        envelopes = [e for e in envelopes if e.agent != roles.STOCK_SELECTION.name]
        envelopes.append(DecisionEnvelope(
            agent=roles.STOCK_SELECTION.name, hypothesis_or_action=ACTION,
            verdict=BLOCK, hard_blockers=["I do not like this book"],
        ))
        record = DecisionProtocol().decide(ACTION, envelopes)
        assert record.outcome == OUTCOME_REJECTED
        assert record.blockers == []
        assert any("without veto authority" in n for n in record.notes)

    def test_veto_holders_are_the_declared_four_plus_domain_agents(self):
        holders = set(roles.veto_holders())
        assert {roles.DATA_QUALITY.name, roles.RISK.name,
                roles.COMPLIANCE.name, roles.ORCHESTRATOR.name} <= holders
        assert roles.STOCK_SELECTION.name not in holders


class TestEvidenceGates:
    def test_approval_without_evidence_is_downgraded_not_counted(self):
        envelopes = _all_approvals()
        envelopes = [e for e in envelopes if e.agent != roles.BACKTEST.name]
        envelopes.append(DecisionEnvelope(
            agent=roles.BACKTEST.name, hypothesis_or_action=ACTION, verdict=APPROVE,
            confidence=0.99,  # confident, but cites nothing
        ))
        record = DecisionProtocol().decide(ACTION, envelopes)
        assert record.outcome == OUTCOME_NEEDS_EVIDENCE
        assert record.downgraded_envelopes[0]["agent"] == roles.BACKTEST.name
        assert roles.BACKTEST.name not in record.approvals

    def test_missing_mandatory_agent_is_not_an_implicit_pass(self):
        envelopes = [e for e in _all_approvals() if e.agent != roles.RISK.name]
        record = DecisionProtocol().decide(ACTION, envelopes)
        assert record.outcome == OUTCOME_NEEDS_EVIDENCE
        assert roles.RISK.name in record.missing_mandatory
        assert any("absent check is not a passed check" in n for n in record.notes)

    def test_all_approvals_with_evidence_are_approved(self):
        record = DecisionProtocol().decide(ACTION, _all_approvals())
        assert record.outcome == OUTCOME_APPROVED
        assert record.approved is True

    def test_intraday_decision_requires_the_microstructure_agent(self):
        """A tick decision must not skip the agent that judges fidelity."""
        daily_set = _all_approvals(involves_intraday=False)
        record = DecisionProtocol().decide(ACTION, daily_set, involves_intraday=True)
        assert record.outcome == OUTCOME_NEEDS_EVIDENCE
        assert roles.MICROSTRUCTURE.name not in record.approvals

    def test_daily_decision_does_not_require_microstructure(self):
        record = DecisionProtocol().decide(ACTION, _all_approvals(), involves_intraday=False)
        assert record.outcome == OUTCOME_APPROVED
        assert roles.MICROSTRUCTURE.name not in record.consulted


class TestDisagreementAndLiveTrading:
    def test_disagreement_is_recorded_even_when_the_outcome_is_clear(self):
        envelopes = [e for e in _all_approvals() if e.agent != roles.CHALLENGER.name]
        envelopes.append(DecisionEnvelope(
            agent=roles.CHALLENGER.name, hypothesis_or_action=ACTION, verdict=REJECT,
            known_limitations=["could not reproduce the reported IC independently"],
        ))
        record = DecisionProtocol().decide(ACTION, envelopes)
        assert record.outcome == OUTCOME_REJECTED
        assert record.disagreements
        assert roles.CHALLENGER.name in record.disagreements[0]["rejecting"]

    @pytest.mark.parametrize("action", [
        "enable live trading for the L1 book",
        "submit order to broker for 600000.SH",
        "开启实盘交易",
    ])
    def test_live_trading_is_refused_before_any_agent_is_consulted(self, action):
        with pytest.raises(LiveTradingAttempt, match="paper and dry-run"):
            DecisionProtocol().decide(action, _all_approvals())

    def test_live_trading_refusal_is_audited(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        protocol = DecisionProtocol(log)
        with pytest.raises(LiveTradingAttempt):
            protocol.decide("enable live trading now", [])
        kinds = [e.kind for e in log.entries()]
        assert "LIVE_TRADING_REFUSED" in kinds


class TestAuditLog:
    def test_entries_chain_and_verify(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        for i in range(5):
            log.append(kind="ENVELOPE", actor=f"agent{i}", subject=ACTION,
                       payload={"i": i})
        assert len(log) == 5
        assert log.verify()["valid"] is True

    def test_editing_an_entry_breaks_verification(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        for i in range(3):
            log.append(kind="ENVELOPE", actor="a", subject=ACTION, payload={"i": i})

        lines = path.read_text(encoding="utf-8").splitlines()
        tampered = json.loads(lines[1])
        tampered["payload"] = {"i": 999}
        lines[1] = json.dumps(tampered, ensure_ascii=False)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = log.verify()
        assert result["valid"] is False
        assert "hash" in result["error"] or "chain" in result["error"]

    def test_deleting_an_entry_breaks_verification(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        for i in range(4):
            log.append(kind="ENVELOPE", actor="a", subject=ACTION, payload={"i": i})
        lines = path.read_text(encoding="utf-8").splitlines()
        del lines[1]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert log.verify()["valid"] is False

    def test_protocol_persists_every_envelope_and_the_decision(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        protocol = DecisionProtocol(log)
        envelopes = _all_approvals()
        protocol.decide(ACTION, envelopes)
        kinds = [e.kind for e in log.entries()]
        assert kinds.count("ENVELOPE") == len(envelopes)
        assert kinds.count("DECISION") == 1
        assert log.verify()["valid"] is True

    def test_log_lives_outside_git(self):
        """The audit trail must not be rewritable by a rebase."""
        from pathlib import Path
        gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
        assert "runtime/" in gitignore.read_text(encoding="utf-8")


class TestRoleDeclarations:
    def test_every_role_declares_scope_and_failure_behaviour(self):
        for role in roles.ALL_ROLES:
            assert role.responsibilities, f"{role.name} has no responsibilities"
            assert role.read_scope, f"{role.name} has no read scope"
            assert role.output_schema, f"{role.name} has no output schema"
            assert role.failure_behaviour in (roles.FAIL_CLOSED, roles.FAIL_OPEN)

    def test_veto_holders_fail_closed(self):
        """A crashed veto-holder must not be read as consent."""
        for role in roles.ALL_ROLES:
            if role.can_veto:
                assert role.failure_behaviour == roles.FAIL_CLOSED, role.name

    def test_data_acquisition_cannot_write_validation_thresholds(self):
        acquisition = roles.role_for(roles.DATA_ACQUISITION.name)
        assert not any("validation" in scope for scope in acquisition.write_scope)

    def test_execution_agent_is_paper_only(self):
        execution = roles.role_for(roles.EXECUTION.name)
        assert "paper_broker" in execution.allowed_tools
        assert not any("live" in tool.lower() for tool in execution.allowed_tools)
