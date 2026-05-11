from quantagent.execution.audit import AuditLogger
from quantagent.execution.audit_replay import AuditReplay


def test_audit_replay_reads_append_only_jsonl(tmp_path):
    logger = AuditLogger(tmp_path, "audit.jsonl")
    path = logger.write("unit_event", {"value": 1})
    result = AuditReplay().replay(path)
    assert result.event_count == 1
    assert result.event_types["unit_event"] == 1

