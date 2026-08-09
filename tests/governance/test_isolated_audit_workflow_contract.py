from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(".github/workflows/isolated-audit-gate.yml")
SCRIPT = Path("scripts/verify_pr_isolated_audit.py")


def test_audit_workflow_runs_from_trusted_base_not_pr_head() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pull_request_target:" in text
    assert "issue_comment:" in text
    assert "checks: write" in text
    assert "github.event.pull_request.base.sha" in text
    assert "github.event.pull_request.head.sha" not in text
    assert "isolated-audit-gate.yml" not in text.replace(
        "name: Isolated Audit Gate", ""
    ) or "checkout" in text.lower()


def test_audit_workflow_does_not_install_or_execute_pr_package() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pip install" not in text
    assert "PYTHONPATH: src" in text
    assert "verify_pr_isolated_audit.py" in text


def test_audit_script_publishes_dedicated_exact_head_check_name() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "head_sha" in text
    assert "AUDIT_CHECK_NAME" in text
    assert '"checks: write"' not in text  # permission belongs only in workflow YAML
    assert "check-runs" in text
