from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from quantagent.cli import app
from quantagent.cli import governance as governance_cli
from quantagent.execution.live_model_trust_v2_execution_policy import (
    GOVERNED_EXECUTION_TRACE_ROLE,
    GOVERNED_REQUIRED_ARTIFACT_ROLES,
)


RUNNER = CliRunner()
HEAD = "a" * 40


def _registered_names() -> set[str]:
    return {command.name for command in app.registered_commands}


def _map_payload(*, root: str = "repo") -> dict[str, dict[str, str]]:
    return {
        role: {"root": root, "path": f"evidence/{role}.json"}
        for role in GOVERNED_REQUIRED_ARTIFACT_ROLES
    }


def _prepare_paths(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    monkeypatch.setattr(governance_cli, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(governance_cli, "_git_head", lambda: HEAD)
    evidence_map = tmp_path / "evidence_map.json"
    manifest = tmp_path / "configs" / "live_model_trust.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    return evidence_map, manifest


def test_governed_v2_issuer_command_is_registered() -> None:
    assert "issue-live-model-trust-v2" in _registered_names()


def test_issuer_help_exposes_no_policy_override_or_force_switch() -> None:
    result = RUNNER.invoke(app, ["issue-live-model-trust-v2", "--help"])
    assert result.exit_code == 0, result.stdout
    text = result.stdout.lower()
    for forbidden in (
        "--force",
        "--ignore-hash",
        "--max-pbo",
        "--min-dsr",
        "--max-spa",
        "--min-fresh-oos-days",
    ):
        assert forbidden not in text


def test_source_commit_must_equal_current_git_head(tmp_path: Path, monkeypatch) -> None:
    evidence_map, manifest = _prepare_paths(tmp_path, monkeypatch)
    evidence_map.write_text("{}", encoding="utf-8")
    result = RUNNER.invoke(
        app,
        [
            "issue-live-model-trust-v2",
            "--evidence-map",
            str(evidence_map),
            "--manifest",
            str(manifest),
            "--model-id",
            "model-v2",
            "--source-commit",
            "b" * 40,
        ],
    )
    assert result.exit_code != 0
    assert "source_commit must equal current git HEAD" in result.output


def test_evidence_map_cannot_introduce_an_untrusted_root(tmp_path: Path, monkeypatch) -> None:
    evidence_map, manifest = _prepare_paths(tmp_path, monkeypatch)
    evidence_map.write_text(json.dumps(_map_payload(root="external")), encoding="utf-8")
    result = RUNNER.invoke(
        app,
        [
            "issue-live-model-trust-v2",
            "--evidence-map",
            str(evidence_map),
            "--manifest",
            str(manifest),
            "--model-id",
            "model-v2",
        ],
    )
    assert result.exit_code != 0
    assert ".root must be one of" in result.output


def test_execution_trace_is_a_mandatory_governed_evidence_role(tmp_path: Path, monkeypatch) -> None:
    evidence_map, manifest = _prepare_paths(tmp_path, monkeypatch)
    payload = _map_payload()
    payload.pop(GOVERNED_EXECUTION_TRACE_ROLE)
    evidence_map.write_text(json.dumps(payload), encoding="utf-8")
    result = RUNNER.invoke(
        app,
        [
            "issue-live-model-trust-v2",
            "--evidence-map",
            str(evidence_map),
            "--manifest",
            str(manifest),
            "--model-id",
            "model-v2",
        ],
    )
    assert result.exit_code != 0
    assert f"evidence map missing object for role {GOVERNED_EXECUTION_TRACE_ROLE}" in result.output


def test_cli_only_binds_evidence_and_never_arms_live(tmp_path: Path, monkeypatch) -> None:
    evidence_map, manifest = _prepare_paths(tmp_path, monkeypatch)
    evidence_map.write_text(json.dumps(_map_payload()), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_issue(path, **kwargs):
        captured["path"] = Path(path)
        captured.update(kwargs)
        return SimpleNamespace(
            manifest_path=str(path),
            verification=SimpleNamespace(
                ok=True,
                evidence={
                    "provenance_assurance": "hash_bound_unsigned_v1",
                    "execution_timing_assurance": "trace_proven:signal_t_close_next_session_close_v1",
                },
            ),
        )

    monkeypatch.setattr(governance_cli, "issue_v2_certificate", fake_issue)
    result = RUNNER.invoke(
        app,
        [
            "issue-live-model-trust-v2",
            "--evidence-map",
            str(evidence_map),
            "--manifest",
            str(manifest),
            "--model-id",
            "model-v2",
        ],
    )
    assert result.exit_code == 0, result.output
    output = json.loads(result.stdout)
    assert output["verification_ok"] is True
    assert output["live_trading_armed_by_command"] is False
    assert output["source_commit"] == HEAD
    assert output["execution_timing_assurance"].startswith("trace_proven:")
    assert captured["source_commit"] == HEAD
    assert set(captured["artifact_locations"]) == set(GOVERNED_REQUIRED_ARTIFACT_ROLES)
