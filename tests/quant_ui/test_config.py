from __future__ import annotations

from pathlib import Path

import pytest

from services.quant_api.config import (
    ApiSettings,
    default_settings,
    project_relative,
    safe_project_path,
)


def test_default_settings_respects_quantagent_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = tmp_path / "external-runtime"
    monkeypatch.setenv("QUANTAGENT_HOME", str(runtime))

    settings = default_settings()

    assert settings.runtime_root == runtime
    assert settings.cache_root == runtime / "cache" / "quant_ui"
    assert settings.jobs_root == runtime / "jobs" / "quant_ui"


def test_runtime_logical_paths_round_trip_outside_project(tmp_path: Path) -> None:
    project = tmp_path / "QuantAgent"
    runtime = tmp_path / "quant-runtime"
    settings = ApiSettings(
        project_root=project,
        runtime_root=runtime,
        cache_root=runtime / "cache" / "quant_ui",
        jobs_root=runtime / "jobs" / "quant_ui",
    ).ensure()
    artifact = runtime / "reports" / "run-1" / "metrics.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    logical = project_relative(settings, artifact)

    assert logical == "runtime/reports/run-1/metrics.json"
    assert safe_project_path(settings, logical) == artifact.resolve()


def test_safe_project_path_rejects_escape_from_project_and_runtime(tmp_path: Path) -> None:
    project = tmp_path / "QuantAgent"
    runtime = tmp_path / "quant-runtime"
    settings = ApiSettings(
        project_root=project,
        runtime_root=runtime,
        cache_root=runtime / "cache" / "quant_ui",
        jobs_root=runtime / "jobs" / "quant_ui",
    ).ensure()

    with pytest.raises(ValueError, match="outside"):
        safe_project_path(settings, tmp_path.parent / "forbidden.json")
