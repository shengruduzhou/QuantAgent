from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from services.quant_api import __main__ as launcher


def test_package_import_does_not_eagerly_create_the_fastapi_app() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import services.quant_api; "
            "assert 'services.quant_api.app' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_launcher_passes_explicit_runtime_and_server_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_run(app: str, **kwargs: Any) -> None:
        calls.append((app, kwargs))

    monkeypatch.setenv("QUANTAGENT_HOME", "/previous/runtime")
    monkeypatch.setattr(launcher.uvicorn, "run", fake_run)

    runtime = tmp_path / "tickflow-runtime"
    launcher.main([
        "--runtime", str(runtime),
        "--host", "0.0.0.0",
        "--port", "9010",
        "--reload",
        "--log-level", "debug",
    ])

    assert calls == [(
        "services.quant_api.app:app",
        {
            "host": "0.0.0.0",
            "port": 9010,
            "reload": True,
            "log_level": "debug",
        },
    )]
    assert Path(launcher.os.environ["QUANTAGENT_HOME"]) == runtime.resolve()


def test_launcher_rejects_invalid_port() -> None:
    with pytest.raises(SystemExit):
        launcher.build_parser().parse_args(["--port", "70000"])
