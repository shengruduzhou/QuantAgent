"""Round 31 F8: importing the API module must not touch the runtime.

``services/quant_api/app.py`` used to run ``app = create_app()`` at import, so
every test that imported ``create_app`` built a container on the default
runtime: it created ``jobs/``, ``paper_orders/writer.lock`` and rewrote the job
store of whatever ``QUANTAGENT_HOME`` pointed at (the operator's real runtime).
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]


def _run(code: str, home: Path) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "QUANTAGENT_HOME": str(home), "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{REPO}"}
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


def test_importing_create_app_has_no_runtime_side_effects(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    result = _run("from services.quant_api.app import create_app; print('ok')", home)
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in home.iterdir()) == [], "import must not create runtime state"


def test_module_level_app_is_still_resolvable_for_uvicorn(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    result = _run(
        "import services.quant_api.app as m; from fastapi import FastAPI; "
        "assert isinstance(m.app, FastAPI); assert m.app is m.app; print('ok')",
        home,
    )
    assert result.returncode == 0, result.stderr
