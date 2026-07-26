"""Repository ``.env`` loading for data-acquisition entrypoints.

Background jobs launched by the scheduler or the workstation JobRunner do not
inherit an interactive shell, so credentials that live in the repository ``.env``
were previously absent and a full-universe backfill died with a bare
``KeyError: 'TICKFLOW_API_KEY'`` after burning its whole time budget. Every
acquisition entrypoint calls :func:`load_repo_env` first.

Existing environment variables always win: an operator export is never
overwritten by the file, and nothing here is ever logged.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LOADED = False


def repo_root() -> Path:
    return _REPO_ROOT


def load_repo_env(path: Path | None = None, force: bool = False) -> list[str]:
    """Load ``.env`` into ``os.environ`` without overriding existing values.

    Returns the list of variable NAMES that were injected (never the values, so
    a caller can log what was picked up without leaking a credential).
    """
    global _LOADED
    if _LOADED and not force and path is None:
        return []
    env_path = path or (_REPO_ROOT / ".env")
    injected: list[str] = []
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip("'\"")
            injected.append(key)
    if path is None:
        _LOADED = True
    return injected


def require(name: str, hint: str = "") -> str:
    """Fetch a required credential, failing with an actionable message."""
    load_repo_env()
    value = os.environ.get(name)
    if not value:
        suffix = f" {hint}" if hint else ""
        raise RuntimeError(f"required environment variable {name} is not set.{suffix}")
    return value
