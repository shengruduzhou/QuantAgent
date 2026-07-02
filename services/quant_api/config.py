from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

from quantagent.config.paths import quant_paths


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ApiSettings:
    project_root: Path = PROJECT_ROOT
    runtime_root: Path = PROJECT_ROOT / "runtime"
    cache_root: Path = PROJECT_ROOT / "runtime" / "cache" / "quant_ui"
    jobs_root: Path = PROJECT_ROOT / "runtime" / "jobs" / "quant_ui"
    index_ttl_seconds: int = 900
    max_table_rows: int = 1_000
    max_chart_points: int = 10_000

    def ensure(self) -> "ApiSettings":
        for path in (self.runtime_root, self.cache_root, self.jobs_root):
            path.mkdir(parents=True, exist_ok=True)
        return self


def default_settings() -> ApiSettings:
    paths = quant_paths(home=PROJECT_ROOT / "runtime")
    return ApiSettings(
        project_root=PROJECT_ROOT,
        runtime_root=paths.home,
        cache_root=paths.cache / "quant_ui",
        jobs_root=paths.home / "jobs" / "quant_ui",
    ).ensure()


def safe_project_path(settings: ApiSettings, value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = settings.project_root / candidate
    resolved = candidate.resolve()
    root = settings.project_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("path is outside the QuantAgent project")
    return resolved


def project_relative(settings: ApiSettings, path: str | Path) -> str:
    resolved = safe_project_path(settings, path)
    return resolved.relative_to(settings.project_root.resolve()).as_posix()


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{sha1(value.encode('utf-8')).hexdigest()[:16]}"
