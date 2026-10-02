from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import os
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
    #: Optional symbol/industry table for the paper venue's industry limit
    #: (env ``QUANTAGENT_PAPER_SECTOR_MAP``). Without it a paper BUY is refused
    #: ``industry_unmeasured`` rather than waved through.
    paper_sector_map: Path | None = None

    def ensure(self) -> "ApiSettings":
        for path in (self.runtime_root, self.cache_root, self.jobs_root):
            path.mkdir(parents=True, exist_ok=True)
        return self


def default_settings() -> ApiSettings:
    # Respect the canonical QuantAgent storage resolver, including
    # QUANTAGENT_HOME.  The Web control plane must inspect the same runtime
    # that research/training jobs write instead of silently falling back to a
    # repository-local directory.
    paths = quant_paths()
    sector_map = os.environ.get("QUANTAGENT_PAPER_SECTOR_MAP", "").strip()
    return ApiSettings(
        project_root=PROJECT_ROOT,
        runtime_root=paths.home,
        cache_root=paths.cache / "quant_ui",
        jobs_root=paths.home / "jobs" / "quant_ui",
        paper_sector_map=Path(sector_map).expanduser() if sector_map else None,
    ).ensure()


def safe_project_path(settings: ApiSettings, value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        parts = candidate.parts
        if parts and parts[0] == "runtime":
            candidate = settings.runtime_root.joinpath(*parts[1:])
        else:
            candidate = settings.project_root / candidate
    resolved = candidate.resolve()
    allowed_roots = {
        settings.project_root.resolve(),
        settings.runtime_root.resolve(),
    }
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise ValueError("path is outside the QuantAgent project and runtime roots")
    return resolved


def project_relative(settings: ApiSettings, path: str | Path) -> str:
    resolved = safe_project_path(settings, path)
    runtime_root = settings.runtime_root.resolve()
    if resolved == runtime_root:
        return "runtime"
    if runtime_root in resolved.parents:
        return (Path("runtime") / resolved.relative_to(runtime_root)).as_posix()
    return resolved.relative_to(settings.project_root.resolve()).as_posix()


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{sha1(value.encode('utf-8')).hexdigest()[:16]}"
