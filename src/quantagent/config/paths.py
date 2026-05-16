"""Canonical storage layout for QuantAgent V7 real-data assets.

Every large artifact produced by the V7 pipeline — Qlib silver/raw dumps,
AkShare/TuShare PIT caches, trained model checkpoints, predictions,
target weights, walk-forward backtest reports and audit logs — is written
outside the repository under a single configurable root. The default
root on Windows is ``E:\\AI量化\\`` (the Chinese name reads as
"AI Quant"); on other platforms it falls back to ``~/AI_quant``.

Callers should resolve the layout through :func:`quant_paths` and not
hard-code ``data/v7`` style paths. ``QUANTAGENT_HOME`` overrides the
root for the entire process; ``QUANTAGENT_DATA_ROOT`` overrides just
the data tier (raw/silver/gold).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform


DEFAULT_QUANT_HOME_ENV = "QUANTAGENT_HOME"
DEFAULT_DATA_ROOT_ENV = "QUANTAGENT_DATA_ROOT"

_WINDOWS_DEFAULT_HOME = Path("E:/AI\u91cf\u5316")
_POSIX_DEFAULT_HOME = Path.home() / "AI_quant"


@dataclass(frozen=True)
class QuantPaths:
    """Resolved layout of the large-asset storage tree.

    ``home`` is the root all other paths sit under. ``data_root`` is split
    into ``raw / silver / gold`` to match the medallion layout already used
    by :mod:`quantagent.data.lake`; ``models``, ``predictions``,
    ``target_weights``, ``reports`` and ``logs`` are sibling directories.
    """

    home: Path
    data_root: Path
    raw: Path
    silver: Path
    gold: Path
    models: Path
    predictions: Path
    target_weights: Path
    reports: Path
    logs: Path
    cache: Path

    def ensure(self) -> "QuantPaths":
        for path in (
            self.home,
            self.data_root,
            self.raw,
            self.silver,
            self.gold,
            self.models,
            self.predictions,
            self.target_weights,
            self.reports,
            self.logs,
            self.cache,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict[str, str]:
        return {
            "home": str(self.home),
            "data_root": str(self.data_root),
            "raw": str(self.raw),
            "silver": str(self.silver),
            "gold": str(self.gold),
            "models": str(self.models),
            "predictions": str(self.predictions),
            "target_weights": str(self.target_weights),
            "reports": str(self.reports),
            "logs": str(self.logs),
            "cache": str(self.cache),
        }


def resolve_quant_home(override: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the QuantAgent home directory.

    Priority order:

    1. Explicit ``override`` argument.
    2. ``QUANTAGENT_HOME`` environment variable.
    3. ``E:\\AI量化`` on Windows.
    4. ``~/AI_quant`` on POSIX systems.
    """
    if override is not None:
        return Path(override).expanduser()
    env_home = os.environ.get(DEFAULT_QUANT_HOME_ENV)
    if env_home:
        return Path(env_home).expanduser()
    if platform.system() == "Windows":
        return _WINDOWS_DEFAULT_HOME
    return _POSIX_DEFAULT_HOME


def quant_paths(
    home: str | os.PathLike[str] | None = None,
    data_root: str | os.PathLike[str] | None = None,
) -> QuantPaths:
    """Build a :class:`QuantPaths` view of the canonical layout.

    The function never creates directories on its own; call ``ensure()``
    on the returned object when a caller actually intends to write.
    """
    home_path = resolve_quant_home(home)
    if data_root is not None:
        data_path = Path(data_root).expanduser()
    else:
        env_data = os.environ.get(DEFAULT_DATA_ROOT_ENV)
        data_path = Path(env_data).expanduser() if env_data else home_path / "data"
    return QuantPaths(
        home=home_path,
        data_root=data_path,
        raw=data_path / "raw",
        silver=data_path / "silver",
        gold=data_path / "gold",
        models=home_path / "models",
        predictions=home_path / "predictions",
        target_weights=home_path / "target_weights",
        reports=home_path / "reports",
        logs=home_path / "logs",
        cache=home_path / "cache",
    )


__all__ = [
    "DEFAULT_QUANT_HOME_ENV",
    "DEFAULT_DATA_ROOT_ENV",
    "QuantPaths",
    "quant_paths",
    "resolve_quant_home",
]
