from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json

from quantagent.config.paths import quant_paths


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_version: str
    artifact_path: str
    feature_version: str
    metrics: dict[str, float]
    created_at: str
    metadata: dict[str, Any]


class ModelRegistry:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else quant_paths().models / "registry"

    def register(self, model_version: str, feature_version: str, metrics: dict[str, float], metadata: dict[str, Any] | None = None) -> ModelRegistryEntry:
        self.root.mkdir(parents=True, exist_ok=True)
        artifact_path = self.root / f"{model_version}.json"
        entry = ModelRegistryEntry(
            model_version=model_version,
            artifact_path=str(artifact_path),
            feature_version=feature_version,
            metrics=metrics,
            created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            metadata=metadata or {},
        )
        artifact_path.write_text(json.dumps(asdict(entry), indent=2, sort_keys=True), encoding="utf-8")
        (self.root / "latest.json").write_text(json.dumps(asdict(entry), indent=2, sort_keys=True), encoding="utf-8")
        return entry

    def latest(self) -> ModelRegistryEntry | None:
        path = self.root / "latest.json"
        if not path.exists():
            return None
        return ModelRegistryEntry(**json.loads(path.read_text(encoding="utf-8")))
