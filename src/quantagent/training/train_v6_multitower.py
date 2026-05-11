from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quantagent.models.v6_model_system import V6ModelSystem
from quantagent.training.model_registry import ModelRegistry
from quantagent.training.v6_dataset import build_v6_dataset
from quantagent.training.validation_report import build_smoke_validation_report


def train_v6_multitower(
    features: pd.DataFrame,
    config: dict[str, Any] | None = None,
    output_dir: str | Path = "artifacts/models/v6",
    dry_run: bool = True,
) -> dict[str, object]:
    cfg = config or {}
    dataset = build_v6_dataset(features)
    model = V6ModelSystem(
        model_version="v6.dry_run" if dry_run else "v6.multitower",
        feature_version=str(features.get("feature_version", pd.Series(["v6.0"])).iloc[0]) if not features.empty else "v6.0",
    )
    report = build_smoke_validation_report()
    registry = ModelRegistry(cfg.get("model_registry_dir", output_dir))
    entry = registry.register(
        model.model_version,
        model.feature_version,
        report.metrics,
        metadata={
            "dry_run": dry_run,
            "rows": int(len(dataset.frame)),
            "feature_columns": dataset.feature_columns,
            "label_columns": dataset.label_columns,
            "architecture": cfg.get("architecture", "v6_multitower"),
        },
    )
    return {"model_version": entry.model_version, "artifact_path": entry.artifact_path, "metrics": report.metrics, "dry_run": dry_run}

